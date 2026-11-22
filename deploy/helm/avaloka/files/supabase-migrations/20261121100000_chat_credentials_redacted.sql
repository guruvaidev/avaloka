-- Credentials typed into chat are stored redacted in public.analysis_messages.
--
-- 1. public.redact_chat_credentials(text): the chat's credential detection
--    (app/core/typed_secrets.py: the seal's rules when the message reads as
--    connection details, and "password=x" / "pwd: x" / a URI's user:pass@ in
--    any message), in SQL. Only each value changes, so the message keeps its
--    meaning: "... password hunter2" becomes "... password [redacted]".
--    tests/test_chat_credentials_redacted_migration.py holds it to the Python
--    function, case by case.
-- 2. A trigger applies it to every user message as it is written or edited.
-- 3. Messages already stored are redacted the same way. Nothing else in a row
--    changes, and it logs a count, never content.
--
-- Safe to run again: a redacted message is left as it is.

CREATE OR REPLACE FUNCTION public._redact_chat_credential_values(t text, pattern text, flags text, kv boolean)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
SET search_path = ''
AS $fn$
DECLARE
  o text;
  c text;
  k int := 0;
  marked text;
  done text := '';
  i int;
  j int;
  v text;
  core text;
  tail text;
BEGIN
  IF t IS NULL OR t = '' THEN
    RETURN t;
  END IF;
  -- Two private-use characters the text does not contain mark each value.
  LOOP
    o := chr(57344 + 2 * k);
    c := chr(57345 + 2 * k);
    EXIT WHEN strpos(t, o) = 0 AND strpos(t, c) = 0;
    k := k + 1;
  END LOOP;
  IF kv THEN
    -- Groups: 1 the label and separator; 2-3 "value"; 4-5 'value'; 6-7 `value`; 8 a bare value.
    marked := regexp_replace(t, pattern, '\1\2\4\6' || o || '\3\5\7\8' || c || '\2\4\6', flags);
  ELSE
    marked := regexp_replace(t, pattern, '\1' || o || '\2' || c, flags);
  END IF;
  IF marked = t THEN
    RETURN t;
  END IF;
  LOOP
    i := strpos(marked, o);
    EXIT WHEN i = 0;
    j := strpos(marked, c);
    done := done || left(marked, i - 1);
    v := substr(marked, i + 1, j - i - 1);
    core := v;
    tail := '';
    -- A bare value ends before trailing sentence punctuation; a quoted one is taken whole.
    IF kv AND right(done, 1) NOT IN ('"', '''', '`') THEN
      core := regexp_replace(v, '[.,;:!?)]+$', '');
      IF core = '' THEN
        core := left(v, 1);
      END IF;
      tail := substr(v, length(core) + 1);
    END IF;
    IF lower(core) IN ('null', 'none', 'empty', 'missing', 'blank', 'set', 'required', 'the', 'a', 'an',
                       'and', 'or', 'for', 'to', 'of', 'in', 'column', 'field', 'is', 'with', 'auth',
                       'authentication', 'protected', 'reset', 'changed', '[redacted]', '<redacted>') THEN
      done := done || core || tail;
    ELSE
      done := done || '[redacted]' || tail;
    END IF;
    marked := substr(marked, j + 1);
  END LOOP;
  RETURN done || marked;
END
$fn$;

CREATE OR REPLACE FUNCTION public.redact_chat_credentials(t text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
SET search_path = ''
AS $fn$
DECLARE
  sep constant text := $re$(?:[[:space:]]*(?:[:=]|=>|\yis\y)[[:space:]]*|[[:space:]]+)$re$;
  val constant text := $re$(?:(")([^"\n⟦]+)"|(')([^'\n⟦]+)'|(`)([^`\n⟦]+)`|([^[:space:]"'`⟦]+)(?=[[:space:]]|$))$re$;
  uri constant text := $re$([a-z][a-z0-9+.-]*://[^[:space:]/:@]+:)([^[:space:]@⟦][^[:space:]@]*)(?=@)$re$;
  out text;
BEGIN
  IF t IS NULL OR t = '' THEN
    RETURN t;
  END IF;
  out := regexp_replace(t, $re$⟦sealed:[a-z_]+:[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]+⟧$re$, '[redacted]', 'g');
  IF t ~* $re$\y(?:host(?:name)?|server|port|bucket|container|database|dsn|connect\w*|regist\w*|endpoint|service[ _]account|storage[ _]account|access[ _]key|credential\w*|sas)\y|://$re$ THEN
    out := public._redact_chat_credential_values(out, uri, 'gi', false);
    out := public._redact_chat_credential_values(out,
      $re$("private_key"[[:space:]]*:[[:space:]]*")((?:[^"\\⟦]|\\.)(?:[^"\\]|\\.)*)(?=")$re$, 'g', false);
    out := public._redact_chat_credential_values(out,
      $re$(\y(?:aws[_ ]secret[_ ]access[_ ]key|secret(?:[_ ]access)?[_ ]key|secret)\y$re$ || sep || ')' || val, 'gi', true);
    out := public._redact_chat_credential_values(out,
      $re$(\y(?:aws[_ ]access[_ ]key[_ ]id|access[_ ]key(?:[_ ]id)?)\y$re$ || sep || ')' || val, 'gi', true);
    out := public._redact_chat_credential_values(out,
      $re$(\y(?:az[_ ]sas[_ ]token|sas[_ ]token|sas)\y$re$ || sep || ')' || val, 'gi', true);
    out := public._redact_chat_credential_values(out,
      $re$(\y(?:az[_ ]access[_ ]key|account[_ ]key)\y$re$ || sep || ')' || val, 'gi', true);
    out := public._redact_chat_credential_values(out,
      $re$(\y(?:password|passwd|pwd|pass)\y$re$ || sep || ')' || val, 'gi', true);
  END IF;
  out := public._redact_chat_credential_values(out, uri, 'gi', false);
  out := public._redact_chat_credential_values(out,
    $re$(\y(?:password|passwd|pwd)\y[[:space:]]*(?:=>|[:=])[[:space:]]*)$re$ || val, 'gi', true);
  RETURN out;
END
$fn$;

GRANT EXECUTE ON FUNCTION public._redact_chat_credential_values(text, text, text, boolean) TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.redact_chat_credentials(text) TO anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION public.analysis_messages_redact_credentials()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $fn$
BEGIN
  IF NEW.role = 'user' THEN
    NEW.content := public.redact_chat_credentials(NEW.content);
  END IF;
  RETURN NEW;
END
$fn$;

DROP TRIGGER IF EXISTS analysis_messages_redact_credentials ON public.analysis_messages;
CREATE TRIGGER analysis_messages_redact_credentials
  BEFORE INSERT OR UPDATE OF content, role ON public.analysis_messages
  FOR EACH ROW EXECUTE FUNCTION public.analysis_messages_redact_credentials();

DO $scrub$
DECLARE
  n bigint;
BEGIN
  UPDATE public.analysis_messages AS m
     SET content = public.redact_chat_credentials(m.content)
   WHERE m.role = 'user'
     AND m.content IS DISTINCT FROM public.redact_chat_credentials(m.content);
  GET DIAGNOSTICS n = ROW_COUNT;
  RAISE NOTICE 'analysis_messages: credentials redacted in % stored user message(s)', n;
END
$scrub$;
