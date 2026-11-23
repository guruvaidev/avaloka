-- Object-store credentials are sealed, never plaintext (D15.3, 15c).
--
-- public.cloud_datasets accepted plaintext access_key/secret_key: its CHECK
-- (check_credentials_format, 0-schema.sql) allowed either the plaintext pair
-- or the ciphertext pair. From here on only the sealed pair is accepted:
--
-- 1. check_credentials_format is dropped and check_credentials_sealed added
--    NOT VALID: every INSERT and UPDATE must carry both *_ciphertext and both
--    *_iv columns and no plaintext, while rows written before this migration
--    are left as they are until the backend seals them.
-- 2. The backend seals the existing plaintext rows under DB_ENCRYPTION_KEY
--    (app/api/cloud_connections.py: seal_plaintext_rows, run at API start-up
--    and by `python -m app.api.cloud_connections seal`). Postgres cannot do
--    that itself: the ciphertext is AES-GCM under SHA-256(DB_ENCRYPTION_KEY),
--    which the database never holds. A later migration runs
--    ALTER TABLE public.cloud_datasets VALIDATE CONSTRAINT check_credentials_sealed
--    once an install reports zero plaintext rows.
-- 3. encrypt_cloud_credentials(uuid, text, text) stored whatever text it was
--    handed as "ciphertext", without an IV, and was callable by anon. It is
--    replaced by a function that refuses: sealing is the backend's job
--    (POST /api/cloud-connections), and the UI no longer writes keys itself.
--
-- Safe to run again: a validated check_credentials_sealed stays validated.

ALTER TABLE public.cloud_datasets DROP CONSTRAINT IF EXISTS check_credentials_format;
-- Added once. A re-run must not drop and re-add it: that would reset a
-- constraint the operator has already validated back to NOT VALID.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'check_credentials_sealed' AND conrelid = 'public.cloud_datasets'::regclass
  ) THEN
    ALTER TABLE public.cloud_datasets ADD CONSTRAINT check_credentials_sealed CHECK (
      access_key IS NULL AND secret_key IS NULL
      AND access_key_ciphertext IS NOT NULL AND access_key_iv IS NOT NULL
      AND secret_key_ciphertext IS NOT NULL AND secret_key_iv IS NOT NULL
    ) NOT VALID;
  END IF;
END $$;

CREATE OR REPLACE FUNCTION public.encrypt_cloud_credentials(p_connection_id uuid, p_access_key text, p_secret_key text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  RAISE EXCEPTION 'cloud credentials are sealed by the backend (POST /api/cloud-connections); this function stored unsealed text and is retired'
    USING ERRCODE = 'feature_not_supported';
END;
$$;
REVOKE ALL ON FUNCTION public.encrypt_cloud_credentials(uuid, text, text) FROM PUBLIC, anon;
