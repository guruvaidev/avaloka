-- Store a profile as each team's lead. Existing teams inherit their owner.
ALTER TABLE public.teams
  ADD COLUMN IF NOT EXISTS lead_profile_id uuid;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
     WHERE table_schema = 'public' AND table_name = 'teams'
       AND column_name = 'owner_profile_id'
  ) THEN
    EXECUTE 'UPDATE public.teams SET lead_profile_id = owner_profile_id WHERE lead_profile_id IS NULL';
  ELSIF EXISTS (
    SELECT 1 FROM information_schema.columns
     WHERE table_schema = 'public' AND table_name = 'teams'
       AND column_name = 'customer_account_id'
  ) THEN
    EXECUTE $sql$
      UPDATE public.teams t
         SET lead_profile_id = ca.owner_user_id
        FROM public.customer_accounts ca
       WHERE ca.id = t.customer_account_id
         AND t.lead_profile_id IS NULL
    $sql$;
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
     WHERE conname = 'teams_lead_profile_id_fkey'
       AND conrelid = 'public.teams'::regclass
  ) THEN
    ALTER TABLE public.teams
      ADD CONSTRAINT teams_lead_profile_id_fkey
      FOREIGN KEY (lead_profile_id) REFERENCES public.profiles(id) ON DELETE SET NULL;
  END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_teams_lead_profile_id
  ON public.teams(lead_profile_id);
