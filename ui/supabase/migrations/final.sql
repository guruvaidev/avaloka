CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public', 'auth'
AS $function$
DECLARE
  v_org uuid;
  v_full_name text;
BEGIN
  v_org := NULLIF(
    NEW.raw_user_meta_data->>'organization_id',
    ''
  )::uuid;

  v_full_name := NEW.raw_user_meta_data->>'full_name';

  INSERT INTO public.profiles (
    user_id,
    organization_id,
    full_name
  )
  VALUES (
    NEW.id,
    COALESCE(v_org, gen_random_uuid()),
    v_full_name
  )
  ON CONFLICT (user_id) DO UPDATE
    SET organization_id = COALESCE(
          public.profiles.organization_id,
          EXCLUDED.organization_id
        ),
        full_name = COALESCE(
          public.profiles.full_name,
          EXCLUDED.full_name
        );

  UPDATE public.app_users
  SET
    auth_user_id = NEW.id,
    accepted_at = now(),
    status = 'Active'
  WHERE lower(email) = lower(NEW.email);

  RETURN NEW;
END;
$function$;


INSERT INTO storage.buckets (id, name, public)
VALUES ('avatars', 'avatars', true)
ON CONFLICT (id) DO NOTHING;


-- Polices for user profiles

CREATE POLICY "Users upload own avatar" ON storage.objects
  FOR INSERT TO authenticated WITH CHECK (
    bucket_id = 'avatars' AND (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users update own avatar" ON storage.objects
  FOR UPDATE TO authenticated USING (
    bucket_id = 'avatars' AND (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users delete own avatar" ON storage.objects
  FOR DELETE TO authenticated USING (
    bucket_id = 'avatars' AND (storage.foldername(name))[1] = auth.uid()::text
  );

DROP POLICY IF EXISTS "Avatars are publicly readable" ON storage.objects;
CREATE POLICY "Users read own avatar"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'avatars' AND (storage.foldername(name))[1] = (auth.uid())::text);

DROP POLICY IF EXISTS "Organization members can read member avatars" ON storage.objects;

CREATE POLICY "Organization members can read member avatars"
ON storage.objects
FOR SELECT
TO authenticated
USING (
  bucket_id = 'avatars'
  AND EXISTS (
    SELECT 1
    FROM public.profiles viewer
    JOIN public.profiles avatar_owner
      ON avatar_owner.organization_id = viewer.organization_id
    WHERE (viewer.user_id = auth.uid() OR viewer.id = auth.uid())
      AND (
        avatar_owner.user_id::text = (storage.foldername(storage.objects.name))[1]
        OR avatar_owner.id::text = (storage.foldername(storage.objects.name))[1]
      )
  )
);