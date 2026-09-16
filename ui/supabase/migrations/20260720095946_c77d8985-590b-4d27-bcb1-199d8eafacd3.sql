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