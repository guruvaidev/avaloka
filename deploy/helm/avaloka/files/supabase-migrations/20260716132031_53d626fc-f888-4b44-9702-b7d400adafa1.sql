DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'analysis_messages'
      AND policyname = 'collaborators read shared analysis messages'
  ) THEN
    CREATE POLICY "collaborators read shared analysis messages"
      ON public.analysis_messages
      FOR SELECT
      TO authenticated
      USING (
        EXISTS (
          SELECT 1
          FROM public.analyses a
          WHERE a.id = analysis_messages.analysis_id
            AND (
              public.is_resource_collaborator('analysis', a.id)
              OR (
                a.project_id IS NOT NULL
                AND public.is_resource_collaborator('project', a.project_id)
              )
            )
        )
      );
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'analysis_messages'
      AND policyname = 'collaborators insert shared analysis messages'
  ) THEN
    CREATE POLICY "collaborators insert shared analysis messages"
      ON public.analysis_messages
      FOR INSERT
      TO authenticated
      WITH CHECK (
        EXISTS (
          SELECT 1
          FROM public.analyses a
          WHERE a.id = analysis_messages.analysis_id
            AND (
              public.is_resource_collaborator('analysis', a.id)
              OR (
                a.project_id IS NOT NULL
                AND public.is_resource_collaborator('project', a.project_id)
              )
            )
        )
      );
  END IF;
END $$;