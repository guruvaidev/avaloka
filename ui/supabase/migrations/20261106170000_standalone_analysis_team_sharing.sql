-- Profile/team sharing for standalone analyses. This stays separate from
-- resource_collaborators, whose recipients are organization app_users.
CREATE TABLE IF NOT EXISTS public.analysis_shares (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id uuid NOT NULL REFERENCES public.analyses(id) ON DELETE CASCADE,
  recipient_profile_id uuid REFERENCES public.profiles(id) ON DELETE CASCADE,
  team_id uuid REFERENCES public.teams(id) ON DELETE CASCADE,
  access_level text NOT NULL DEFAULT 'view'
    CHECK (access_level IN ('view', 'comment', 'edit')),
  can_share boolean NOT NULL DEFAULT false,
  shared_by uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT analysis_shares_exactly_one_recipient
    CHECK ((recipient_profile_id IS NOT NULL) <> (team_id IS NOT NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_analysis_shares_profile
  ON public.analysis_shares(analysis_id, recipient_profile_id)
  WHERE recipient_profile_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_analysis_shares_team
  ON public.analysis_shares(analysis_id, team_id)
  WHERE team_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analysis_shares_profile
  ON public.analysis_shares(recipient_profile_id)
  WHERE recipient_profile_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analysis_shares_team
  ON public.analysis_shares(team_id)
  WHERE team_id IS NOT NULL;

ALTER TABLE public.analysis_shares ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.analysis_shares FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.analysis_shares TO service_role;

CREATE OR REPLACE FUNCTION public.ensure_standalone_analysis_share()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_project_id uuid;
BEGIN
  SELECT project_id INTO v_project_id
    FROM public.analyses
   WHERE id = NEW.analysis_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Analysis not found.' USING ERRCODE = '23503';
  END IF;
  IF v_project_id IS NOT NULL THEN
    RAISE EXCEPTION 'Only standalone analyses can be shared with Professional teams.' USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS ensure_standalone_analysis_share ON public.analysis_shares;
CREATE TRIGGER ensure_standalone_analysis_share
  BEFORE INSERT OR UPDATE OF analysis_id ON public.analysis_shares
  FOR EACH ROW EXECUTE FUNCTION public.ensure_standalone_analysis_share();

CREATE OR REPLACE FUNCTION public.analysis_owner_profile(p_analysis_id uuid)
RETURNS uuid
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT owner_id FROM public.analyses WHERE id = p_analysis_id
$$;

CREATE OR REPLACE FUNCTION public.protect_shared_analysis_identity()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  IF NEW.id IS DISTINCT FROM OLD.id THEN
    RAISE EXCEPTION 'Analysis IDs cannot be changed.' USING ERRCODE = '42501';
  END IF;
  IF NEW.owner_id IS DISTINCT FROM OLD.owner_id
     AND public.current_profile_id() IS DISTINCT FROM OLD.owner_id THEN
    RAISE EXCEPTION 'Only the analysis owner can change its owner.' USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS protect_shared_analysis_identity ON public.analyses;
CREATE TRIGGER protect_shared_analysis_identity
  BEFORE UPDATE OF id, owner_id ON public.analyses
  FOR EACH ROW EXECUTE FUNCTION public.protect_shared_analysis_identity();

-- 0 = no access, 1 = view, 2 = comment, 3 = edit. Team grants are dynamic:
-- team owners, leads, and current memberships inherit the team's grant.
CREATE OR REPLACE FUNCTION public.analysis_share_access_level(p_analysis_id uuid)
RETURNS integer
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  WITH caller AS (
    SELECT public.current_profile_id() AS profile_id
  ), requested AS (
    SELECT id, coalesce(parent_analysis_id, id) AS root_id, owner_id
      FROM public.analyses
     WHERE id = p_analysis_id
  ), accessible_teams AS (
    SELECT tm.team_id
      FROM public.team_memberships tm, caller c
     WHERE tm.user_id = c.profile_id
    UNION
    SELECT t.id
      FROM public.teams t, caller c
     WHERE t.owner_profile_id = c.profile_id
        OR t.lead_profile_id = c.profile_id
  ), grants AS (
    SELECT 3 AS level
      FROM requested r, caller c
     WHERE c.profile_id IS NOT NULL
       AND r.owner_id = c.profile_id
    UNION ALL
    SELECT CASE s.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END
      FROM public.analysis_shares s, requested r, caller c
     WHERE s.analysis_id IN (r.id, r.root_id)
       AND (s.recipient_profile_id = c.profile_id
         OR s.team_id IN (SELECT team_id FROM accessible_teams))
    UNION ALL
    SELECT CASE rc.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END
      FROM public.resource_collaborators rc
      JOIN public.app_users au ON au.id = rc.app_user_id
      JOIN requested r ON rc.resource_id IN (r.id, r.root_id)
     WHERE rc.resource_type = 'analysis'
       AND au.auth_user_id = auth.uid()
  )
  SELECT coalesce(max(level), 0)::integer FROM grants
$$;

CREATE OR REPLACE FUNCTION public.can_share_analysis(p_analysis_id uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  WITH caller AS (SELECT public.current_profile_id() AS profile_id),
  requested AS (
    SELECT id, coalesce(parent_analysis_id, id) AS root_id, owner_id
      FROM public.analyses WHERE id = p_analysis_id
  ),
  accessible_teams AS (
    SELECT tm.team_id FROM public.team_memberships tm, caller c WHERE tm.user_id = c.profile_id
    UNION
    SELECT t.id FROM public.teams t, caller c
     WHERE t.owner_profile_id = c.profile_id OR t.lead_profile_id = c.profile_id
  )
  SELECT EXISTS (
    SELECT 1 FROM requested r, caller c WHERE r.owner_id = c.profile_id
  ) OR EXISTS (
    SELECT 1
      FROM public.analysis_shares s, requested r, caller c
     WHERE s.analysis_id IN (r.id, r.root_id)
       AND s.can_share
       AND (s.recipient_profile_id = c.profile_id
         OR s.team_id IN (SELECT team_id FROM accessible_teams))
  )
$$;

REVOKE ALL ON FUNCTION public.analysis_owner_profile(uuid) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.analysis_share_access_level(uuid) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.can_share_analysis(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.analysis_owner_profile(uuid) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.analysis_share_access_level(uuid) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.can_share_analysis(uuid) TO authenticated, service_role;

-- These add read/write routes for profile/team grants without changing the
-- existing owner and enterprise app_user collaborator policies.
DROP POLICY IF EXISTS "profile shares read standalone analyses" ON public.analyses;
CREATE POLICY "profile shares read standalone analyses" ON public.analyses
  FOR SELECT TO authenticated
  USING (project_id IS NULL AND public.analysis_share_access_level(id) >= 1);

DROP POLICY IF EXISTS "profile shares update standalone analyses" ON public.analyses;
CREATE POLICY "profile shares update standalone analyses" ON public.analyses
  FOR UPDATE TO authenticated
  USING (project_id IS NULL AND public.analysis_share_access_level(id) >= 3)
  WITH CHECK (
    project_id IS NULL
    AND owner_id = public.analysis_owner_profile(id)
  );

DROP POLICY IF EXISTS "profile shares read analysis messages" ON public.analysis_messages;
CREATE POLICY "profile shares read analysis messages" ON public.analysis_messages
  FOR SELECT TO authenticated
  USING (public.analysis_share_access_level(analysis_id) >= 1);

DROP POLICY IF EXISTS "editors append shared analysis messages" ON public.analysis_messages;
CREATE POLICY "editors append shared analysis messages" ON public.analysis_messages
  FOR INSERT TO authenticated
  WITH CHECK (
    ((role = 'user' AND author_id = public.current_profile_id())
      OR (role = 'assistant' AND author_id IS NULL))
    AND public.analysis_share_access_level(analysis_id) >= 3
  );

DROP POLICY IF EXISTS "profile shares read analysis comments" ON public.analysis_comments;
CREATE POLICY "profile shares read analysis comments" ON public.analysis_comments
  FOR SELECT TO authenticated
  USING (public.analysis_share_access_level(analysis_id) >= 1);

DROP POLICY IF EXISTS "commenters add shared analysis comments" ON public.analysis_comments;
CREATE POLICY "commenters add shared analysis comments" ON public.analysis_comments
  FOR INSERT TO authenticated
  WITH CHECK (
    author_id = public.current_profile_id()
    AND public.collab_features_enabled()
    AND public.analysis_share_access_level(analysis_id) >= 2
  );

CREATE OR REPLACE FUNCTION public.set_analysis_shares_updated_at()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS set_analysis_shares_updated_at ON public.analysis_shares;
CREATE TRIGGER set_analysis_shares_updated_at
  BEFORE UPDATE ON public.analysis_shares
  FOR EACH ROW EXECUTE FUNCTION public.set_analysis_shares_updated_at();
