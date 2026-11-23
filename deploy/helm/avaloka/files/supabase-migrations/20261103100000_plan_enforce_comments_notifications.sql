-- NEW-BG-01 follow-up, server side: commenting, and notifying other people,
-- need a Professional plan or higher.
--
-- Owner ruling 2026-11-01: share_analysis, comment_on_analysis and notifications
-- are Professional and Enterprise (app/core/editions.py). Free shares reports by
-- email and public link, and nothing else. Enforcement approved as its own 1.6
-- security change.
--
-- Needs 20261101140000_plan_rules_pro_collab.sql, which defines
-- public.collab_features_enabled() (Professional or higher, or the audited
-- app-admin bypass) and avaloka_meta.audit_org_feature_bypass().
--
-- RESTRICTIVE policies are ANDed with every permissive policy on the table, so
-- the plan check holds whichever permissive policy would have let the row in.
--
--   analysis_comments, report_comments
--     INSERT and UPDATE need Professional or higher. SELECT and DELETE are
--     unchanged: comments already written stay readable, and an author can
--     still remove their own after a plan lapses.
--
--   notifications
--     INSERT of a notification addressed to SOMEONE ELSE needs Professional or
--     higher. A notification to yourself (recipient_id = actor_id, the app's own
--     "created" / "saved" notices) stays open on every plan. Reading, marking
--     read and deleting your own notifications is unchanged, so a Free user
--     still receives what reaches them: a collaborator invitation from a
--     Professional user, comments and mentions from Professional users on work
--     shared with them, and the backend's analysis and training
--     completed / failed notices (written with the service role, which RLS does
--     not apply to).
--
-- Not changed here: resource_share_links. One table carries both report and
-- analysis share links (resource_type), and Free keeps public report links;
-- that split is pending a ruling.
--
-- Server functions that write with the admin (service-role) client bypass RLS,
-- so each makes the same check itself (ui/src/lib/org-features.ts,
-- assertPlanFeature(..., "professional", ...)).
--
-- App-admin bypass: collab_features_enabled() lets an app admin through; every
-- such write is recorded by the same audit trigger #661 uses.
--
-- Idempotent (DROP ... IF EXISTS before each CREATE). Not in the frozen baseline.

-- ---- comments ---------------------------------------------------------------
DROP POLICY IF EXISTS "pro plan to add analysis comments" ON public.analysis_comments;
CREATE POLICY "pro plan to add analysis comments" ON public.analysis_comments
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.collab_features_enabled());
DROP POLICY IF EXISTS "pro plan to change analysis comments" ON public.analysis_comments;
CREATE POLICY "pro plan to change analysis comments" ON public.analysis_comments
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.collab_features_enabled())
  WITH CHECK (public.collab_features_enabled());

DROP POLICY IF EXISTS "pro plan to add report comments" ON public.report_comments;
CREATE POLICY "pro plan to add report comments" ON public.report_comments
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.collab_features_enabled());
DROP POLICY IF EXISTS "pro plan to change report comments" ON public.report_comments;
CREATE POLICY "pro plan to change report comments" ON public.report_comments
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.collab_features_enabled())
  WITH CHECK (public.collab_features_enabled());

-- ---- notifications ----------------------------------------------------------
DROP POLICY IF EXISTS "pro plan to notify others" ON public.notifications;
CREATE POLICY "pro plan to notify others" ON public.notifications
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (recipient_id = actor_id OR public.collab_features_enabled());

-- ---- audit of the app-admin bypass -------------------------------------------
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.analysis_comments;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.analysis_comments
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('professional');
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.report_comments;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.report_comments
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('professional');
-- A notification to yourself needs no plan, so it is not a bypass.
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.notifications;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT ON public.notifications
  FOR EACH ROW WHEN (NEW.recipient_id IS DISTINCT FROM NEW.actor_id)
  EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('professional');
