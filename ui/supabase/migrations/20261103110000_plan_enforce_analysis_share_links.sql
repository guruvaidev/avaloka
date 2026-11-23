-- NEW-BG-01 follow-up, server side: analysis share links need a Professional
-- plan or higher.
--
-- Owner ruling 2026-11-02: analysis share links are Professional or higher
-- (share_analysis in app/core/editions.py). Report share links stay open on
-- every plan: Free shares reports by email and public link.
--
-- Needs 20261101140000_plan_rules_pro_collab.sql (public.collab_features_enabled()
-- and avaloka_meta.audit_org_feature_bypass()). Follows
-- 20261103100000_plan_enforce_comments_notifications.sql, which left this table
-- for this ruling.
--
-- resource_share_links carries project, analysis and report links in one table
-- (resource_type). The RESTRICTIVE policies below are ANDed with the existing
-- owner policies and apply ONLY to rows whose resource_type is 'analysis':
--   INSERT  a new analysis link needs Professional or higher.
--   UPDATE  changing an analysis link (access level, public/private), or turning
--           another row into one, needs Professional or higher.
-- Not changed: SELECT and DELETE. Links already created on a lower plan keep
-- resolving (the token resolvers read with the service role, and owners can
-- still read them), and an owner can still revoke one. No row is deleted.
-- Project and report links are not affected.
--
-- App-admin bypass: collab_features_enabled() lets an app admin through; each
-- such write to an analysis link is recorded by the #661 audit trigger.
--
-- Idempotent (DROP ... IF EXISTS before each CREATE). Not in the frozen baseline.

DROP POLICY IF EXISTS "pro plan to add analysis share links" ON public.resource_share_links;
CREATE POLICY "pro plan to add analysis share links" ON public.resource_share_links
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (resource_type IS DISTINCT FROM 'analysis' OR public.collab_features_enabled());
DROP POLICY IF EXISTS "pro plan to change analysis share links" ON public.resource_share_links;
CREATE POLICY "pro plan to change analysis share links" ON public.resource_share_links
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (resource_type IS DISTINCT FROM 'analysis' OR public.collab_features_enabled())
  WITH CHECK (resource_type IS DISTINCT FROM 'analysis' OR public.collab_features_enabled());

-- Audit of the app-admin bypass, for analysis links only (other links need no plan).
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.resource_share_links;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.resource_share_links
  FOR EACH ROW WHEN (NEW.resource_type = 'analysis')
  EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('professional');
