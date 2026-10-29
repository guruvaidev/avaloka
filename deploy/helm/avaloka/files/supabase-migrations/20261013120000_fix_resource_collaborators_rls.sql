-- =====================================================================
-- Fix "infinite recursion detected in policy" when adding collaborators.
--
-- Why:
--  * projects has TWO collaborator SELECT policies doing the same thing:
--      "Collaborators can view shared projects"  -> is_resource_collaborator()  (SECURITY DEFINER, safe)
--      "collaborators read shared projects"      -> raw subquery on resource_collaborators (RLS applies)
--    The second one makes every read of projects evaluate resource_collaborators'
--    own policies, which read reports / app_users / profiles, which read back.
--  * The four "Report creators ..." policies on resource_collaborators run a
--    large subquery over reports / app_users / profiles with RLS on.
--
-- Fix:
--  1. Drop the duplicate raw-subquery projects policy (the definer-based one stays).
--  2. Move the "Report creators" check into one SECURITY DEFINER function, so
--     the policies on resource_collaborators no longer read other RLS tables.
--  3. Remove privileges anon never needs on this table.
--
-- Safe to run more than once. Rollback section at the bottom.
-- =====================================================================

begin;

-- 1. Duplicate projects policy (raw read of resource_collaborators under RLS)
drop policy if exists "collaborators read shared projects" on public.projects;

-- Make sure the definer-based replacement exists (it does today; this keeps a
-- fresh database, e.g. local Kubernetes, identical).
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'projects'
      and policyname = 'Collaborators can view shared projects'
  ) then
    create policy "Collaborators can view shared projects"
      on public.projects for select to authenticated
      using (public.is_resource_collaborator('project', id));
  end if;
end $$;

-- 2. "Report creators" check as one SECURITY DEFINER function.
--    Same logic as the current policies: the caller created the report (by
--    auth uid, profile id or app_user id), and the invited app_user belongs to
--    the report's organization.
create or replace function public.can_share_report(p_report_id uuid, p_app_user_id uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1
    from public.reports r
    join public.app_users invited_user on invited_user.id = p_app_user_id
    where r.id = p_report_id
      and invited_user.organization_id = r.organization_id
      and (
        r.created_by = auth.uid()
        or exists (
          select 1 from public.profiles creator_profile
          where (creator_profile.id = r.created_by or creator_profile.user_id = r.created_by)
            and (creator_profile.user_id = auth.uid() or creator_profile.id = auth.uid())
        )
        or exists (
          select 1 from public.app_users creator_user
          where (creator_user.id = r.created_by
                 or creator_user.auth_user_id = r.created_by
                 or creator_user.profile_id = r.created_by)
            and (creator_user.auth_user_id = auth.uid() or creator_user.profile_id = auth.uid())
        )
      )
  );
$$;

revoke all on function public.can_share_report(uuid, uuid) from public, anon;
grant execute on function public.can_share_report(uuid, uuid) to authenticated, service_role;

drop policy if exists "Report creators read collaborators"   on public.resource_collaborators;
drop policy if exists "Report creators insert collaborators" on public.resource_collaborators;
drop policy if exists "Report creators update collaborators" on public.resource_collaborators;
drop policy if exists "Report creators delete collaborators" on public.resource_collaborators;

create policy "Report creators read collaborators"
  on public.resource_collaborators for select to authenticated
  using (resource_type = 'report' and invited_by = auth.uid()
         and public.can_share_report(resource_id, app_user_id));

create policy "Report creators insert collaborators"
  on public.resource_collaborators for insert to authenticated
  with check (resource_type = 'report' and invited_by = auth.uid()
              and public.can_share_report(resource_id, app_user_id));

create policy "Report creators update collaborators"
  on public.resource_collaborators for update to authenticated
  using (resource_type = 'report' and invited_by = auth.uid()
         and public.can_share_report(resource_id, app_user_id))
  with check (resource_type = 'report' and invited_by = auth.uid()
              and public.can_share_report(resource_id, app_user_id));

create policy "Report creators delete collaborators"
  on public.resource_collaborators for delete to authenticated
  using (resource_type = 'report' and invited_by = auth.uid()
         and public.can_share_report(resource_id, app_user_id));

-- The "owners ... resource collaborators" policies already use the
-- SECURITY DEFINER can_manage_resource_share(); they are left unchanged.

-- 3. anon never reads or writes collaborators (RLS does not apply to TRUNCATE).
revoke all on table public.resource_collaborators from anon;

commit;

-- =====================================================================
-- ROLLBACK (run manually only if needed)
-- =====================================================================
-- begin;
-- create policy "collaborators read shared projects" on public.projects
--   for select to authenticated
--   using (exists (select 1 from public.resource_collaborators rc
--                  join public.app_users au on au.id = rc.app_user_id
--                  where rc.resource_type = 'project' and rc.resource_id = projects.id
--                    and au.auth_user_id = auth.uid()));
-- -- re-create the four original "Report creators ..." policies from the
-- -- pg_policies export taken on 2026-10-13, then:
-- drop function if exists public.can_share_report(uuid, uuid);
-- grant all on table public.resource_collaborators to anon;
-- commit;
