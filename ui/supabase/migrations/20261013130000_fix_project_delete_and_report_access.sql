-- =====================================================================
-- Follow-up to 20261013120000_fix_resource_collaborators_rls.sql
--
-- 1. projects delete: "owners and collaborators delete projects" let ANY
--    collaborator delete a project, including view-only and comment-only
--    collaborators. Now only the owner, or a collaborator with
--    access_level = 'edit', can delete.
--
-- 2. get_my_report_access(): shared reports failed for every non-creator:
--    a) it calls public.has_role(...), which does not exist in this database
--       ("function public.has_role(uuid, app_role) does not exist");
--    b) it is declared STABLE but runs an UPDATE (the app_users backfill),
--       which Postgres refuses ("UPDATE is not allowed in a non-volatile
--       function").
--    Fixed by making it VOLATILE and calling has_role only when it exists.
--    Return values are unchanged: 'edit' | 'comment' | 'view' | 'none'.
--
-- Safe to run more than once. Rollback section at the bottom.
-- =====================================================================

begin;

-- ---------------------------------------------------------------------
-- 1. Edit-level collaborator check (SECURITY DEFINER: no RLS recursion)
-- ---------------------------------------------------------------------
create or replace function public.is_resource_editor(_resource_type text, _resource_id uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1
    from public.resource_collaborators rc
    join public.app_users au on au.id = rc.app_user_id
    where rc.resource_type = _resource_type
      and rc.resource_id   = _resource_id
      and au.auth_user_id  = auth.uid()
      and lower(rc.access_level) = 'edit'
  );
$$;

revoke all on function public.is_resource_editor(text, uuid) from public, anon;
grant execute on function public.is_resource_editor(text, uuid) to authenticated, service_role;

drop policy if exists "owners and collaborators delete projects" on public.projects;
drop policy if exists "owners and editors delete projects" on public.projects;

create policy "owners and editors delete projects"
  on public.projects for delete to authenticated
  using (
    owner_id = (select p.id from public.profiles p where p.user_id = auth.uid())
    or public.is_resource_editor('project', id)
  );

-- ---------------------------------------------------------------------
-- 2. get_my_report_access(): VOLATILE + optional has_role
-- ---------------------------------------------------------------------
create or replace function public.get_my_report_access(_report_id uuid)
returns text
language plpgsql
volatile                      -- was STABLE; the backfill below is an UPDATE
security definer
set search_path = public
as $function$
declare
  _auth uuid := auth.uid();
  _auth_email text;
  _profile uuid;
  _created_by uuid;
  _analysis_id uuid;
  _project_id uuid;
  _level text;
  _is_admin boolean := false;
begin
  if _auth is null then
    return 'none';
  end if;

  select created_by, analysis_id, project_id
    into _created_by, _analysis_id, _project_id
  from public.reports where id = _report_id;

  if not found then
    return 'none';
  end if;

  select email into _auth_email from auth.users where id = _auth limit 1;

  select id into _profile from public.profiles where user_id = _auth limit 1;
  if _profile is null then
    select id into _profile from public.profiles where id = _auth limit 1;
  end if;

  if _created_by = _auth or (_profile is not null and _created_by = _profile) then
    return 'edit';
  end if;

  -- Admin override only where a has_role() helper is installed. Called
  -- dynamically so this function still compiles and runs without it.
  if exists (
    select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public' and p.proname = 'has_role'
  ) then
    begin
      execute 'select public.has_role($1, ''admin''::public.app_role)'
        into _is_admin using _auth;
    exception when others then
      _is_admin := false;     -- missing app_role type or different signature
    end;
    if coalesce(_is_admin, false) then
      return 'edit';
    end if;
  end if;

  -- Backfill missing auth_user_id link opportunistically (unchanged)
  if _auth_email is not null then
    update public.app_users
      set auth_user_id = _auth, updated_at = now()
    where auth_user_id is null
      and lower(email) = lower(_auth_email);
  end if;

  select rc.access_level into _level
  from public.resource_collaborators rc
  join public.app_users au on au.id = rc.app_user_id
  where (
      au.auth_user_id = _auth
      or (_auth_email is not null and lower(au.email) = lower(_auth_email))
      or (_profile is not null and au.profile_id = _profile)
    )
    and (
      (rc.resource_type = 'report' and rc.resource_id = _report_id)
      or (rc.resource_type = 'analysis' and rc.resource_id = _analysis_id)
      or (_project_id is not null and rc.resource_type = 'project' and rc.resource_id = _project_id)
    )
  order by case lower(rc.access_level)
    when 'edit' then 1
    when 'comment' then 2
    when 'view' then 3
    else 4 end
  limit 1;

  return coalesce(lower(_level), 'none');
end;
$function$;

commit;

-- =====================================================================
-- ROLLBACK (run manually only if needed)
-- =====================================================================
-- begin;
-- drop policy if exists "owners and editors delete projects" on public.projects;
-- create policy "owners and collaborators delete projects" on public.projects
--   for delete to authenticated
--   using ((owner_id = (select profiles.id from profiles where profiles.user_id = auth.uid()))
--          or is_resource_collaborator('project', id));
-- drop function if exists public.is_resource_editor(text, uuid);
-- -- get_my_report_access: re-create from the definition exported on 2026-10-13
-- -- (query H). Note that the original fails for non-creators.
-- commit;
