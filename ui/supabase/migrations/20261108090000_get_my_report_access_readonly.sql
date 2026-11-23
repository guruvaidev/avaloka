-- get_my_report_access runs inside RLS (reports/report_comments, and projects via reports).
-- It must not write: GET requests run in read-only transactions (Postgres 25006).
create or replace function public.get_my_report_access(_report_id uuid)
returns text
language plpgsql
stable
security definer
set search_path to 'public'
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

  if exists (
    select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public' and p.proname = 'has_role'
  ) then
    begin
      execute 'select public.has_role($1, ''admin''::public.app_role)'
        into _is_admin using _auth;
    exception when others then
      _is_admin := false;
    end;
    if coalesce(_is_admin, false) then
      return 'edit';
    end if;
  end if;

  -- No write here: the email match below covers rows whose auth_user_id is not linked yet.
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
    when 'edit' then 1 when 'comment' then 2 when 'view' then 3 else 4 end
  limit 1;

  return coalesce(lower(_level), 'none');
end;
$function$;

-- One-time backfill of the links the function used to write on the fly.
with c as (
  select distinct on (u.id) au.id as app_user_id, u.id as auth_id
  from public.app_users au
  join auth.users u on lower(u.email) = lower(au.email)
  where au.auth_user_id is null
    and not exists (select 1 from public.app_users x where x.auth_user_id = u.id)
  order by u.id, au.id
)
update public.app_users a
set auth_user_id = c.auth_id, updated_at = now()
from c
where a.id = c.app_user_id;

notify pgrst, 'reload schema';
