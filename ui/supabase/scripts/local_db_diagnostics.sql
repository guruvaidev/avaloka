-- Read-only diagnostics for the in-cluster Supabase database.
-- Run:
--   kubectl exec -i deploy/avaloka-supabase-db -- psql -U postgres -d postgres -X \
--     < local_db_diagnostics.sql > local_db_report.txt 2>&1
-- Nothing here changes data. Errors (e.g. a missing table) are expected and fine.

\pset pager off
\pset format aligned
\x auto

\echo '==================== 0. Row counts ===================='
select (select count(*) from auth.users)            as auth_users,
       (select count(*) from public.profiles)        as profiles,
       (select count(*) from public.organizations)   as organizations,
       (select count(*) from public.app_users)       as app_users,
       (select count(*) from public.projects)        as projects,
       (select count(*) from public.analyses)        as analyses,
       (select count(*) from public.resource_collaborators) as collaborators;

\echo '==================== 1. Migration tracking ===================='
select to_regclass('supabase_migrations.schema_migrations') as migrations_table;
select version from supabase_migrations.schema_migrations order by 1;

\echo '==================== 2. Triggers on auth.users and public.organizations ===================='
select tgrelid::regclass as tbl, tgname, tgfoid::regprocedure as fn, tgenabled
from pg_trigger
where tgrelid in ('auth.users'::regclass, 'public.organizations'::regclass,
                  'public.profiles'::regclass, 'public.app_users'::regclass,
                  'public.projects'::regclass, 'public.analyses'::regclass)
  and not tgisinternal
order by 1, 2;

\echo '==================== 3. Key function definitions ===================='
select p.proname, p.prosecdef, pg_get_userbyid(p.proowner) as owner,
       pg_get_functiondef(p.oid) as body
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where n.nspname = 'public'
  and p.proname in ('handle_new_user','ensure_org_owner_app_user','is_resource_collaborator',
                    'is_resource_editor','can_manage_resource_share','can_share_report',
                    'current_org_id','prevent_profile_org_change')
order by 1;

\echo '==================== 4. RLS enabled? ===================='
select relname, relrowsecurity as rls_on, relforcerowsecurity as rls_forced
from pg_class
where relnamespace = 'public'::regnamespace
  and relname in ('analyses','projects','reports','app_users','profiles','organizations',
                  'resource_collaborators','departments','config_roles','branches','organization_teams')
order by 1;

\echo '==================== 5. Policies ===================='
select tablename, policyname, cmd, roles, qual, with_check
from pg_policies
where schemaname = 'public'
  and tablename in ('analyses','projects','reports','app_users','profiles','organizations',
                    'resource_collaborators','departments','config_roles','branches','organization_teams')
order by tablename, policyname;

\echo '==================== 6. Grants on projects / analyses ===================='
select table_name, grantee, string_agg(privilege_type, ',' order by privilege_type) as privs
from information_schema.role_table_grants
where table_schema = 'public' and table_name in ('projects','analyses','resource_collaborators')
  and grantee in ('anon','authenticated','service_role')
group by 1, 2 order by 1, 2;

\echo '==================== 7. Columns of core tables ===================='
select table_name, column_name, data_type, column_default, is_nullable
from information_schema.columns
where table_schema = 'public'
  and table_name in ('profiles','organizations','app_users','projects','analyses',
                     'config_roles','departments','branches','organization_teams')
order by table_name, ordinal_position;

\echo '==================== 8. owner_id id-space (analyses / projects) ===================='
select 'analyses' as t,
  count(*) filter (where owner_id in (select id from profiles))      as profile_ids,
  count(*) filter (where owner_id in (select user_id from profiles)) as auth_uids,
  count(*) filter (where owner_id in (select id from app_users))     as app_user_ids,
  count(*) as total
from analyses
union all
select 'projects',
  count(*) filter (where owner_id in (select id from profiles)),
  count(*) filter (where owner_id in (select user_id from profiles)),
  count(*) filter (where owner_id in (select id from app_users)),
  count(*)
from projects;

\echo '==================== 9. BG-01: profiles pointing at an org that does not exist ===================='
select p.id as profile_id, p.user_id, p.organization_id,
       exists (select 1 from app_users au where au.auth_user_id = p.user_id) as linked_elsewhere
from profiles p
left join organizations o on o.id = p.organization_id
where p.organization_id is not null and o.id is null;

\echo '==================== 10. BG-02: invited users never linked to their login ===================='
select au.organization_id, au.email, au.status, au.invited_at,
       exists (select 1 from app_users x where x.auth_user_id = u.id) as login_used_by_other_row
from app_users au
join auth.users u on lower(u.email) = lower(au.email)
where au.auth_user_id is null;

\echo '==================== 11. BG-03: sharing rows by type ===================='
select resource_type, count(*) from resource_collaborators group by 1 order by 1;

\echo '==================== 12. BG-05: role / department values in use ===================='
select role, department, branch, count(*) from app_users group by 1, 2, 3 order by 4 desc;

\echo '==================== 13. Auth settings that matter ===================='
select count(*) filter (where email_confirmed_at is null) as unconfirmed_users,
       count(*) as total_users
from auth.users;

\echo '==================== done ===================='
