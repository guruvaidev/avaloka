-- BG-01 follow-up: remove the organizations that 20260907000000_fix_new_user_org_provisioning
-- auto-created for self-serve users. 20261103120000_bg01_no_org_for_self_serve stopped creating
-- them for NEW users but left existing ones ("Existing users' auto-created orgs are NOT touched").
-- A leftover org makes the user look like a member of "another organization", so they cannot be
-- invited (checkInviteEmail). Free / Professional users must have profiles.organization_id = NULL.
--
-- An org is cleaned only when ALL of these hold:
--   * the owner is on the free or professional plan;
--   * the only member is the owner (no other app_users rows, no other profiles in the org);
--   * every analysis and project in it is owned by the owner (they are detached, not deleted:
--     organization_id -> NULL; the owner keeps access through owner_id);
--   * nothing else references it: reports, subscriptions, departments, billing_invoices,
--     branches, config_roles, organization_teams, payment_methods;
--   * no other public table holds rows with that organization_id (checked below; aborts if any).
-- Enterprise orgs and shared orgs are never touched.
--
-- No BEGIN/COMMIT: apply with psql --single-transaction (-1).
-- Dry run: (echo 'begin;'; cat <this file>; echo 'rollback;') | psql ...

-- profiles_prevent_org_change allows organization_id changes for service_role only.
select set_config('request.jwt.claim.role', 'service_role', true);

create temp table _bg01_orgs on commit drop as
select o.id, o.name, o.owner_profile_id, lower(u.email) as owner_email
from public.organizations o
join public.profiles p on p.id = o.owner_profile_id
join auth.users u on u.id = p.user_id
where avaloka_meta.plan_tier_of(u.id) in ('free', 'professional')
  and not exists (
    select 1 from public.app_users a
    where a.organization_id = o.id
      and a.profile_id is distinct from o.owner_profile_id
      and a.auth_user_id is distinct from u.id
  )
  and not exists (select 1 from public.profiles x where x.organization_id = o.id and x.id <> o.owner_profile_id)
  and not exists (select 1 from public.analyses x where x.organization_id = o.id and x.owner_id is distinct from o.owner_profile_id)
  and not exists (select 1 from public.projects x where x.organization_id = o.id and x.owner_id is distinct from o.owner_profile_id)
  and not exists (select 1 from public.reports x where x.organization_id = o.id)
  and not exists (select 1 from public.subscriptions x where x.organization_id = o.id)
  and not exists (select 1 from public.departments x where x.organization_id = o.id)
  and not exists (select 1 from public.billing_invoices x where x.organization_id = o.id)
  and not exists (select 1 from public.branches x where x.organization_id = o.id)
  and not exists (select 1 from public.config_roles x where x.organization_id = o.id)
  and not exists (select 1 from public.organization_teams x where x.organization_id = o.id)
  and not exists (select 1 from public.payment_methods x where x.organization_id = o.id);

-- What will be cleaned, with what is detached.
select c.name, c.owner_email,
       (select count(*) from public.analyses x where x.organization_id = c.id) as analyses_detached,
       (select count(*) from public.projects x where x.organization_id = c.id) as projects_detached
from _bg01_orgs c
order by c.owner_email;

-- 1. Detach the owner's own analyses and projects (kept, now personal).
update public.analyses x set organization_id = null from _bg01_orgs c where x.organization_id = c.id;
update public.projects x set organization_id = null from _bg01_orgs c where x.organization_id = c.id;

-- 2. Detach the owner (self-serve users have no org).
update public.profiles p
set organization_id = null
from _bg01_orgs c
where p.id = c.owner_profile_id
  and p.organization_id = c.id;

-- 3. Remove the owner's membership row in the org.
delete from public.app_users a
using _bg01_orgs c
where a.organization_id = c.id;

-- 4. Safety net: no other public table may still point at these orgs (tables without a
--    foreign key would be left dangling). Abort, naming the table, if one does.
do $$
declare t record; n bigint;
begin
  for t in
    select c.table_name
    from information_schema.columns c
    join information_schema.tables tb on tb.table_schema = c.table_schema and tb.table_name = c.table_name
    where c.table_schema = 'public' and c.column_name = 'organization_id'
      and tb.table_type = 'BASE TABLE'
      and c.table_name not in ('organizations')
  loop
    execute format('select count(*) from public.%I where organization_id in (select id from _bg01_orgs)', t.table_name) into n;
    if n > 0 then
      raise exception 'public.% still has % row(s) for an org being removed; nothing changed', t.table_name, n;
    end if;
  end loop;
end $$;

-- 5. Remove the now-empty orgs.
delete from public.organizations o
using _bg01_orgs c
where o.id = c.id;

notify pgrst, 'reload schema';
