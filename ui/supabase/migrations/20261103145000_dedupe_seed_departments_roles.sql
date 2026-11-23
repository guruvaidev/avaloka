-- Before BG-05/BG-06 (20261103150000_bg05_bg06_roles_departments.sql): remove the
-- duplicate default rows that the org seed created twice, milliseconds apart
-- (department "Management", roles "Admin"/"Analyst" in Management).
-- With duplicates present, BG-05 skips its unique indexes (warning only), so
-- names per organisation would stay non-unique.
--
-- Duplicates that differ (a department's code, a role's permissions) are kept:
-- which one is right is an admin's decision. This warns and continues, as BG-05
-- documents. It used to raise, which stopped the upgrade of any install holding
-- such rows. The hook records a file only when it commits, so every install that
-- hit the exception runs this version; one that recorded the old version had no
-- such rows. 20261117110000_org_name_keys_per_org.sql then creates the unique
-- indexes for every organisation without duplicates.
--
-- Only exact duplicates are removed: same organisation, same name (case-insensitive),
-- and for roles the same permissions. The oldest row of each group is kept.
-- No BEGIN/COMMIT: apply with psql --single-transaction (-1).

-- Roles first (they carry the department name, not an id, on databases without BG-05).
delete from public.config_roles r
using (
  select id, row_number() over (
           partition by organization_id, lower(role), lower(department), permissions
           order by created_at, id) as rn
  from public.config_roles
) d
where r.id = d.id and d.rn > 1;

delete from public.departments x
using (
  select id, row_number() over (
           partition by organization_id, lower(department), upper(code)
           order by created_at, id) as rn
  from public.departments
) d
where x.id = d.id and d.rn > 1;

-- What is left differs and needs a decision: warn, keep the rows, continue.
do $$
begin
  if exists (select 1 from public.departments group by organization_id, lower(department) having count(*) > 1) then
    raise warning 'departments still have duplicate names that differ (code) in % organisation(s); kept until an admin merges them',
      (select count(distinct organization_id) from (select organization_id from public.departments
         group by organization_id, lower(department) having count(*) > 1) d);
  end if;
  if exists (select 1 from public.config_roles group by organization_id, lower(role), lower(department) having count(*) > 1) then
    raise warning 'config_roles still have duplicate names that differ (permissions) in % organisation(s); kept until an admin merges them',
      (select count(distinct organization_id) from (select organization_id from public.config_roles
         group by organization_id, lower(role), lower(department) having count(*) > 1) r);
  end if;
end $$;
