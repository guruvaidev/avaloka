create or replace function public.is_org_owner_of_profile(_profile_id uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1
    from public.profiles p
    join public.organizations o on o.id = p.organization_id
    where p.id = _profile_id
      and p.organization_id is not null
      and o.owner_profile_id = public.current_profile_id()
  )
$$;

revoke all on function public.is_org_owner_of_profile(uuid) from public, anon;
grant execute on function public.is_org_owner_of_profile(uuid) to authenticated, service_role;

create policy "org owners read member projects"
on public.projects for select to authenticated
using (public.is_org_owner_of_profile(owner_id));

create policy "org owners read member analyses"
on public.analyses for select to authenticated
using (public.is_org_owner_of_profile(owner_id));