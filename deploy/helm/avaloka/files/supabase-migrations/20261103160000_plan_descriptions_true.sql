-- Plan descriptions that claim only what each plan includes.
--
-- 20260921000000_seed_plans.sql seeded the plan cards' taglines. One promised
-- "ecommerce" capabilities that no plan has, and two cited user counts instead
-- of the plan's contents. These say what the plan includes, per the edition
-- matrix (app/core/editions.py, docs/EDITIONS.md).
--
-- Only seeded rows are touched, and only while they still hold the exact text
-- the seed wrote: each row is matched by its seeded plans.id AND its seeded
-- description. A description an operator has changed matches neither pair and
-- is left alone, as is any plan the seed did not create.
--
-- Idempotent: once a row is rewritten its description no longer equals the old
-- text, so running this again updates nothing.

update public.plans as p
set description = v.new_description,
    updated_at  = now()
from (values
    ('b6697d97-cee2-4e5c-b89b-3179acad41fd'::uuid,
     'Perfect for individuals exploring AI-powered data analysis.',
     'Conversational data analysis, Auto Insights and file connectors.'),
    ('2abfb8c8-e811-4f7c-aa1a-8f70b773d7db'::uuid,
     'For growing teams who need advanced ecommerce and reporting capabilities.',
     'Scheduling, Reports, database and cloud connectors, and teams that share and comment on analyses.'),
    ('9b66fe6b-ef14-4de3-8a4f-b2f27befaeb5'::uuid,
     'Chosen by 300K+ users.',
     'Scheduling, Reports, database and cloud connectors, and teams that share and comment on analyses.'),
    ('1905809c-e56b-49dd-9655-1f303cd1c452'::uuid,
     'Built for organizations with advanced AI and security needs.',
     'Everything in Professional, plus departments, roles, branches and app users.'),
    ('e79a2977-4939-47f5-b0c7-874b45fe3917'::uuid,
     'Chosen by 300+ users.',
     'Everything in Professional, plus departments, roles, branches and app users.')
) as v(id, old_description, new_description)
where p.id = v.id
  and p.description = v.old_description;
