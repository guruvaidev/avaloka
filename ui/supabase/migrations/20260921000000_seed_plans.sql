-- Seed the canonical billing plans so fresh deploys come up with plan cards
-- populated (no hand-seeding after every cluster/workstation restart).
--
-- Idempotent: keyed on plans.id, so re-running never duplicates rows and never
-- breaks the foreign key from subscriptions.plan_id. Safe to apply any number of
-- times.
--
-- Ordering: the filename timestamp (20260921000000) sorts AFTER the migration
-- that creates public.plans, so the table exists by the time this runs.

insert into public.plans
    (id, plan_type, name, description, billing_interval, max_users, price,
     stripe_price_id, is_active, created_at, updated_at, paypal_plan_id)
values
    ('b6697d97-cee2-4e5c-b89b-3179acad41fd', 'free', 'Free',
     'Perfect for individuals exploring AI-powered data analysis.',
     null, null, 0.00, null, true, now(), now(), null),

    ('2abfb8c8-e811-4f7c-aa1a-8f70b773d7db', 'professional', 'Professional',
     'For growing teams who need advanced ecommerce and reporting capabilities.',
     'monthly', null, 4.99, 'price_1TzZPq0EINf8Vm2nU9g5075G', true, now(), now(),
     'P-3S606036XK8830522NJZRUQI'),

    ('9b66fe6b-ef14-4de3-8a4f-b2f27befaeb5', 'professional', 'Professional',
     'Chosen by 300K+ users.',
     'yearly', null, 59.88, 'price_1TzZR70EINf8Vm2nXhMw102j', true, now(), now(),
     'P-6T357712PM770242ANJZRVIQ'),

    ('1905809c-e56b-49dd-9655-1f303cd1c452', 'enterprise', 'Enterprise',
     'Built for organizations with advanced AI and security needs.',
     'monthly', null, 99.99, 'price_1TzZSj0EINf8Vm2nvf0s56nt', true, now(), now(),
     'P-25983237X7142182LNJZRVWI'),

    ('e79a2977-4939-47f5-b0c7-874b45fe3917', 'enterprise', 'Enterprise',
     'Chosen by 300+ users.',
     'yearly', null, 1199.88, 'price_1TzZTK0EINf8Vm2nutXAbmlK', true, now(), now(),
     'P-69E73597WR334025GNJZRWAA')
on conflict (id) do update set
    plan_type        = excluded.plan_type,
    name             = excluded.name,
    description      = excluded.description,
    billing_interval = excluded.billing_interval,
    max_users        = excluded.max_users,
    price            = excluded.price,
    stripe_price_id  = excluded.stripe_price_id,
    is_active        = excluded.is_active,
    paypal_plan_id   = excluded.paypal_plan_id,
    updated_at       = now();
