-- Sample data for the quickstart's database connector: one synthetic table,
-- generated here so no file and no network is needed. Deterministic (no
-- random()), so every quickstart sees the same 2,000 orders.
CREATE TABLE IF NOT EXISTS orders (
    order_id     integer PRIMARY KEY,
    order_date   date        NOT NULL,
    region       text        NOT NULL,
    channel      text        NOT NULL,
    product      text        NOT NULL,
    units        integer     NOT NULL,
    unit_price   numeric(8,2) NOT NULL,
    discount_pct numeric(4,1) NOT NULL,
    returned     boolean     NOT NULL
);

INSERT INTO orders
SELECT g,
       date '2025-01-01' + (g * 7919 % 365),
       (ARRAY['North','South','East','West'])[1 + g % 4],
       (ARRAY['web','store','partner'])[1 + (g * 31) % 3],
       (ARRAY['basic','standard','premium','enterprise'])[1 + (g * 17) % 4],
       1 + (g * 13) % 9,
       (ARRAY[19.00, 49.00, 99.00, 249.00])[1 + (g * 17) % 4],
       ((g * 7) % 5) * 5.0,
       -- Returns rise with discount and are commonest in the partner channel.
       ((g * 7) % 5) * 5 + CASE WHEN (g * 31) % 3 = 2 THEN 12 ELSE 0 END > (g * 37) % 40
FROM generate_series(1, 2000) AS g
ON CONFLICT DO NOTHING;
