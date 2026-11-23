"""The agent trace: what each agent did in one analysis, and how they handed off.

Recording (:mod:`app.trace.recorder`) wraps every graph node and router when
the graph is built, and attaches a usage callback to every chat model at the
one place models are constructed (``app.core.inference``). Nothing else in the
agents has to know it is being traced.

Two audiences, two rules:

* **The trace** (:mod:`app.trace.build`) is the user's own analysis, stored in
  their cluster and served only to them. It may show their columns, formulas
  and values, because it is describing their work back to them.
* **Telemetry** (:mod:`app.trace.telemetry`) is the projection of the same
  steps into the ingest schema in ``telemetry_service/schema.py``: agent names,
  timings, token counts, outcome and exception *class*. It carries no data, no
  column name and no file name, by construction and by test.
"""
