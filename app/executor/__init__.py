"""Where model-written code runs, and what it is allowed to see.

Generated analysis code, the validator's sample check of it, and the DTA
transform's sample preview are all code a model wrote. Run in the API process,
or in a child that inherits its environment, that code sees everything the
server holds: provider keys, ``DB_ENCRYPTION_KEY``, ``MCP_SERVICE_TOKEN``,
``AVALOKA_PII_KEY``, Supabase keys, database passwords. None of that is needed
to compute a group-by on a CSV.

* ``app.executor.env.scrubbed_env`` is the environment such a child gets: an
  allowlist, ported from 2.0's ``app/executor/env.py``.
* ``app.executor.sample.run_main_on_sample`` is the validator's sample check,
  run in a child with that environment and with time and memory limits,
  instead of ``exec()`` in the server.

Egress (what that child can reach on the network) is not controlled here.
"""
