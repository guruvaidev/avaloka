"""Where model-written code runs, and what it is allowed to see.

Generated analysis code, the sample check the validator runs on it, and the
DTA transform's sample preview are all code a model wrote. Run in the API
process (or a child that inherits its environment) they see everything the
API holds: the credential-sealing key, provider keys, the auth secret,
``MCP_SERVICE_TOKEN``, ``AVALOKA_PII_KEY``, Redis, the connection registry
on /state, and a network that reaches every database the user connected.
None of that is needed to compute a group-by on a CSV.

``app.executor.client.run_script`` is the one way such code is run:

* With ``AVALOKA_EXECUTOR_URL`` set (the quickstart compose and the Helm chart
  set it), the script and exactly the input files it was given are sent to the
  executor service (``app.executor.server``), which runs on a network with no
  route to the databases or the internet, as a non-root user, on a read-only
  root filesystem, with no /state and no secrets in its environment. The
  declared output file comes back in the reply. A configured executor that
  cannot be reached is an error; there is no fallback to running locally.
* Without it (a developer's checkout, the CLI, tests) the script runs as a
  local child process, but with the environment reduced to the allowlist in
  ``app.executor.env``, so it still inherits no secret.

``app.executor.sample.run_main_on_sample`` is the validator's sample check. It
goes through the executor when one is configured; without one it runs its own
child with that environment, ``python -I``, a private temp dir and time and
memory limits -- never ``exec()`` in the API process.
"""
