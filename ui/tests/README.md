# UI tests

| Command             | Runs                                                  |
| ------------------- | ----------------------------------------------------- |
| `npm run test:unit` | Vitest: `tests/unit/` and `tests/component/` (jsdom)  |
| `npm run test:e2e`  | Playwright: `tests/e2e/` against a running deployment |
| `npm test`          | Both, unit first                                      |

Needs Node `^22.22.2` (jsdom's floor) — the same major as `ui/Dockerfile`.

## Layout

- `tests/unit/` — pure logic. Import with the `@/` alias, no rendering.
- `tests/component/` — React components rendered with Testing Library. Recharts
  is real; only `ResponsiveContainer` is replaced, because jsdom has no layout
  and it would otherwise measure 0×0 and draw nothing.
- `tests/e2e/` — Playwright. One spec, on purpose.

## End-to-end environment

The spec skips, naming what is missing, unless all of these hold:

| Variable            | Meaning                                                              |
| ------------------- | -------------------------------------------------------------------- |
| `E2E_BASE_URL`      | Where the UI is served. Must answer HTTP.                            |
| `E2E_API_URL`       | The backend the UI talks to. Must answer HTTP.                       |
| `E2E_STORAGE_STATE` | Path to a Playwright storage-state file holding a signed-in session. |
| `E2E_ANALYSIS_PATH` | Optional. Page with the prompt box; default `/analysis`.             |
| `E2E_PROMPT`        | Optional. The question to ask.                                       |

Login is by emailed link, so a session cannot be created inside the test. Record
one once, by hand, and keep the file out of the repository:

```sh
npx playwright install chromium
npx playwright codegen --save-storage="$HOME/.config/avaloka/e2e-state.json" "$E2E_BASE_URL"
```
