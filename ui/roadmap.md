# Cloud bucket picker
- [x] Fix six bucket picker presentation details; 24 focused tests pass, fixture navigation/48px rows/truncation checked in dark/light; preserve transport and exclusive selection. Full suite retains unrelated chart assertions and Ray auth-mock failures; real bucket metadata/session unavailable.

# Ray jobs
- [x] Implement visible-only 5s/30s refresh, failure retention/back-off, live timestamps, new-row emphasis, terminal logs and short-zone fallback; 163 tests pass and browser fixtures verify automatic new jobs/completion, final 20 rows, IST and failure recovery. Automated build OK; real chat-to-transfer remains blocked by external sign-in.
- [x] Fix reported row counts, display-only repeated ID prefixes and local-start/cluster-clock labels; 156 tests pass and browser fixtures verify clipboard, IST start, unchanged logs and cached 20 rows after closing. Live named-job verification remains blocked by external sign-in.
- [x] Add authenticated jobs route, feature-gated rail navigation, filters and live polling.
- [x] Add logs, stop confirmation, upgrade/error handling and chat ID links.
- [x] Verify 133 tests and desktop/mobile light/dark fixture rendering, logs download and stop confirmation; automated build OK.
- [ ] Verify live Ray jobs for the signed-in external account: the available preview session redirects to sign-in and cannot access that account's jobs.

# Port 8010 proxy forwarding investigation
- [x] Use computing-flooring-mailman-speeds tunnel in shared proxy config and test expectations; preserve MCP settings. Six proxy tests pass; direct GET /version verified health 200 → application fetch → upstream/frontend 200 with version 1.6.0. Existing diagnostic stages retained; automated build OK.
- [ ] Update managed VITE_API_BASE: .env editing is prohibited; shared proxy config does not consume this stale value.
- [ ] Verify a real signed-in UI request: own-account preview session restored, but /datasets redirected to login before any proxy request. App's separate external auth session remains unavailable. Requires the user to sign in to the app and trigger the operation; Uvicorn logs are not accessible here.

# Welcome screen UI fixes
- [x] BG14: Title Case sidebar menu labels.
- [x] BG15–BG16: Consistent logo bounds and secondary View More action.
- [x] BG17: AA contrast for shared subdued text (7.07:1 or better on checked light surfaces).
- [x] BG18: Browse files CTA and visible upload interaction states.
- [x] BG19: Route-driven active rail indicator and aria-current verified on Welcome.
- [x] Verify Welcome screen controls and visual states without changing data flow. File picker, expanded sources, and drag-over checked; subsequent protected-page navigation redirected to login, preventing its active-state check.

# Auto Insights fixes
- [x] Overview follow-up: category histograms, scatter-only header scale control, readable axes/rates/insights, ring/tiles and separate issue/suggestion cards. Eighteen regression tests and mortgage-style browser fixtures passed at 1280/1027/390px, including equal-width bins, rate tooltip, 237px scatter area, selected controls, expansion and callback. Actual mortgage_loans_test.csv was unavailable; complete report delivered.
- [x] G33–G36: authoritative points, year/code axes, valid scatters, hidden-point note and automatic/manual log scales. Twelve regression tests and browser mortgage-style fixtures passed in Overview and Stack; real uploaded mortgage data was unavailable. Complete change report delivered.
- [x] Add Stack-only numbers/sample scope pills and read-only sample list using existing metadata; verify and deliver exact change report. Fixture browser checks passed for full-file/sample/unknown scope, read-only popover, footer independence and 1366/786/390px. Signed-in /proanalysis has no saved chart to test; chip mappings drop optional sample names/lists, so no-name fallback is used unless config.dataset supplies them.
- [x] Follow-up items 1–4: immediate filled pies and chart sizing, readiness fill/reasoning, viewport-safe Stack and arrow clearance, full meanings with metadata fallback.
- [x] Verify follow-up chart types and widths; deliver complete before/after report. Six fixture chart types passed at 1366/1200/1024/786/390px; signed-in /proanalysis has no recent analyses or saved charts, so real saved-dataset verification remains unavailable.
- [x] Option A: persistent dataset header, unstacked navigator/chart/statistics layout and Column statistics tab.
- [x] Verify dataset switching, keyboard navigation, statistics controls and responsive rendering; provide exact change report. Isolated configuration fixtures passed at 1366/1024/390px, signed-in demo modal passed; saved live chart configs unavailable for this account.
- [x] Render backend histogram bins; retain fallback only when points are missing.
- [x] Honor full-file scope in the first upload/cloud chat summary.
- [x] Show every bar category and percent-rate axes/tooltips.
- [x] Verify targeted behavior and provide exact before/after changes. Isolated browser checks passed; actual ICU upload was not available for verification.

# Upload message and Auto Insights presentation
- [x] Prefer the backend markdown upload message with the existing fallback.
- [x] Add dataset overview, chart headline/facts, dataset findings, and column statistics.
- [x] Use whole-file row counts and readable histogram ticks without changing requests or plotted bins.
- [x] Verify the presentation helpers, preview build, and Auto Insights modal in the running app; a saved profile-overview dataset was not available for full live-data verification.
# Data View sample/results split
- [x] BG45–BG47: bottom panels removed, first-message findings union from loaded data, contained right chat and remembered collapse/drawer. All 66 tests and desktop/drawer checks passed; actual Global YouTube Statistics saved analysis unavailable for authenticated reload verification.
- [x] Data sample and Analysis results as separate sub-views with navigation, persistence, CSV/chart/summary source, filter icon fix.

# Upload modal BG20–BG26
- [x] Split upload/review, compact replace/add files, editable names, real issue details and pinned actions.
- [x] Confirm no cleaning options are sent; label unknown backend rules, repair SQL Server name, run regression and fixture browser checks, deliver report. Real upload/start persistence verification blocked by the separate external sign-in dependency.

# Analysis results presentation (issues 1–7)
- [x] Theme-aware result header, measured expansion, separated links and accessible multi-run controls.
- [x] Group header filters, reposition plan badges and display readable dataset chip names.
- [x] Regression checks: 16 files / 119 tests pass; automated typecheck/build OK. Dark/light browser fixtures verify header, links, one/two-run navigation, dropdown and exact 4px filter gaps without page overflow; complete report delivered.
- [ ] Verify the saved Global YouTube Statistics analysis and run a second prompt through real chat: this preview has no usable saved analysis/session for that external backend, so fixture checks do not establish live sending or persistence.
