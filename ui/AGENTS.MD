Keep the analysis voice welcome portrait and Insights Assistant portrait sourced from the same avatarPortrait constant, so changing the character updates both views.
Keep report detection, safe structured-value parsing, and Summary presentation in AnalysisReportSummary; leave analysis result fetching and raw table/CSV behavior in the analysis route so report formatting stays display-only.
Normalize backend histogram bins and percent metadata in dynamicChart; preserve range labels in chart-points so charts and avatar explanations use the same data.
Keep dataset-profile presentation inside AutoInsightsModal and pass suggestion text outward through a callback, so it never owns chat or request behavior.
Keep Auto Insights header, navigator and column-statistics presentation in auto-insights/AutoInsightsPanels; consume existing per-dataset configs only so display changes cannot alter dataset loading or chat behavior.
Apply Stack bar emphasis through DynamicChart's opt-in stackPresentation prop; default rendering stays unchanged so Overview and other charts are not restyled.
Keep Stack chart scope in ChartScopeStrip using the active group's existing dataset metadata, independently of column-statistics scope, to prevent inferred provenance or new data requests.
Keep year/identifier classification, invalid-scatter conversion and log eligibility in chart-semantics; normalize once in dynamicChart so Overview, Stack and avatar share authoritative backend points.
Keep tick/domain formatting and scatter-only scale policy in chart-display, consumed by DynamicChart; shared pure display helpers make chart rules regression-testable without changing backend points.- Data View table/chart-builder/column-summary helpers live in src/lib/data-view.ts and UI in src/components/dashboard/data-view/; they operate only on rows already loaded so sorting/filtering/charting never trigger backend requests.

Keep upload review presentation in dashboard/upload and pure naming/readiness/profile helpers in upload-review; consume existing upload metadata and preserve request shapes.
Keep workspace findings union in workspace-findings and height/sidebar containment scoped to the analysis workspace; derive first-message presentation from loaded slides/messages without writes or requests so reloads are deterministic.
Keep 8010 proxy health checks and application forwarding on the shared BACKEND_API_BASE constant, ignoring caller-provided base URLs so both stages use the same trusted target; leave MCP service bases independent.
- Multi-dataset (several sheets/files) presentation lives in src/lib/top-insights.ts, src/lib/multi-dataset-view.ts and components/dashboard/multi-dataset; upload kind/selection rules in upload-review. Single-dataset paths must stay untouched so one-file analyses never change.

Keep analysis chip naming in DatasetChipName using upload-review helpers; the route retains all selection and request behavior.
Keep Ray jobs API transport and queries in src/lib/ray-jobs.ts and its UI in src/components/ray-jobs; mirror the limits endpoint's external-session proxy transport so jobs use the same signed-in identity without changing the shared client.
Keep Ray job result parsing, display-only ID normalization and local start/cluster-clock formatting in the Ray jobs library; cache only already-loaded log results by original submission ID in the query client so display improvements never add log requests or change backend identifiers.
Keep Ray jobs live refresh in Query options with retained last-good list data; scope visibility/focus listeners to the Ray jobs UI so other pages' refresh policies stay unchanged.
Keep bucket-picker classification and labels display-only in bucket-picker, with folder-name interaction in BucketFolderName, so existing listing and registration transport remain unchanged.
Use ui/src as a symlink to the application src directory for bucket-picker tests' ../../src imports, so tests exercise real source without duplicating it.
