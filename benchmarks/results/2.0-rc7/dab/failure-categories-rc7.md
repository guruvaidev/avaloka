| query | cell | category | evidence |
|---|---|---|---|
| agnews:1 | haiku-5.5 | free-text extraction | "sports articles were identified by keyword matching" -> article 78 (GT The Rundown) |
| agnews:2 | haiku-5.5 | free-text extraction | "11 of 111 ... Judgment calls" (GT 16/111) |
| agnews:3 | haiku-5.5 | wrong join key/logic | "Only 2,273 of the 14,860 Europe article IDs have a matching document" -> 35.3/yr vs 336.6 |
| bookreview:2 | haiku-5.5 | unknown | 11 books listed, misses "Something That Feels Like Truth"; language filter details LIKE "%written in English%" vs purchase-id suffix join, not separable from logs |
| crmarenapro:2 | haiku-5.5 | free-text extraction | policy article chosen from knowledge text -> ka0Wt000000Ens5IAC (GT ...Eq0MIAS) |
| crmarenapro:3 | haiku-5.5 | validator/harness issue | review_failed; all 3 verdicts "requires a numeric result" on a stage-label question |
| crmarenapro:9 | haiku-5.5 | validator/harness issue | review_failed ("requires a numeric result", then "adds explanation") |
| crmarenapro:12 | haiku-5.5 | wrong aggregation/filter | opportunities filtered on CreatedDate in April -> 005Wt000003NJgAIAW (as 8bd7e5d0) |
| DEPS_DEV_V1:1 | haiku-5.5 | give-up | review x2 "abandons the task despite retained evidence"; 3 tool errors |
| DEPS_DEV_V1:2 | haiku-5.5 | give-up | budget_exceeded: "token limit reached (2018077/2000000)" |
| GITHUB_REPOS:1 | haiku-5.5 | output shape/format | right value as "1 of 3 (about 33%)"; validator needs a decimal -> "No value in LLM output rounds to 0.33" |
| googlelocal:2 | haiku-5.5 | free-text extraction | "identified by matching massage in the business name or description"; omits J B Oriental Inc |
| googlelocal:3 | haiku-5.5 | output shape/format | all 5 GT names present; hours abbreviated -> "Missing hours [Saturday, 5-11PM] for TACOS LA CABANA" |
| music_brainz_20k:1 | haiku-5.5 | unknown | "two of the four duplicate track records ... other two have no Apple Music sales in Canada" -> 601.44 vs 1059.46 |
| music_brainz_20k:3 | haiku-5.5 | free-text extraction | grouped by exact normalized title+artist -> "Groovey"; GT needs fuzzy title variants of "Zo gaat het leven aan je voor" |
| PANCANCER_ATLAS:1 | haiku-5.5 | wrong table/column | reported histological_type names (Oligoastrocytoma...) where GT groups by ICD-O-3 codes (9382/3...) |
| PATENTS:1 | haiku-5.5 | wrong aggregation/filter | returned 93 codes; misses A22B, includes A41H/A44D not in GT |
| PATENTS:2 | haiku-5.5 | wrong aggregation/filter | "34 German (DE) patents granted July-December 2019" (as 8bd7e5d0) |
| PATENTS:3 | haiku-5.5 | give-up | review x2 "abandons the task"; 3 tool errors |
| stockmarket:2 | haiku-5.5 | give-up | three-strikes; truncated args {"_invalid_arguments": ...} |
| stockmarket:4 | haiku-5.5 | give-up | "Not a complete answer ... checked only about 90 of the 234"; _invalid_arguments errors |
| yelp:3 | haiku-5.5 | other | "the business collection contains only 5 records" (it has 100) -> "1 business" |
| yelp:4 | haiku-5.5 | free-text extraction | "category is not a stored field. I derived it from the first-listed category" -> 3.53 vs 3.63 |
| yelp:5 | haiku-5.5 | wrong aggregation/filter | "offers WiFi was taken as WiFi = u'free'" -> MO/FL/LA tie (GT PA) |
| yelp:7 | haiku-5.5 | free-text extraction | "Categories come from the business description text"; Breakfast & Brunch missed |
| agnews:1 | gpt-oss-120b | free-text extraction | keyword sports filter on titles -> "Madden, ESPN Football..." |
| agnews:2 | gpt-oss-120b | free-text extraction | keyword science/tech list -> "58/111" |
| agnews:3 | gpt-oss-120b | free-text extraction | business = title/description contains "business" -> 41.8/yr |
| agnews:4 | gpt-oss-120b | provider error | tool_use_failed (ToolCallRejected): phantom tool 'commentary' {"answer": "South America"} (GT Africa) |
| bookreview:1 | gpt-oss-120b | give-up | three-strikes |
| crmarenapro:4 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'json' {"answer": "November"} |
| crmarenapro:6 | gpt-oss-120b | free-text extraction | chose knowledge article after reading 20 KB rows -> ka0Wt000000Eq0MIAS (GT ...EnwvIAC) |
| crmarenapro:7 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'commentary' {"answer": "None"} |
| crmarenapro:8 | gpt-oss-120b | wrong aggregation/filter | "sorted_agents[0]" among zero-transfer agents -> 005Wt000003NBcAIAW (as 8bd7e5d0) |
| crmarenapro:10 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'commentary' {"answer": "No agent processed more than one case..."} |
| crmarenapro:11 | gpt-oss-120b | wrong table/column | product taken from a June Case's order item -> 01tWt000006hTUkIAM (GT ...V8LIAU) |
| crmarenapro:12 | gpt-oss-120b | wrong aggregation/filter | CreatedDate-in-April filter + un-normalised ContractID join -> #005Wt000003NEa3IAG |
| DEPS_DEV_V1:1 | gpt-oss-120b | give-up | timeout: "tool 'query_db' exceeded 120.000s" |
| DEPS_DEV_V1:2 | gpt-oss-120b | provider error | context_length_exceeded |
| GITHUB_REPOS:1 | gpt-oss-120b | give-up | tool_error: "maximum failed tool calls reached" |
| GITHUB_REPOS:2 | gpt-oss-120b | give-up | three-strikes |
| GITHUB_REPOS:3 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'json' {"answer": "1077"} (= GT) |
| GITHUB_REPOS:4 | gpt-oss-120b | give-up | three-strikes |
| googlelocal:1 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'json' carrying a ranked list |
| googlelocal:2 | gpt-oss-120b | free-text extraction | name/description contains "massage"; omits J B Oriental Inc |
| googlelocal:4 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'commentary' carrying a table |
| music_brainz_20k:1 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'json' {"answer": "...$601.44..."} (GT 1059.46) |
| music_brainz_20k:2 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'commentary' {"answer": "Amazon Music"} (= GT) |
| music_brainz_20k:3 | gpt-oss-120b | wrong aggregation/filter | "GROUP BY track_id" -> "Systemisch bled" (as 8bd7e5d0) |
| PANCANCER_ATLAS:1 | gpt-oss-120b | give-up | declared non-answer: "Unable to compute ... clinical table does not contain a field that can be reliably joined" |
| PANCANCER_ATLAS:2 | gpt-oss-120b | give-up | three-strikes |
| PANCANCER_ATLAS:3 | gpt-oss-120b | give-up | declared non-answer "cannot be computed"; review also flagged "requires a numeric result" |
| PATENTS:1 | gpt-oss-120b | give-up | three-strikes |
| PATENTS:2 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'repo_browser.search' |
| PATENTS:3 | gpt-oss-120b | provider error | context_length_exceeded |
| stockindex:1 | gpt-oss-120b | wrong aggregation/filter | "Date >= '2020-01-01' AND Date < '2021-01-01'" (2020 only, not since 2020) -> NSEI |
| stockindex:3 | gpt-oss-120b | wrong aggregation/filter | ranking IXIC, NSEI, GDAXI...; also "United\u202fStates" (narrow NBSP) fails the country window |
| stockmarket:2 | gpt-oss-120b | give-up | budget: "max iterations reached" |
| stockmarket:3 | gpt-oss-120b | output shape/format | volumes right (23781.42...) but "Company" field holds the full description text -> "No number found near name" |
| stockmarket:4 | gpt-oss-120b | give-up | budget: "max iterations reached" |
| stockmarket:5 | gpt-oss-120b | give-up | declared non-answer after 4 tool errors: "I'm unable to compute ..." |
| yelp:1 | gpt-oss-120b | provider error | tool_use_failed: phantom tool 'json' {"answer": "The dataset does not contain any review records..."} |
| yelp:2 | gpt-oss-120b | give-up | three-strikes |
| yelp:3 | gpt-oss-120b | unknown | "25" (GT 35) after suffix-matching businessref/businessid; cause not visible |
| yelp:4 | gpt-oss-120b | give-up | three-strikes |
| yelp:5 | gpt-oss-120b | give-up | tool_error: "maximum failed tool calls reached" |
| yelp:6 | gpt-oss-120b | free-text extraction | right business, but "belongs to the Cafes category" (one of four listed in its description) |
| yelp:7 | gpt-oss-120b | give-up | three-strikes |

| category | haiku-5.5 | gpt-oss-120b |
|---|---:|---:|
| wrong join key/logic | 1 | 0 |
| free-text extraction | 7 | 6 |
| output shape/format | 2 | 1 |
| wrong aggregation/filter | 4 | 5 |
| wrong table/column | 1 | 1 |
| give-up | 5 | 16 |
| provider error | 0 | 13 |
| validator/harness issue | 2 | 0 |
| other | 1 | 0 |
| unknown | 2 | 1 |
| total | 25 | 43 |
