# Avaloka AI 1.0 — User Guide

**Document version 1.0 · Applies to Avaloka 1.6**

---

## About this guide

This is the **hands-on guide**: what you click, what you type, and what you see
back. It walks through Avaloka the way you actually meet it — sign in, upload a
file, ask a question, connect a database, train a model.

If you want the complete capability and integration reference instead — every
format, every endpoint, every edition gate — read the
[Technical User Guide](TECHNICAL_USER_GUIDE.md). The two are companions: this
one shows you the product, that one specifies it.

> **A note on screenshots.** The images in this guide are placeholders. The
> previous guide's captures were taken against an earlier UI and no longer match
> what you will see, so they were removed rather than left to mislead — a stale
> screenshot is worse than none, because you trust it and then cannot find the
> control it shows. Each placeholder names the capture needed; see
> [images/user-guide/README.md](images/user-guide/README.md).

---

## 1. Welcome and core concepts

Avaloka is an AI team of data scientists you direct in plain language. Give it a
dataset and ask for what you want — *"find the fastest-growing channels"*,
*"move this into my warehouse"*, *"train a model to predict churn"* — and
Avaloka plans the analysis, writes and checks the code, runs it, and shows you
the results.

You never write code, and nothing important happens without your say-so: you
review plans and approve decisions along the way. One conversation can carry a
dataset the whole distance — from a raw file, to cleaned and reshaped data, to
charts and insights, to a trained model you can query for predictions.

### What you can do

| | |
| --- | --- |
| **Analyze** | Upload a dataset and ask questions in plain English. Each answer arrives as a new dataset you can keep refining. |
| **Visualize** | Charts appear automatically in the Auto Insights panel — chosen to fit your data, no configuration needed. |
| **Move data** | Register a database or cloud bucket as a connection, then ask Avaloka to transfer your dataset there, transforming it on the way. |
| **Model and predict** | Ask Avaloka to train a model, approve its plan, then test predictions in chat or from the Inference Panel. |
| **Stay in context** | Avaloka remembers your data's structure, your preferences, and your past results, so each request builds on the last. |

### How it works

Behind the chat is a team of specialised agents led by a **Planner** — the agent
you are actually talking to. When you send a prompt, the Planner decides what
needs to happen and delegates: a **Coder** writes the Python, a **Validator**
checks it against your data's real structure and sends it back for fixes if
anything is off, an **Execution** agent runs it, and a **Visualization** agent
turns results into charts. Specialist agents handle transfers and model
training, and heavy workloads run on a compute cluster rather than in your
session.

You do not manage any of this. It is worth knowing only because this guide
refers to these agents by name, and because it explains Avaloka's rhythm: a
moment of planning, then code, checks, execution, and results.

### Your workspace

> **[Screenshot placeholder — `dashboard-overview.png`]**
> *The Dashboard: your files on the left, data and results in the middle, the AI Assistant alongside.*

| Area | What it is |
| --- | --- |
| **Dashboard** | Home base — files left, data and results centre, AI Assistant alongside |
| **Upload File / File History** | Add datasets and revisit earlier ones |
| **Data Profile & Statistics** | An automatic summary of each dataset: columns, types, missing values |
| **AI Assistant** | The chat panel where everything happens. Each file gets its own thread, listed under **Chats** |
| **Auto Insights** | Automatically generated charts for whatever data you are viewing |
| **SQL Datastore / Cloud Dataset** | Register connections to databases and cloud buckets |
| **Inference Panel** | Query your trained models for predictions |

### Words this guide uses

- **Dataset** — the analyzable data behind a file you uploaded or a source you connected. Every analysis produces a new one.
- **Prompt** — a plain-English request you type into the chat.
- **Plan** — Avaloka's proposed course of action, shown for review before big steps like model training.
- **Connection** — a saved link to a database or cloud bucket, referred to by a short alias you choose.
- **Transfer** — moving a dataset into a connection, with optional transformations on the way.
- **Model / run ID** — a trained predictor, and the receipt identifying the exact training run that produced it.
- **Session** — your current working context: your chat threads and what Avaloka currently remembers.

---

## 2. Getting started

### Before you begin

You will need a modern browser — Chrome, Edge, Firefox, or Safari, kept
reasonably up to date — and a valid work email address.

Self-hosting instead? See [INSTALL.md](INSTALL.md).

### Creating your account

If your organisation already has a workspace, ask your administrator to invite
you rather than signing up separately — that way your account is attached to the
right team and its data.

1. Go to the Avaloka sign-in page.
2. Click **Sign Up / Get Started** in the top-right corner.
3. Enter your **Email**.
4. Enter a **Password**.
5. Re-enter it in **Confirm Password**.
6. Enter your full name and organisation.
7. Review the Terms of Service and Privacy Policy, then accept them.
8. Click **Create Account**.

Avaloka sends a verification email. **You cannot sign in until you verify your
address.** Open the message and click **Verify Email**. If it has not arrived
within a few minutes, check your spam folder, or click **Resend email** on the
verification screen.

### Logging in

1. Click **Log In** in the top-right corner.
2. Enter your registered email and password.
3. *(Optional)* Select **Remember me** to stay signed in on this device. Leave it clear on shared or public computers.
4. Click **Log In**.

First time in, you will see an onboarding flow and an empty workspace.

### Resetting a forgotten password

1. On the login page, click **Forgot password?**
2. Enter your registered email and click **Send reset link**.
3. Open the reset email and click **Reset Password**. The link is single-use and expires.
4. Enter and confirm your new password, then click **Update Password**.

For security, Avaloka shows the same confirmation whether or not the address is
registered. If no email arrives, the likeliest cause is that you signed up with
a different address.

---

## 3. Working with data

### Uploading a file

Click **Upload File**. Accepted formats are **CSV, Excel, JSON, and Parquet**,
up to **100 MB per file**.

> **[Screenshot placeholder — `upload-complete.png`]**
> *A finished upload with the dataset open.*

The left panel lists everything you have uploaded; clicking a file opens the
dataset.

> **[Screenshot placeholder — `file-history.png`]**
> *File History listing previously uploaded datasets.*

### Reading the Data Profile

The moment a dataset arrives, Avaloka gets to know it — counting and typing
every column, measuring what is missing, and explaining what each column means.
The result appears before you have asked a single question.

Every upload runs the same pipeline automatically:

1. **Upload** — the file lands in Avaloka's storage.
2. **Sampling** — a portfolio of representative samples is built so browsing and reasoning stay fast.
3. **Profiling** — every column is measured: type, missing values, distinct values, ranges.
4. **AI analysis** — column meanings and readiness are explained.

> **[Screenshot placeholder — `data-profile-card.png`]**
> *The profile card above a dataset.*

Click **Show More** for the full briefing:

- **Score and reasoning** — why the readiness score is what it is
- **Key Relationships** — pairs of columns expected to move together
- **Red Flags** — problems to know about before you trust your results
- **Quick Wins** — small clean-up steps worth doing first

> **[Screenshot placeholder — `quick-insights-expanded.png`]**
> *Quick Insights expanded: score, relationships, red flags, quick wins.*

### Prompting

Instead of building formulas, ask for what you want. Avaloka creates a plan
based on your prompt and decides which agents are needed — ask it to train a
model and the Model Training agent is routed in; ask it to reshape a dataset and
the Coder and Validator handle it.

Try something concrete:

> "Identify the fastest-growing channels by comparing subscribers_for_last_30_days to total subscribers. List the top 20 and include Youtuber, category, Country, subscribers, and subscribers_for_last_30_days."

When Avaloka finishes, the result appears in the output view as a **new
dataset**. Your original is untouched — every analysis produces a new dataset you
can keep refining.

> **[Screenshot placeholder — `prompt-and-output.png`]**
> *A prompt and the output dataset it produced.*

If a prompt produces no dataset — a question answered in prose, for example —
you will see the answer in the chat area instead.

> **[Screenshot placeholder — `no-output-message.png`]**
> *The chat message shown when a prompt produces no output dataset.*

### Sampled or complete?

For speed, Avaloka explores on a sample. When you need an exact figure, say so:

> "How many distinct aisles are there?"        *— fast, sampled*
> "Now compute that on the entire dataset."    *— exact, full scan*

**Avaloka always tells you which it used.** This matters: a sampled
distinct-count is an estimate, and treating it as exact is how a confident wrong
answer happens.

---

## 4. Visualization

Avaloka creates charts for you automatically. Whenever you upload a dataset or
run an analysis, the **Auto Insights** panel fills with **three to five charts**
chosen to explain the data in front of you. You do not have to ask, or pick
chart types.

> **[Screenshot placeholder — `auto-insights-panel.png`]**
> *Auto Insights with automatically generated charts.*

Every prompt that produces a result dataset also refreshes the charts. Asking a
chart-shaped question steers what gets visualized:

> "Plot monthly revenue by region."
> "Chart trips by hour of day."
> "Scatter fare vs distance."
> "Compare holiday vs non-holiday sales visually."

Charts are interactive — hover for values, click the expand icon for fullscreen.

> **[Screenshot placeholder — `chart-expanded.png`]**
> *A chart opened fullscreen.*

| Chart | Best for |
| --- | --- |
| **Bar** | Comparing categories, such as sales per region |
| **Line** | Change over time or another ordered axis |
| **Pie** | How a whole splits into parts, for a small number of categories |

---

## 5. Connections

A connection stores the address and sign-in details for one place your data
lives — a database or a cloud storage bucket. You create it once and refer to it
afterwards by the name you gave it. Connections are created in the **Data
Sources** panel.

> Connections to external databases and clouds are a **paid capability**. File
> upload and analysis are available in every edition. See [EDITIONS.md](EDITIONS.md).

### Connecting to a database

In **Data Sources**, expand **SQL Datastore** → **Connections** → **New
Connection**.

> **[Screenshot placeholder — `sql-connection-form.png`]**
> *The Database Connection Details form.*

| Field | What to enter |
| --- | --- |
| Connection Name * | A name for this connection, e.g. `Production Database` |
| Customer ID * | A short identifier — letters, numbers and underscores only |
| Customer Name * | Your organisation's full name |
| Contact Email * | A contact address for this connection |
| Connection Type * | The purpose of this database, e.g. `production` |
| Database Type * | Select from the dropdown |
| Host * | The server address, e.g. `db.example.com` |
| Port | The port number, e.g. `5432` |
| Database Name * | The database to connect to |
| Username * | The sign-in username |
| Password * | The sign-in password |
| Query Timeout (seconds) | How long a query may run before stopping |
| Max Rows | The maximum rows a query returns |

Fields marked `*` are required. Click **Test & Register** — Avaloka tests the
connection before saving, so an incorrect detail is reported immediately rather
than during a later transfer.

> **[Screenshot placeholder — `connection-dashboard.png`]**
> *The Connection Dashboard after a successful register.*

### Connecting to cloud storage

Under the **Cloud** tab, click **New Connection**. Avaloka supports **AWS,
Azure, and GCP**.

> **[Screenshot placeholder — `cloud-connection-form.png`]**
> *The cloud connection modal.*

After **Test & Save Connection**, your connections are listed with actions: the
folder icon browses datasets inside the connection, the pen icon updates the
details.

> **[Screenshot placeholder — `cloud-browser.png`]**
> *Browsing datasets inside a cloud connection.*

---

## 6. Data Transfer

The **Data Transfer Agent (DTA)** moves data from one place to another — into a
database, into a cloud bucket, or into a different file format — and can clean
or reshape it on the way. You describe the transfer in chat; the agent writes
and runs it.

> "Transfer aisles.csv from the instacart-mba connection to the walmart connection under transfers/aisles_copy.csv."
> "Move my cleaned housing dataset to the nyc-taxi bucket connection."

> **[Screenshot placeholder — `dta-transfer-chat.png`]**
> *A transfer requested and confirmed in chat.*

### What it can read

| Source | Supported |
| --- | --- |
| Files | CSV, JSON, Parquet |
| Databases | MySQL, PostgreSQL |

CSV and Parquet can be read **directly from a cloud bucket**. JSON can only be
read after it has been loaded into Avaloka — if your JSON is in a bucket,
convert it to CSV or Parquet first.

### What it can write

| Destination | Supported |
| --- | --- |
| Files | CSV, TSV, JSON, Parquet, Excel (`.xlsx`), XML, Avro, ORC |
| Databases | MySQL, PostgreSQL |

All eight file formats are available when writing to a cloud bucket.

> **Read and write are not symmetric.** Avaloka writes eight file formats but
> reads three. If you need to transfer *from* a format not in the read list,
> load it into Avaloka first, then transfer.

### Databases you can query but not transfer into

You can register and query **SQLite, SQL Server, Oracle, and MariaDB**.
Transfers currently target **MySQL and PostgreSQL** only.

---

## 7. Model training and inference

The **Model Training Agent (MTA)** turns a dataset into a model you can test in
chat. Tell it what to predict, review its choices, approve training, then ask
for predictions.

### Worked example

**Dataset:** `housing.csv` (California Housing Prices)
**Target:** `median_house_value` · **Task:** regression

**1. Upload the dataset.** Confirm `median_house_value` appears as a column
before requesting training.

**2. Ask for training in one direct request:**

> "Train a regression model to predict median_house_value using longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, and ocean_proximity."

**3. Review the plan.** It shows the model type, target, selected features,
epochs, optimizer, learning rate, and batch size. **Training has not started
yet.**

> **[Screenshot placeholder — `mta-training-plan.png`]**
> *The training plan shown for review.*

**4. Change or expand it.** Ask *"Show the full detailed training plan"* for
everything, or amend it conversationally — *"Add SibSp as a feature too"*,
*"Change it to 5 epochs"*.

**5. Approve:**

> "Looks good — go ahead and train it."

> **[Screenshot placeholder — `mta-training-progress.png`]**
> *Training in progress.*

**6. Predict.** Provide example values in chat, or use the **Inference Panel**.

> **[Screenshot placeholder — `inference-panel.png`]**
> *The Inference Panel with a prediction returned.*

Before training begins, Avaloka screens the data for **leakage** — a feature
that gives away the answer. If it finds any, training is blocked and the finding
explained. A leaking model reports excellent scores and fails in production, so
this is a refusal worth having.

Every result is compared against a **trivial baseline**. If a model does not
beat it, Avaloka says so plainly — that is a real finding, not a failure.

---

## 8. Context Memory

Context Memory lets Avaloka remember your data, preferences, and past results
while you work, so each prompt builds on what came before. It works
automatically, and you can also tell it explicitly what to remember.

**What Avaloka remembers:**

- **Your dataset** — its columns and layout, so you do not re-describe it every prompt.
- **Your working style** — built up quietly as you work; for example, that you prefer medians, or always exclude null rows.
- **Things you ask it to remember** — any preference or fact you state explicitly. These are **protected**: newer memories never crowd them out.
- **Your past results** — charts, tables, and model results it can reuse instead of redoing the work.

> "Preference: always include row counts in any table you show me."
> "Use the cleaned dataset from before."
> "What did we find about fraud rates in earlier analyses?"

---

## 9. Scheduling

Append *when to run* to any prompt and Avaloka creates a scheduled job:

> "Every Monday 9am, recompute the fraud rate by ProductCD and store the result."

> **[Screenshot placeholder — `scheduler-task-created.png`]**
> *A scheduled job created from a prompt.*

For a repeating task, say how often to run it and how many times to repeat. Any
phrasing of the schedule works.

Ask for your tasks in chat at any time:

> "List my scheduled tasks."

> **[Screenshot placeholder — `scheduler-task-list.png`]**
> *The task list returned in chat.*

When a job finishes, its status message updates. If the job modified a dataset,
the output view refreshes automatically.

Model training, data transfers, distributed execution, and inference-service
startup all schedule their own jobs when requested through chat.

---

## 10. FAQ and troubleshooting

| What you see | What to do |
| --- | --- |
| Cannot sign in after registering | Verify your email first — the account is inactive until you do |
| No verification or reset email | Check spam; the likeliest cause is a different address at signup |
| A capability is refused | It may be gated by your edition — see [EDITIONS.md](EDITIONS.md) |
| Two answers give different numbers | One was sampled, one was complete. Ask "on the entire dataset" for the authoritative figure |
| Training will not start | The integrity screen found leakage. Read the finding — a leaking feature usually should be dropped |
| A model reports poor results | It did not beat the baseline. The signal may not be in the data |
| A transfer will not read my file | Check the read formats in §6 — reads are narrower than writes |
| Upload rejected | Files are limited to 100 MB; use a connection for larger data |

Always log out on shared devices: click your profile icon in the top-right and
select **Log Out**.

---

## Where to go next

| | |
| --- | --- |
| Every capability, format, endpoint and edition gate | [Technical User Guide](TECHNICAL_USER_GUIDE.md) |
| Installing and licensing | [INSTALL.md](INSTALL.md) · [EDITIONS.md](EDITIONS.md) |
| Calling Avaloka from code | [api.md](api.md) · [cli.md](cli.md) |
| Running it on Kubernetes | [deployment.md](deployment.md) |

---

*Avaloka is open source under the Apache License 2.0.*
