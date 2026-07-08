# Model Regression Detector

A CI-style pipeline that evaluates a customer-support email classifier
against a hand-labeled golden dataset, detects quality regressions between
prompt or model changes, and surfaces them before they ship.

The problem it solves: LLM-powered features degrade silently. A prompt
edit that looks harmless can drop accuracy by 10–15%. This system gives
you a reproducible, scored baseline and a diff between any two runs so
regressions show up as numbers, not surprise incidents.

---

## Quickstart

```bash
git clone <repo>
cd model-regression-detector

cp .env.example .env
# Edit .env — fill in OPENROUTER_API_KEY
# Get a key at https://openrouter.ai/keys

pip install -r requirements.txt

python cli.py run       # evaluate the golden dataset, save the run
python cli.py compare   # diff the two most recent runs
python cli.py drift     # detect slow degradation over the rolling window
```

`run` prints per-case predictions and a pass rate summary. If every case
errors (bad API key, no credits, network failure) the result is printed
but not saved — infrastructure failures are not recorded as baselines.

`compare` prints the pass-rate delta, any regressions or improvements
by case ID, per-category deltas, and per-case judge-score drift.

`drift` computes the rolling average pass rate over the last
`drift_window_runs` runs and fires a warning if it has dropped by more
than `drift_threshold` from the oldest run in the window — catching
gradual degradation that no single run's diff would surface.

`compare` also takes two optional flags:

```bash
python cli.py compare --report --alert
```

- `--report` renders a self-contained HTML diff report to
  `reports/<run_id>.html` (gitignored) — run metadata, a scorecard, a
  table of regressed cases (old vs. new category/summary), and a
  matplotlib pass-rate trend chart embedded as a base64 PNG, so the file
  opens offline with no external assets.
- `--alert` posts a Slack message (badge, headline numbers, drift status)
  to the webhook at `SLACK_WEBHOOK_URL` in `.env`. If that variable is
  unset, the same JSON payload is printed to the console instead of
  erroring, so the pipeline is demoable without a real webhook.

`--config` works on all three commands; `run` additionally accepts
`--dataset` and `--prompt`:

```bash
python cli.py --config config.yaml run \
    --dataset data/golden_v1.json \
    --prompt prompts/email_classifier_v1.yaml
```

---

## Architecture

Each run follows this path:

1. **Classifier** (`src/classifier.py`) — sends each email through the
   versioned prompt (`prompts/email_classifier_v1.yaml`) and parses the
   JSON response (`{"category": ..., "summary": ...}`) into a typed
   `ClassifierOutput`.
2. **Async runner** (`src/runner.py`) — evaluates all cases concurrently
   up to `concurrency_limit`, with per-case error isolation so one LLM
   failure doesn't abort the run.
3. **Scoring** (`src/scoring.py`) — compares predicted vs. expected
   category (exact match) and calls a separate judge model
   (`gpt-4o`) to score the generated summary against the reference
   summary on a 1–5 rubric.
4. **Storage** (`src/storage.py`) — writes the full `RunResult` to
   `results/<run_id>.json` and a summary row to `runs.db` (SQLite).
   Refuses to persist a run where every case errored; those are
   infrastructure failures, not quality measurements.
5. **Diff** (`src/diff.py`) — computes pass-rate delta, per-category
   deltas, regression/improvement lists, and an alert level
   (`ok` / `warning` / `critical`) against configured thresholds.
6. **Drift** (`src/drift.py`) — reads the last N run summaries from
   SQLite and computes a rolling average pass rate. Fires a warning when
   the average has dropped by more than `drift_threshold` from the oldest
   run in the window, catching gradual decline that no single diff catches.
7. **Report** (`src/report.py`) — renders a `RunComparison` and the two
   `RunResult`s into an offline HTML file: metadata, scorecard, regressed
   cases, and a trend chart built from the last `report_trend_runs` runs.
8. **Alerts** (`src/alerts.py`) — posts a structured status message to a
   Slack incoming webhook, or prints the same payload to console if no
   webhook is configured.

All LLM calls go through one function — `call_llm_full` in `src/llm.py`.
This keeps the provider, model, and token-budget logic in one place;
swapping from OpenRouter to another OpenAI-compatible endpoint is a
one-line config change.

---

## Golden dataset and labeling philosophy

The golden dataset (`data/golden_v1.json`) contains 15 hand-labeled
cases covering the four categories, three difficulty levels, and several
edge conditions:

| Category    | Cases | What it covers |
|-------------|-------|----------------|
| `billing`   | 4     | Overcharges, failed promo codes, duplicate charges, subscription billing after cancellation |
| `technical` | 3     | App crashes (with device info), defective hardware, intermittent verification loop |
| `account`   | 3     | Password reset failure, 2FA lockout on new device, post-password-change login failure |
| `general`   | 5     | Shipping/returns policy, wrong-size return, missing shipment, minimal-signal ticket, positive feedback |

The `general` category is a deliberate catch-all: it covers questions,
logistics, policy inquiries, compliments, and any ticket that doesn't fit
a narrower category. The hard cases (`gc-010`: "its not working";
`gc-011`: heavy misspellings; `gc-012`: mixed English/Spanish) test
robustness to real-world input quality.

**Labels are human ground truth and are never auto-generated.** The
`expected_category` and `expected_summary` fields exist to be written and
verified by a person. The `notes` field on each case records the labeling
rationale, including which cases are genuinely ambiguous and why a
particular category was chosen over alternatives. Generating labels from
the model being tested defeats the purpose of an independent evaluation.

---

## Evaluation methodology and noise floor

Understanding what movement in the metrics actually means requires knowing
the measurement noise first. We ran two identical back-to-back evaluations
(same dataset, same prompt, same model, no code changes) to establish the
noise floor:

**Classifier noise (gpt-4o-mini, temperature=0.0):** zero category flips
across all 15 cases in both runs. At temperature zero the classifier is
fully deterministic — any change in `pass_rate` or `category_match`
between two runs reflects a real difference in the prompt or model, not
sampling noise.

**Judge noise (gpt-4o, temperature=0.0):** one case (`gc-011`, the
misspelled missing-shipment email) scored 5 in run 1 and 4 in run 2 —
a ±1-point drift on a borderline case. Across 15 cases this shifts
`mean_judge_score` by ~0.067. Judge scores on clear cases are stable;
treat ±1 on a single case as noise, not signal.

**Threshold rationale:**

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| `warning_threshold` | 0.05 | One case flipping on this dataset = 0.067 delta. Warning fires at 0.05 — above the 0.0 category noise floor, so any warning is a real signal. |
| `critical_threshold` | 0.15 | Approximately two or more cases flipping; warrants immediate investigation before shipping. |

Alert level uses `abs(delta)` so a surprising improvement also fires —
a large positive swing often means the dataset or labels changed, not that
the model got better.

**Baseline results** (golden_v1, prompt v1.0.0, gpt-4o-mini classifier):
- Pass rate: **93.3%** (14/15)
- Mean judge score: **~4 / 5** (judge scores vary ±1 on borderline
  cases; see [Evaluation methodology and noise floor](#evaluation-methodology-and-noise-floor))
- One miss: `gc-010` ("its not working") — predicted `technical`,
  expected `general`. This is the hard minimal-signal case; the miss is
  expected at this dataset size.

---

## Configuration

`config.yaml` controls all tunables. Every field can be overridden with
an environment variable.

```yaml
warning_threshold: 0.05    # env: WARNING_THRESHOLD
critical_threshold: 0.15   # env: CRITICAL_THRESHOLD
drift_window_runs: 7        # env: DRIFT_WINDOW_RUNS
drift_threshold: 0.05       # env: DRIFT_THRESHOLD
report_trend_runs: 20       # env: REPORT_TREND_RUNS
concurrency_limit: 5        # env: CONCURRENCY_LIMIT
classifier_model: openai/gpt-4o-mini  # env: CLASSIFIER_MODEL
judge_model: openai/gpt-4o            # env: JUDGE_MODEL
classifier_max_tokens: 256  # env: CLASSIFIER_MAX_TOKENS
judge_max_tokens: 16        # env: JUDGE_MAX_TOKENS
```

`classifier_max_tokens: 256` is enough for any valid JSON
`{"category": ..., "summary": ...}` response.
`judge_max_tokens: 16` is enough for a single digit reply.
These caps matter on pay-per-token providers where an uncapped
`max_tokens` can exhaust account credit before the first token is
generated.

Models are referenced by OpenRouter path (`provider/model-name`).
Switching to a different provider requires only changing `base_url` in
`src/llm.py` and updating the model strings.

---

## Docker

Build once, run without installing anything locally:

```bash
docker build -t model-regression-detector .
```

Secrets and config overrides are read from the environment at `docker run`
time — never baked into the image. Mount `results/`, `reports/`, and
`runs.db` as volumes so run history persists across containers:

```bash
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-... \
  -e SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... \
  -e CRITICAL_THRESHOLD=0.10 \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/runs.db:/app/runs.db" \
  model-regression-detector run

docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-... \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/runs.db:/app/runs.db" \
  model-regression-detector compare --report
```

Without the volume mounts, each container starts with an empty baseline
history.

---

## CI: regression gate on prompt/dataset PRs

`.github/workflows/eval.yml` runs on any pull request that touches
`prompts/**` or `data/**`. It evaluates the golden dataset twice — once
against the base branch's prompt/dataset (the "before" baseline) and once
against the PR's changed version (the "after") — using the same classifier
code both times, so the pass-rate delta is attributable to the prompt or
dataset change itself. It then posts a summary comment on the PR (status
badge, pass-rate before/after, regressions/improvements) and **fails the
job — blocking merge — if the alert level is critical**.

The Slack-alert step is `continue-on-error` — a webhook outage or
misconfiguration is logged but never blocks merge. Only the eval result
itself (`alert_level == critical`) can fail the job; that check runs
`if: always()` so it fires regardless of whether the Slack step succeeded.

**Required repo secrets** (Settings → Secrets and variables → Actions →
New repository secret):

| Secret | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Classifier + judge LLM calls. Without it the job fails immediately. |
| `SLACK_WEBHOOK_URL` | No | Also posts the same alert to Slack. If unset, the workflow prints the payload to the job log instead of erroring (same console fallback as running locally). |

`GITHUB_TOKEN` (used to post the PR comment) is provided automatically by
Actions — no setup needed.

---

## Tests

```bash
pytest
```

64 unit tests, no network calls. The test suite covers:

- **`tests/test_runner.py`** — error isolation (LLM exception → `status="error"`,
  not a crashed run), error cases excluded from `pass_rate`, fractional
  pass rates, all-error edge case.
- **`tests/test_diff.py`** — alert level boundaries (at and around both
  thresholds), regression/improvement detection (correct→wrong,
  wrong→correct, error transitions, stable cases), per-category delta
  calculation, unlabeled and new-case handling.
- **`tests/test_drift.py`** — rolling average arithmetic, threshold
  boundary (exactly at, just below), gradual decline scenario, improving
  trend, judge score averaging with `None` values, insufficient-data
  edge cases.
- **`tests/test_report.py`** — every HTML section is present (metadata,
  scorecard, regressed cases, trend), regressed-case rows show old vs.
  new category/summary, trend chart falls back to a placeholder message
  under 2 runs and embeds a valid base64 PNG otherwise, HTML-escaping of
  case content.
- **`tests/test_alerts.py`** — Slack payload shape and badge per alert
  level, drift status inclusion, webhook POST vs. console fallback when
  `SLACK_WEBHOOK_URL` is unset (including the `.env.example` placeholder
  URL), webhook failure raises.

All LLM calls and webhook POSTs are mocked; tests run offline and complete
in under 5 seconds.

---

## Repository layout

```
.github/
  workflows/
    eval.yml                 # PR regression gate for prompts/**, data/**
Dockerfile                   # slim Python 3.12 image; ENTRYPOINT -> cli.py
.dockerignore                # excludes .venv, .env, results/, reports/, runs.db, __pycache__
cli.py                      # entry point: run, compare, drift
config.yaml                 # tunables and noise-floor documentation
data/
  golden_v1.json            # 15 hand-labeled evaluation cases
prompts/
  email_classifier_v1.yaml  # versioned prompt config with few-shot examples
results/                    # one JSON file per run (gitignored)
reports/                    # one HTML report per --report run (gitignored)
runs.db                     # SQLite summary index (gitignored)
src/
  llm.py                    # single LLM wrapper (call_llm_full)
  classifier.py             # prompt builder and output parser
  dataset.py                # golden dataset loader and validator
  runner.py                 # async eval orchestrator
  scoring.py                # category match + LLM-as-judge
  diff.py                   # per-run comparison and alert logic
  drift.py                  # rolling-window slow-drift detector
  report.py                 # offline HTML diff report generator
  alerts.py                 # Slack webhook alerts (console fallback)
  storage.py                # JSON + SQLite persistence; guards against all-error runs
  models.py                 # Pydantic contracts for all I/O
  config.py                 # config loader with env overrides
  prompts.py                # prompt YAML loader
tests/
  test_runner.py
  test_diff.py
  test_drift.py
  test_report.py
  test_alerts.py
```
