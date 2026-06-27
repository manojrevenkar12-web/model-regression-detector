# Model Regression Detection System

A CI-style pipeline that tests an LLM feature (customer-support email
classifier) against a hand-labeled golden dataset, detects quality
regressions between runs, and alerts via Slack.

## Stack
- Python 3.12, Pydantic for all I/O contracts
- LLM via OpenRouter (OpenAI-compatible), model gpt-4o-mini
- Storage: JSON files + SQLite, git-friendly
- Slack incoming webhooks for alerts
- pytest for the test suite

## Hard rules
1. NEVER auto-generate golden-dataset labels. The expected category and
   summary for each test case are human ground truth, written by me.
   You may draft candidate email *inputs*, but I write/verify the labels.
2. Never commit secrets. API keys live in .env (gitignored).
3. Ask before adding any new dependency.
4. Keep every LLM call behind one wrapper function so the provider/model
   is swappable in one place.

## Conventions
- Conventional Commits (feat:, fix:, refactor:, test:, docs:, chore:).
- Type hints everywhere. Async batching for LLM calls.
- README reads like onboarding docs for a new teammate, not a tutorial.