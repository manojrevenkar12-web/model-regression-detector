# Packages the eval runner + golden dataset + reporting/alerting layer.
# Secrets (OPENROUTER_API_KEY, SLACK_WEBHOOK_URL) and threshold overrides are
# read from the environment at `docker run` time — nothing is baked into the
# image. See config.py for the full list of env-overridable settings
# (WARNING_THRESHOLD, CRITICAL_THRESHOLD, DRIFT_WINDOW_RUNS, DRIFT_THRESHOLD,
# REPORT_TREND_RUNS, CONCURRENCY_LIMIT, CLASSIFIER_MODEL, JUDGE_MODEL,
# CLASSIFIER_MAX_TOKENS, JUDGE_MAX_TOKENS).
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cli.py config.yaml ./
COPY src/ ./src/
COPY data/ ./data/
COPY prompts/ ./prompts/

# results/, reports/, and runs.db are written at runtime — mount a volume at
# /app to persist them across containers (see README for the run command).

ENTRYPOINT ["python", "cli.py"]
