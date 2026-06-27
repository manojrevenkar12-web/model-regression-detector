MODEL REGRESSION DETECTION SYSTEM
What You're Building
AI WITH UNNATI
Projects
Tips
Resources
Instagram
A CI/CD-style pipeline that continuously tests any LLM-powered feature against a
golden dataset whenever a prompt or model changes, detects quality regressions, and alerts your team via Slack before bad outputs reach users.
WHY THIS PROJECT LANDS INTERVIEWS
Every AI team ships prompt changes blind. This project proves you think about what happens after deployment, the exact mindset hiring managers are desperate for and almost no candidates demonstrate.
https://unnati-23.github.io/ai-engineer-courses-guide/15-ai-engineering-projects-that-land-jobs.html?utm_source=sp_auto_dm&fbclid=PAVERFW…
4/656/21/26, 1:41 PM
15 AI Engineering Projects That Actually Land Jobs | Unnati Tripathi
Tech Stack
Component
Tool / Library
Why This Choice
Language
Python 3.11+
Industry standard for ML tooling
LLM Provider
OpenAI API (gpt-4o / gpt-4o- mini)
Custom + RAGAS or DeepEval
Widely recognized; easy to swap later
Eval Framework
Shows you understand eval beyond
accuracy
Data Storage
SQLite + JSON files
Zero infrastructure, portable, git-friendly
Alerting
Slack Webhooks
What real teams actually use
Scheduling
GitHub Actions
Runs on every PR; free tier is enough
Visualization
Streamlit or simple HTML report
Quick dashboard for diff views
Containerization
Docker
Shows production readiness
Step-by-Step Build Guide
Define the LLM Feature Under Test
(Day 1–2)
1       Build a simple LLM feature:
A customer support email classifier that reads an
email and returns a category (billing, technical, account, general) plus a one- sentence summary. Wrap it in a single Python function with the prompt as a configurable parameter.
2       Version your prompts:
Store prompts as versioned YAML files in a /prompts
directory. Each file has a version ID, timestamp, the system prompt, and any few-shot examples. This is the “code” you're running CI against.
3       Create the interface contract:
Define a simple PromptConfig dataclass that
your eval pipeline consumes. Input: email text. Output: structured JSON with category and summary. Keep it typed with Pydantic.
Build the Golden Dataset
(Day 2–4)
1       Curate 50–100 test cases by hand:
Write real-looking customer emails across
all categories. For each one, write the correct category and an ideal summary. Do NOT generate these with an LLM, the whole point is that these are human- verified ground truth.
https://unnati-23.github.io/ai-engineer-courses-guide/15-ai-engineering-projects-that-land-jobs.html?utm_source=sp_auto_dm&fbclid=PAVERFW…
5/656/21/26, 1:41 PM
15 AI Engineering Projects That Actually Land Jobs | Unnati Tripathi
2       Include edge cases deliberately:
Add ambiguous emails that could be two
categories, extremely short emails, emails with typos, emails in mixed languages, sarcastic emails. Label these with an “expected_difficulty” field.
3       Store as versioned JSON:
Each test case gets a stable ID, the input, expected
output, difficulty tag, and a notes field explaining why this case matters. Version the dataset file itself so you can track when the eval bar changes.
Build the Evaluation Engine
(Day 4–7)
1       Create the test runner:
Write a function that takes a PromptConfig and the
golden dataset, runs every test case through the LLM feature, and collects raw outputs. Use async batching to keep costs low and speed high.
2       Implement multi-dimensional scoring:
Don't just check if the category
matches. Score on: exact category match (binary), summary relevance (use an LLM-as-judge to rate 1–5), latency per request, and token usage. Store all dimensions per test case.
3       Build the comparison logic:
The core value of this system is diffing. For every
eval run, compare against the previous run. Calculate: overall pass rate delta, per-category accuracy delta, list of specific cases that flipped from pass to fail (regressions), and cases that flipped from fail to pass (improvements).
4       Add statistical significance:
If 2 out of 80 cases flipped, is that signal or noise?
Implement a simple threshold system: flag as warning if delta exceeds 3%, flag as critical if delta exceeds 8%. Make these configurable.
Build the Alerting and Reporting Layer
(Day 7–9)
1       Create the diff report:
Generate an HTML report that shows: run metadata
(prompt version, model, timestamp), a summary scorecard comparing this run
to the baseline, a table of every regressed case showing the old output vs. the
new output side by side, and a trend chart showing scores over the last N runs.
2       Wire up Slack alerts:
Use Slack's incoming webhook API. Send a structured
message with: pass/warn/fail status, the headline numbers (e.g., “3 regressions detected, accuracy dropped from 94% to 89%”), and a link to the full HTML diff report.
3       Add drift detection:
Beyond per-run diffs, track a rolling average of scores over
time. If the 7-run moving average drops below a threshold even though no single
https://unnati-23.github.io/ai-engineer-courses-guide/15-ai-engineering-projects-that-land-jobs.html?utm_source=sp_auto_dm&fbclid=PAVERFW…
6/656/21/26, 1:41 PM
15 AI Engineering Projects That Actually Land Jobs | Unnati Tripathi
run triggered an alert, fire a “slow drift” warning. This catches gradual degradation that per-run checks miss.
Wire into CI/CD
(Day 9–11)
1       Create a GitHub Action workflow:
Trigger the eval pipeline on every PR that
modifies files in the /prompts directory. The action should run the eval, generate the report, post a summary comment on the PR with pass/fail status, and block merge if critical regressions are detected.
2       Containerize everything:
Write a Dockerfile that packages the eval runner, the
golden dataset, and the reporting layer. The container should accept environment variables for the LLM API key, Slack webhook URL, and threshold configs.
3       Write a README that reads like internal documentation:
Include a one-
paragraph summary of what this does, setup instructions, how to add new test cases to the golden dataset, how to adjust thresholds, and architecture decisions with rationale. Do NOT write it like a tutorial. Write it like onboarding docs for a new teammate joining your team.
Polish for Portfolio
(Day 11–12)
1       Record a 3-minute Loom walkthrough:
Show the system running end to end,
change a prompt, trigger the eval, show the Slack alert, walk through the diff report. This is more persuasive than any README.
2       Write a short blog post or README section:
Explain the problem (teams ship
prompt changes blind), your approach (CI/CD for model behavior), and one specific design decision you're proud of (e.g., why you track slow drift separately from per-run regressions).
INTERVIEW TALKING POINT
Lead with how you built the golden dataset. Explain that you seeded it with hand-labeled data, then expanded it over time using failure cases. This signals you understand that evaluation quality is bounded by data quality, a production insight most candidates miss entirely