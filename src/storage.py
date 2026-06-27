import sqlite3
from pathlib import Path

from src.models import RunResult

_RESULTS_DIR = Path("results")
_DB_PATH = Path("runs.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id          TEXT PRIMARY KEY,
            run_at          TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            prompt_version  TEXT NOT NULL,
            classifier_model TEXT NOT NULL,
            judge_model     TEXT NOT NULL,
            total_cases     INTEGER NOT NULL,
            labeled_cases   INTEGER NOT NULL,
            error_cases     INTEGER NOT NULL,
            pass_rate       REAL,
            mean_judge_score REAL,
            mean_latency_ms REAL
        )
    """)
    conn.commit()


def save_run(run: RunResult) -> Path:
    """Persist full RunResult to JSON and write summary row to SQLite."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    json_path = _RESULTS_DIR / f"{run.run_id}.json"
    json_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    labeled = [c for c in run.cases if c.expected_category]
    error_count = sum(1 for c in run.cases if c.status == "error")

    with _connect() as conn:
        _init_db(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, run_at, dataset_version, prompt_version, classifier_model,
             judge_model, total_cases, labeled_cases, error_cases,
             pass_rate, mean_judge_score, mean_latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.run_at.isoformat(),
                run.dataset_version,
                run.prompt_version,
                run.classifier_model,
                run.judge_model,
                len(run.cases),
                len(labeled),
                error_count,
                run.pass_rate,
                run.mean_judge_score,
                run.mean_latency_ms,
            ),
        )

    return json_path


def load_run(run_id: str) -> RunResult:
    """Load a full RunResult from its JSON file."""
    json_path = _RESULTS_DIR / f"{run_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Run result not found: {json_path}")
    return RunResult.model_validate_json(json_path.read_text(encoding="utf-8"))


def list_runs(limit: int = 50) -> list[dict]:
    """Return run summary rows from SQLite, newest first."""
    with _connect() as conn:
        _init_db(conn)
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY run_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
