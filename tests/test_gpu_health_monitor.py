"""Tests for the CPU-side GPU health watchdog (ADR-022 companion) — no SSH / no OCI.

Covers report parsing and the anomaly-evaluation rules (the part that decides what counts as an
error the queue poll would miss).
"""

from __future__ import annotations

from src.gpu_runner.health_monitor import (
    evaluate,
    format_progress_line,
    parse_report,
    progress_jobs,
    progress_payload,
    select_current_progress,
)

_HEALTHY = {
    "RUNNER_ACTIVE": "active",
    "RUNNER_SUBSTATE": "running",
    "RUNNER_RESULT": "success",
    "RUNNER_ERRORS": "0",
    "LASTERR": "",
    "OOM_KILLS": "0",
    "GPU_OK": "1",
    "GPU_UTIL": "37",
    "DISK_USE_PCT": "60",
}


def test_parse_report_keeps_equals_in_values() -> None:
    raw = "RUNNER_ACTIVE=active\nLASTERR=Traceback: x = y failed\nDISK_USE_PCT=60\nREPORT_END=1"
    rep = parse_report(raw)
    assert rep["RUNNER_ACTIVE"] == "active"
    assert rep["LASTERR"] == "Traceback: x = y failed"     # value with '=' preserved
    assert rep["DISK_USE_PCT"] == "60"


def test_healthy_report_has_no_anomalies() -> None:
    assert evaluate(dict(_HEALTHY)) == []


def test_runner_failed_unit_flagged() -> None:
    rep = {**_HEALTHY, "RUNNER_ACTIVE": "failed", "RUNNER_RESULT": "exit-code"}
    anomalies = evaluate(rep)
    assert any("runner unit unhealthy" in a for a in anomalies)


def test_journal_errors_flagged_with_sample() -> None:
    rep = {**_HEALTHY, "RUNNER_ERRORS": "2", "LASTERR": "Traceback (most recent call last)"}
    anomalies = evaluate(rep)
    assert any("error line(s) in the runner journal" in a and "Traceback" in a for a in anomalies)


def test_oom_kill_flagged() -> None:
    rep = {**_HEALTHY, "OOM_KILLS": "1"}
    assert any("OOM-kill" in a for a in evaluate(rep))


def test_gpu_fault_flagged() -> None:
    rep = {**_HEALTHY, "GPU_OK": "0"}
    assert any("nvidia-smi failed" in a for a in evaluate(rep))


def test_disk_full_flagged_at_threshold() -> None:
    assert evaluate({**_HEALTHY, "DISK_USE_PCT": "90"}) != []
    assert evaluate({**_HEALTHY, "DISK_USE_PCT": "89"}) == []


def test_runner_inactive_needs_streak_to_flag() -> None:
    # A single inactive read (boot / normal self-stop window) must NOT alarm …
    rep = {**_HEALTHY, "RUNNER_ACTIVE": "inactive", "RUNNER_RESULT": "success"}
    assert evaluate(rep, inactive_streak=1) == []
    # … but a persistent one (VM still billing with no runner) must.
    anomalies = evaluate(rep, inactive_streak=3)
    assert any("still billing" in a for a in anomalies)


def test_missing_numeric_fields_do_not_crash() -> None:
    rep = {"RUNNER_ACTIVE": "active", "RUNNER_RESULT": "success"}
    assert evaluate(rep) == []      # absent RUNNER_ERRORS/OOM/DISK treated as 0/ok


def test_progress_json_decoded_and_summarized() -> None:
    raw = (
        'RUNNER_ACTIVE=active\n'
        'PROGRESS=[{"run_id":"r1","family":"attention_lstm","status":"training",'
        '"fold_i":2,"n_folds":6,"epoch_i":8,"n_epochs":15,"pct_complete":31.1,'
        '"hit_rate":0.4}]\n'
        'REPORT_END=1\n'
    )
    rep = parse_report(raw)
    jobs = progress_jobs(rep)
    assert jobs[0]["family"] == "attention_lstm"
    line = format_progress_line(jobs)
    assert "attention_lstm" in line
    assert "fold=3/6" in line
    assert "epoch=8/15" in line
    assert "31.1%" in line


def test_failed_job_progress_is_anomaly() -> None:
    import json
    jobs = [{"family": "tcn", "status": "failed", "run_id": "x", "error": "CUDA OOM"}]
    rep = {**_HEALTHY, "PROGRESS": json.dumps(jobs)}
    anomalies = evaluate(rep)
    assert any("status=failed" in a and "CUDA OOM" in a for a in anomalies)


def test_select_current_progress_prefers_training_over_recent_upload() -> None:
    jobs = [
        {"family": "xgboost", "status": "uploading", "age_s": 90, "mtime": 8,
         "pct_complete": 100.0, "hit_rate": 0.14},
        {"family": "attention_lstm", "status": "training", "age_s": 20, "mtime": 9,
         "n_folds": 3, "fold_i": 1, "n_epochs": 15, "epoch_i": 2, "pct_complete": 37.8},
    ]
    cur = select_current_progress(jobs)
    assert [j["family"] for j in cur] == ["attention_lstm"]


def test_select_current_progress_drops_stale_uploads() -> None:
    jobs = [
        {"family": "attention_lstm", "status": "uploading", "age_s": 12_000,
         "n_folds": 3, "fold_i": 2, "n_epochs": 3, "epoch_i": 3, "pct_complete": 100.0},
        {"family": "xgboost", "status": "uploading", "age_s": 8_000,
         "n_folds": 3, "fold_i": 2, "pct_complete": 100.0, "hit_rate": 0.14},
        {"family": "attention_lstm", "status": "training", "age_s": 20,
         "n_folds": 3, "fold_i": 1, "n_epochs": 15, "epoch_i": 2, "pct_complete": 37.8,
         "hit_rate": 0.1097},
    ]
    cur = select_current_progress(jobs)
    assert len(cur) == 1
    assert cur[0]["status"] == "training"
    assert cur[0]["n_epochs"] == 15
    line = format_progress_line(jobs, n_on_disk=6)
    assert "training" in line
    assert "epoch=2/15" in line
    assert "uploading" not in line
    assert "1 shown / 6 on disk" in line
    assert not line.startswith("last:")


def test_select_current_progress_idle_shows_newest_last() -> None:
    jobs = [
        {"family": "xgboost", "status": "uploading", "age_s": 5000, "mtime": 1,
         "pct_complete": 100.0},
        {"family": "attention_lstm", "status": "uploading", "age_s": 100, "mtime": 9,
         "n_folds": 3, "fold_i": 2, "n_epochs": 3, "epoch_i": 3, "pct_complete": 100.0},
    ]
    # 100s < 900s live window, so the recent upload is still "current"
    cur = select_current_progress(jobs)
    assert len(cur) == 1
    assert cur[0]["family"] == "attention_lstm"
    stale = [
        {**jobs[0], "age_s": 12_000, "mtime": 1},
        {**jobs[1], "age_s": 11_000, "mtime": 2, "family": "attention_lstm"},
    ]
    cur = select_current_progress(stale)
    assert len(cur) == 1
    line = format_progress_line(stale, n_on_disk=5)
    assert line.startswith("last:")
    assert "5 on disk" in line


def test_progress_payload_accepts_wrapped_jobs() -> None:
    import json
    raw = json.dumps({
        "n_on_disk": 6,
        "jobs": [{
            "family": "attention_lstm", "status": "training", "age_s": 5,
            "fold_i": 1, "n_folds": 3, "epoch_i": 2, "n_epochs": 15, "pct_complete": 37.8,
        }],
    })
    jobs, n = progress_payload({**_HEALTHY, "PROGRESS": raw})
    assert n == 6
    assert jobs[0]["status"] == "training"
    assert progress_jobs({**_HEALTHY, "PROGRESS": raw})[0]["family"] == "attention_lstm"


def test_stale_failed_leftover_is_not_anomaly_when_training() -> None:
    import json
    jobs = [
        {"family": "tcn", "status": "failed", "run_id": "old", "error": "CUDA OOM", "age_s": 20_000},
        {"family": "attention_lstm", "status": "training", "run_id": "now", "age_s": 8,
         "fold_i": 0, "n_folds": 3, "epoch_i": 1, "n_epochs": 15},
    ]
    anomalies = evaluate({**_HEALTHY, "PROGRESS": json.dumps(jobs)})
    assert anomalies == []
