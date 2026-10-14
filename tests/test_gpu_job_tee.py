"""ADR-073 live job_exec log tee (no GPU, no bundle)."""

from __future__ import annotations

import sys
from pathlib import Path

import src.gpu_runner.runner as runner


def test_tee_subprocess_writes_live_log(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    extra = tmp_path / "durable" / "job.log"
    rc = runner._tee_subprocess(
        [sys.executable, "-c", "import sys; print('hello-fold'); print('epoch 1', file=sys.stderr)"],
        cwd=str(tmp_path),
        env=dict(**{k: v for k, v in __import__("os").environ.items()}),
        log_path=log,
        extra_logs=[extra],
    )
    assert rc == 0
    text = log.read_text()
    assert "hello-fold" in text
    assert "epoch 1" in text  # stderr merged
    assert extra.read_text() == text


def test_job_exec_failed_carries_out_dir(tmp_path: Path) -> None:
    err = runner.JobExecFailed("boom", out_dir=tmp_path)
    assert err.out_dir == tmp_path
    assert "boom" in str(err)
