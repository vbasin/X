"""Public, synthetic-only release gate."""
import json
from datetime import timedelta
from html.parser import HTMLParser
from pathlib import Path
import socket
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.common.causal import assert_known_at_or_before
from src.demo import FEATURES, run_demo, synthetic_frame
from src.experiment.manifest import RunManifest

pytestmark = pytest.mark.publication


def test_synthetic_features_are_causal_and_reproducible():
    frame = synthetic_frame()
    pd.testing.assert_frame_equal(frame, synthetic_frame())
    assert frame.index.tz is not None
    assert len(frame) == 720
    assert np.isfinite(frame[FEATURES]).all().all()
    assert_known_at_or_before(
        pd.DatetimeIndex(frame.context_available_ts).asi8, frame.index.asi8,
    )
    prefix = synthetic_frame(days=4)
    pd.testing.assert_frame_equal(frame.loc[prefix.index, FEATURES], prefix[FEATURES])
    assert (frame.exit_ts - frame.index == np.timedelta64(60, "s")).all()


def test_demo_persists_contracts_and_costs(tmp_path, monkeypatch):
    import psycopg
    import psycopg_pool

    def forbidden(*args, **kwargs):
        pytest.fail("Offline demo attempted database access")

    monkeypatch.setattr(psycopg, "connect", forbidden)
    monkeypatch.setattr(psycopg_pool, "ConnectionPool", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    output = run_demo(tmp_path / "run")
    manifest = RunManifest.read(output)
    assert manifest.extra["synthetic_only"] is True
    assert manifest.extra["promotion_eligible"] is False
    assert manifest.validation.n_folds == 5
    predictions = pd.read_parquet(output / "oof_predictions.parquet")
    assert len(predictions) == 450
    assert not predictions.ts.duplicated().any()
    assert set(predictions.direction) <= {"up", "down", "flat"}
    ledger = pd.read_parquet(output / "trade_ledger.parquet")
    assert len(ledger) > 0
    assert (ledger.commission == 3.36).all()
    assert np.allclose(ledger.gross_dollars - ledger.commission, ledger.net_dollars)
    assert (ledger.entry_slippage_points == 0.05).all()
    assert (ledger.exit_slippage_points == 0.05).all()
    for fold in json.loads((output / "fold_metrics.json").read_text()):
        assert pd.Timestamp(fold["train_end"]) + timedelta(minutes=1) < pd.Timestamp(fold["test_start"])
    assert "SYNTHETIC DATA ONLY" in (output / "report.html").read_text()
    links = []

    class ReportLinks(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                links.extend(value for key, value in attrs if key == "href")

    ReportLinks().feed((output / "report.html").read_text())
    assert links
    assert all((output / relative).is_file() for relative in links)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_demo(output)
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_seed_leaves_no_output(tmp_path, seed):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="seed"):
        run_demo(output, seed=seed)
    assert not output.exists()


def test_different_seeds_change_the_synthetic_input():
    assert not synthetic_frame(41)[FEATURES].equals(synthetic_frame(42)[FEATURES])


@pytest.mark.parametrize("state", ["dirty", "detached"])
def test_update_helper_preserves_local_work_and_never_auto_pushes(tmp_path, state):
    repo = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Fixture",
        "-c", "user.email=fixture@example.invalid", "commit", "-q", "--allow-empty",
        "-m", "Fixture\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
    ], check=True)
    if state == "dirty":
        (repo / "preserve.txt").write_text("local work\n")
    else:
        subprocess.run(["git", "-C", str(repo), "switch", "--detach", "-q"], check=True)
    script = Path(__file__).resolve().parents[1] / "deploy/gpu/git_update.sh"
    result = subprocess.run(["bash", str(script), str(repo)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "ERROR:" in result.stdout
    if state == "dirty":
        assert (repo / "preserve.txt").read_text() == "local work\n"
    assert "git push" not in script.read_text()
    assert "reset" not in script.read_text()
