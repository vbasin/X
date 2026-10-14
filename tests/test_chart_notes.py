"""Live Chart scenario notes persistence."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.webui import chart_notes
from src.webui import datasources as ds
from src.webui.app import create_app
from src.webui.jobs import JobManager


@pytest.fixture()
def notes_root(tmp_path, monkeypatch):
    root = tmp_path / "chart_notes"
    monkeypatch.setattr(chart_notes, "store_root", lambda: root)
    return root


@pytest.fixture()
def client(tmp_path, monkeypatch, notes_root):
    monkeypatch.setattr(ds, "SCOPING_ROOT", tmp_path / "scoping")
    monkeypatch.setattr(ds, "SEARCH_ROOT", tmp_path / "search")
    jm = JobManager(root=tmp_path / "jobs")
    with TestClient(create_app(jobs=jm, enable_live_worker=False), base_url="http://127.0.0.1") as c:
        yield c


def test_save_list_load_delete(notes_root):
    rec = chart_notes.save_note({
        "name": "NQU6 2026-09-17 09:30–10:15 CT",
        "text": "opening drive",
        "symbol": "NQU6",
        "instrument": "/NQ",
        "range": {
            "start": "2026-09-17T14:30:00Z",
            "end": "2026-09-17T15:15:00Z",
            "start_ct": "2026-09-17 09:30:00",
            "end_ct": "2026-09-17 10:15:00",
            "label": "NQU6 2026-09-17 09:30–10:15 CT",
        },
        "chart": {
            "timeframe": "1min",
            "overlays": ["ps_sar", "of_cvd"],
            "sar": {"start": 0.03, "increment": 0.02, "maximum": 0.2},
            "emas": [{"n": 1, "lookback": 21}],
        },
    })
    assert rec["id"]
    assert (notes_root / f"{rec['id']}.json").is_file()
    listed = chart_notes.list_notes()
    assert listed and listed[0]["id"] == rec["id"]
    loaded = chart_notes.load_note(rec["id"])
    assert loaded["text"] == "opening drive"
    assert loaded["chart"]["sar"]["start"] == 0.03
    chart_notes.delete_note(rec["id"])
    assert chart_notes.list_notes() == []


def test_save_note_keeps_price_box(notes_root):
    rec = chart_notes.save_note({
        "name": "NQZ6 box",
        "text": "offer pulled",
        "symbol": "NQZ6",
        "range": {
            "start": "2026-09-29T14:00:00Z",
            "end": "2026-09-29T14:30:00Z",
            "label": "NQZ6 09:00–09:30",
            "price_lo": 21450.25,
            "price_hi": 21480.5,
            "pane": 0,
        },
    })
    loaded = chart_notes.load_note(rec["id"])
    assert loaded["range"]["price_lo"] == 21450.25
    assert loaded["range"]["price_hi"] == 21480.5
    assert loaded["range"]["pane"] == 0
    with pytest.raises(ValueError, match="price_hi"):
        chart_notes.save_note({
            "name": "bad box",
            "range": {
                "start": "2026-09-29T14:00:00Z",
                "end": "2026-09-29T14:30:00Z",
                "price_lo": 10,
                "price_hi": 9,
            },
        })


def test_notes_api(client):
    headers = {"X-VTech-Console": "1"}
    created = client.post("/api/chart/notes", headers=headers, json={
        "name": "NQZ6 2026-09-24 08:30–09:00 CT",
        "text": "cash open",
        "symbol": "NQZ6",
        "range": {
            "start": "2026-09-24T13:30:00Z",
            "end": "2026-09-24T14:00:00Z",
            "label": "NQZ6 2026-09-24 08:30–09:00 CT",
        },
        "chart": {"timeframe": "1min", "overlays": ["ps_sar"]},
    })
    assert created.status_code == 201, created.text
    nid = created.json()["id"]
    listed = client.get("/api/chart/notes")
    assert listed.status_code == 200
    assert any(r["id"] == nid for r in listed.json())
    one = client.get(f"/api/chart/notes/{nid}")
    assert one.status_code == 200
    assert one.json()["text"] == "cash open"
    deleted = client.delete(f"/api/chart/notes/{nid}", headers=headers)
    assert deleted.status_code == 200
    assert client.get(f"/api/chart/notes/{nid}").status_code == 404
