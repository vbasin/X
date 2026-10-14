"""Tests: ResourceGovernor admission + runner continuous-scheduler helpers."""

from __future__ import annotations

import os
from unittest import mock

import pytest

import src.gpu_runner.runner as runner
from src.gpu_runner.runner import JobAlloc, ResourceGovernor


def _gov(cpus=30, gpu_slots=2, mem_gb=200.0) -> ResourceGovernor:
    with mock.patch.object(os, "cpu_count", return_value=cpus), \
         mock.patch.object(runner, "_gpu_slot_capacity", return_value=gpu_slots), \
         mock.patch.object(runner, "_mem_total_gb", return_value=mem_gb):
        return ResourceGovernor()


def test_dl_job_gets_gpu_slot_and_bounded_cpus():
    g = _gov()
    alloc = g.try_admit({"family": "attention_lstm"})
    assert alloc is not None and alloc.gpu_slots == 1
    assert 2 <= alloc.cpus <= 8
    assert g.gpu_used == 1


def test_tree_job_takes_gpu_slot_opportunistically_else_cpu():
    g = _gov()
    a1 = g.try_admit({"family": "xgboost"})
    a2 = g.try_admit({"family": "attention_lstm"})
    a3 = g.try_admit({"family": "xgboost"})
    assert a1.gpu_slots == 1            # free slot → cuda
    assert a2.gpu_slots == 1            # DL takes the second slot
    assert a3 is not None and a3.gpu_slots == 0   # no slot left → CPU fallback, not held


def test_second_dl_job_waits_for_gpu_slot():
    g = _gov(gpu_slots=1)
    assert g.try_admit({"family": "attention_lstm"}).gpu_slots == 1
    assert g.try_admit({"family": "tcn"}) is None          # held until the slot frees
    g.release(JobAlloc(cpus=6, mem_gb=24.0, gpu_slots=1, family="attention_lstm",
                       compute_class="gpu"))
    assert g.try_admit({"family": "tcn"}).gpu_slots == 1


def test_cpu_ledger_blocks_then_releases():
    g = _gov(cpus=12, gpu_slots=0)
    a1 = g.try_admit({"family": "hmm"})
    assert a1 is not None
    # Exhaust the CPU ledger
    allocs = [a1]
    while (a := g.try_admit({"family": "hmm"})) is not None:
        allocs.append(a)
        if len(allocs) > 10:
            break
    assert g.try_admit({"family": "hmm"}) is None
    for a in allocs:
        g.release(a)
    assert g.cpu_used == 0 and g.try_admit({"family": "hmm"}) is not None


def test_empty_ledger_always_admits_oversized_job():
    g = _gov(cpus=2, gpu_slots=0, mem_gb=4.0)   # tiny host
    alloc = g.try_admit({"family": "xgboost"})   # cost exceeds host — must still admit
    assert alloc is not None and alloc.cpus >= 1


def test_smoke_jobs_are_cheap():
    g = _gov()
    alloc = g.try_admit({"smoke": True})
    assert alloc.cpus == 1 and alloc.gpu_slots == 0


def test_unknown_family_fallback_profile():
    cc, cost = runner._family_profile("no_such_family_xyz")
    assert cc in ("cpu", "gpu")


def test_nvidia_lib_paths_shape():
    libs = runner._nvidia_lib_paths()
    assert isinstance(libs, str)
    if libs:
        assert "nvidia" in libs and ":" not in libs.split(":")[0][:1]


def test_execute_env_injection_no_alloc_keeps_signature(tmp_path):
    """_execute_bundle_job(store, spec) still works positionally (existing tests rely on it)."""
    import inspect
    sig = inspect.signature(runner._execute_bundle_job)
    params = list(sig.parameters)
    assert params[:2] == ["store", "spec"]
    assert sig.parameters["alloc"].kind is inspect.Parameter.KEYWORD_ONLY
