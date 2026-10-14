"""VMConfig resource budgets track live RAM, not a previous OCI shape."""
from src.common.vm_config import VMConfig


def _cfg(total_gb: float, available_gb: float, cpu: int) -> VMConfig:
    return VMConfig(
        total_ram_gb=total_gb,
        available_ram_gb=available_gb,
        cpu_count=cpu,
        swap_total_gb=4.0,
        swap_free_gb=4.0,
        work_mem="256MB",
        maintenance_work_mem="1GB",
    )


def test_postgres_buffers_scale_with_ram_and_clamp() -> None:
    small = _cfg(8.0, 6.0, 2)
    assert small.postgres_shared_buffers_gb() == 2
    assert small.postgres_effective_cache_gb() == 4
    mid = _cfg(47.0, 42.0, 4)
    assert mid.postgres_shared_buffers_gb() == 11
    assert mid.postgres_effective_cache_gb() == 23
    huge = _cfg(376.0, 350.0, 12)
    assert huge.postgres_shared_buffers_gb() == 32  # clamp
    assert huge.postgres_effective_cache_gb() == 188


def test_workers_for_uses_live_cpu_and_ram() -> None:
    cfg = _cfg(47.0, 42.0, 4)
    assert cfg.workers_for(mem_per_worker_gb=7.0) == 4
    assert cfg.workers_for(mem_per_worker_gb=0.5) == 4
    tiny = _cfg(8.0, 3.0, 2)
    assert tiny.workers_for(mem_per_worker_gb=7.0) == 1
