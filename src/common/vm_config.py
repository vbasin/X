"""Runtime VM configuration: hardware detection + derived run parameters.

A single call to ``get_vm_config()`` reads ``/proc/meminfo`` and
``os.cpu_count()`` once, then returns an immutable ``VMConfig`` dataclass.
The result is cached per process via ``lru_cache``; call
``get_vm_config.cache_clear()`` if you need a fresh reading mid-process.

Usage — scripts / CLI::

    from src.common.vm_config import get_vm_config
    cfg = get_vm_config()
    print(cfg)           # human-readable summary at startup

    # argparse default that auto-scales to the current machine:
    parser.add_argument("--workers", type=int, default=cfg.workers_for())

    # Task-specific memory budget (download workers are lighter than feature-engine):
    dl_workers  = cfg.workers_for(mem_per_worker_gb=0.5)
    fe_workers  = cfg.workers_for(mem_per_worker_gb=7.0)

Usage — ingestion / DB::

    cfg = get_vm_config()
    cur.execute(f"SET work_mem = '{cfg.work_mem}'")
    cur.execute(f"SET maintenance_work_mem = '{cfg.maintenance_work_mem}'")
    chunks = store.to_df(count=cfg.ingest_chunk_size())

Usage — storage check::

    from src.common.vm_config import get_vm_config, fmt_bytes
    cfg = get_vm_config()
    st  = cfg.storage_for("/opt/vtech/data")
    print(st.status_line(need_bytes=5 * 1024**3, label="download"))
    # ✅ /opt/vtech/data: 412.30 GB free / 500.00 GB total — 407.30 GB spare after download

Profiles (``mem_per_worker_gb`` guidance):
  7.0  — feature-engine patch scripts  (dc_, iv_, gex_, ht_)
  7.0  — options ingest workers        (QNE, NQ opt)
  0.5  — download workers              (network-bound, low RAM)
  2.0  — light processing              (monitor, verify, patch_close)
"""
from __future__ import annotations

import functools
import os
import shutil
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_proc_meminfo() -> tuple[int, int, int, int]:
    """Return (MemTotal_kb, MemAvailable_kb, SwapTotal_kb, SwapFree_kb) from /proc/meminfo.

    Falls back to (8 GB, 4 GB, 0, 0) on non-Linux or read error.
    Swap values are informational only — worker/chunk sizing uses RAM exclusively
    because swap throughput (disk-backed) is orders of magnitude slower than RAM.
    """
    total_kb = available_kb = swap_total_kb = swap_free_kb = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                elif line.startswith("SwapTotal:"):
                    swap_total_kb = int(line.split()[1])
                elif line.startswith("SwapFree:"):
                    swap_free_kb = int(line.split()[1])
                if all(v is not None for v in
                       (total_kb, available_kb, swap_total_kb, swap_free_kb)):
                    break
    except OSError:
        pass

    if total_kb is None:
        total_kb = 8 * 1024 * 1024      # 8 GB fallback
    if available_kb is None:
        available_kb = total_kb // 2
    if swap_total_kb is None:
        swap_total_kb = 0
    if swap_free_kb is None:
        swap_free_kb = 0

    return total_kb, available_kb, swap_total_kb, swap_free_kb


def _fmt_bytes(b: int) -> str:
    """Format byte count as a PostgreSQL memory string (e.g. '6GB', '512MB')."""
    gib = b >> 30
    return f"{gib}GB" if gib >= 1 else f"{b >> 20}MB"


def fmt_bytes(b: int | float) -> str:
    """Human-readable byte count for display (e.g. '2.45 GB', '345.22 MB', '1.23 TB')."""
    if b >= 1e12:
        return f"{b / 1e12:.2f} TB"
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    return f"{b / 1e6:.2f} MB"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageInfo:
    """Point-in-time disk usage for a single filesystem.

    Obtained via ``VMConfig.storage_for(path)``; not cached because
    disk usage changes frequently.
    """

    path: str
    free_bytes: int
    total_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    @property
    def free_gb(self) -> float:
        return self.free_bytes / 1e9

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1e9

    @property
    def used_pct(self) -> float:
        return (self.used_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0.0

    def fits(self, need_bytes: int | float, safety_pct: float = 0.0) -> bool:
        """True when ``need_bytes`` (plus optional safety margin) fits in free space."""
        return self.free_bytes >= need_bytes * (1.0 + safety_pct)

    def headroom_after(self, need_bytes: int | float) -> int:
        """Bytes remaining free after reserving ``need_bytes``."""
        return int(self.free_bytes - need_bytes)

    def status_line(
        self,
        need_bytes: int | float,
        label: str = "operation",
        warn_threshold_gb: float = 10.0,
    ) -> str:
        """One-line ✅ / ⚠️ / ❌ summary for a specific space requirement.

        Parameters
        ----------
        need_bytes:
            Bytes required by the planned operation.
        label:
            Short description used in the message (e.g. 'download', 'ingest').
        warn_threshold_gb:
            Headroom below which a ⚠️  warning is raised (default 10 GB).
        """
        headroom = self.headroom_after(need_bytes)
        base = f"{self.path}: {fmt_bytes(self.free_bytes)} free / {fmt_bytes(self.total_bytes)} total"
        if headroom < 0:
            return f"❌  {base} — need {fmt_bytes(need_bytes)}, only {fmt_bytes(self.free_bytes)} free"
        if headroom < warn_threshold_gb * 1e9:
            return f"⚠️   {base} — only {fmt_bytes(headroom)} spare after {label}"
        return f"✅  {base} — {fmt_bytes(headroom)} spare after {label}"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VMConfig:
    """Immutable snapshot of VM hardware + derived run parameters.

    Attributes
    ----------
    total_ram_gb:
        Physical RAM installed (GB).
    available_ram_gb:
        RAM currently available to new processes (GB).  This is what matters
        for sizing workers and chunk buffers.
    cpu_count:
        Logical CPU cores reported by the OS (``os.cpu_count()``).
    work_mem:
        PostgreSQL ``work_mem`` string: 2 % of available RAM,
        clamped to 256 MB – 8 GB.
    maintenance_work_mem:
        PostgreSQL ``maintenance_work_mem`` string: 50 % of available RAM,
        clamped to 1 GB – 128 GB.  Used for ``CREATE INDEX`` sort buffers.
    """

    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    swap_total_gb: float
    swap_free_gb: float
    work_mem: str
    maintenance_work_mem: str

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def storage_for(self, path: str) -> StorageInfo:
        """Return current disk usage for the filesystem containing ``path``.

        Not cached — reads live from the OS each call.
        Raises ``OSError`` if the path does not exist.
        """
        try:
            st = shutil.disk_usage(path)
        except (FileNotFoundError, OSError) as exc:
            raise OSError(f"Cannot stat disk at {path!r}: {exc}") from exc
        return StorageInfo(path=path, free_bytes=st.free, total_bytes=st.total)

    # ------------------------------------------------------------------
    # Parallelism
    # ------------------------------------------------------------------

    def postgres_shared_buffers_gb(self) -> int:
        """shared_buffers: ~25% of physical RAM, clamped 1–32 GB."""
        return max(1, min(32, int(self.total_ram_gb * 0.25)))

    def postgres_effective_cache_gb(self) -> int:
        """effective_cache_size: ~50% of physical RAM, clamped 2–256 GB."""
        return max(2, min(256, int(self.total_ram_gb * 0.50)))

    def postgres_maintenance_work_mem_gb(self) -> int:
        """maintenance_work_mem: ~8% of physical RAM, clamped 1–8 GB."""
        return max(1, min(8, int(self.total_ram_gb * 0.08)))

    def workers_for(
        self,
        mem_per_worker_gb: float = 7.0,
        cpu_cap: int | None = None,
    ) -> int:
        """Recommended parallel worker count for a given per-worker memory budget.

        Parameters
        ----------
        mem_per_worker_gb:
            Estimated peak RAM consumed by a single worker process.
            Default (7.0 GB) matches the feature-engine patch scripts.
            See module docstring for task-specific guidance.
        cpu_cap:
            Optional hard cap on CPU-based parallelism.  Defaults to
            ``self.cpu_count``.

        Returns
        -------
        int
            ``min(cpu_limit, floor(available_ram_gb / mem_per_worker_gb))``,
            always at least 1.
        """
        cpu_limit = cpu_cap if cpu_cap is not None else self.cpu_count
        mem_limit = max(1, int(self.available_ram_gb / mem_per_worker_gb))
        return min(cpu_limit, mem_limit)

    # ------------------------------------------------------------------
    # Ingestion sizing
    # ------------------------------------------------------------------

    def ingest_chunk_size(
        self,
        bytes_per_row: int = 100,
        target_pct: float = 0.01,
    ) -> int:
        """Rows per DataFrame chunk for ``DBNStore.to_df(count=...)``.

        Targets ``target_pct`` of available RAM per in-memory chunk,
        clamped to [1 M, 20 M] rows and rounded to the nearest million.

        Parameters
        ----------
        bytes_per_row:
            Estimated in-memory size of one decoded row.
            ~100 B for TBBO, ~80 B for BBO-1s, ~50 B for definition rows.
        target_pct:
            Fraction of available RAM to target per chunk (default 1 %).
            Raise to 2–3 % on memory-rich machines for fewer commits.
        """
        available_bytes = self.available_ram_gb * (1 << 30)
        raw = int((available_bytes * target_pct) / bytes_per_row)
        clamped = max(1_000_000, min(raw, 20_000_000))
        return round(clamped / 1_000_000) * 1_000_000

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        fe_workers = self.workers_for(mem_per_worker_gb=7.0)
        dl_workers = self.workers_for(mem_per_worker_gb=0.5)
        swap_str = f", swap {self.swap_free_gb:.0f}/{self.swap_total_gb:.0f} GB free" if self.swap_total_gb > 0 else ""
        return (
            f"VMConfig("
            f"RAM {self.available_ram_gb:.0f}/{self.total_ram_gb:.0f} GB avail"
            f"{swap_str}, "
            f"CPU {self.cpu_count} cores | "
            f"FE workers {fe_workers}, DL workers {dl_workers} | "
            f"work_mem {self.work_mem}, maint_mem {self.maintenance_work_mem})"
        )

    def summary_lines(self) -> list[str]:
        """Multi-line startup banner for script output."""
        swap_line = (
            f"  Swap : {self.swap_free_gb:.0f} GB free / {self.swap_total_gb:.0f} GB total"
            " (not used for sizing — disk-backed)"
        ) if self.swap_total_gb > 0 else "  Swap : none configured"
        return [
            f"  RAM  : {self.available_ram_gb:.0f} GB available / {self.total_ram_gb:.0f} GB total",
            swap_line,
            f"  CPU  : {self.cpu_count} logical cores",
            f"  PG   : work_mem={self.work_mem}  maintenance_work_mem={self.maintenance_work_mem}",
            f"  Parallelism: FE/ingest={self.workers_for(7.0)}  download={self.workers_for(0.5)}",
            f"  Chunk: {self.ingest_chunk_size():,} rows (TBBO @ 100 B/row, 1% RAM)",
        ]


# ---------------------------------------------------------------------------
# Factory (cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_vm_config() -> VMConfig:
    """Detect VM hardware and return optimal run parameters.

    Reads ``/proc/meminfo`` and ``os.cpu_count()`` exactly once per process
    (result is cached by ``lru_cache``).  To force a fresh reading:

        get_vm_config.cache_clear()
        cfg = get_vm_config()
    """
    total_kb, available_kb, swap_total_kb, swap_free_kb = _read_proc_meminfo()

    total_gb     = total_kb     / (1024 ** 2)
    available_gb = available_kb / (1024 ** 2)
    swap_total_gb = swap_total_kb / (1024 ** 2)
    swap_free_gb  = swap_free_kb  / (1024 ** 2)

    available_bytes = available_kb * 1024
    cpu_count       = os.cpu_count() or 4

    work_mem = max(256 << 20, min(int(available_bytes * 0.02), 8 << 30))   # 256 MB–8 GB
    maint    = max(  1 << 30, min(int(available_bytes * 0.50), 128 << 30)) # 1 GB–128 GB

    return VMConfig(
        total_ram_gb=total_gb,
        available_ram_gb=available_gb,
        cpu_count=cpu_count,
        swap_total_gb=swap_total_gb,
        swap_free_gb=swap_free_gb,
        work_mem=_fmt_bytes(work_mem),
        maintenance_work_mem=_fmt_bytes(maint),
    )
