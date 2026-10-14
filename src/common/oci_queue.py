"""OCI Queue-backed message queue for the GPU execution plane — ADR-022.

Replaces the hand-rolled Object-Storage marker FSM (``src/gpu_runner/queue.py``) with the
managed **OCI Queue** service. The managed queue provides, out of the box, what the marker
FSM implemented by hand:

* **visibility timeout** (≤ 12 h, extendable) — a consumed message is hidden from other
  consumers and auto-**redelivered** if the worker dies without deleting it (replaces the
  ``claimed/`` + ``heartbeat/`` bookkeeping and stale detection);
* **long polling** — one blocking ``get`` up to 30 s (replaces busy ``LIST`` polling);
* **dead-letter queue** — after N delivery attempts (replaces manual retry logic);
* **channels** — a lightweight sub-destination string per message, used here for
  ``jobs.<objective_id>`` and ``results.<objective_id>`` (replaces per-objective prefixes).

Large blobs (OOF parquet, diagnostics, code bundles) still travel via Object Storage; a queue
message only carries small JSON (job spec pointers, or a result summary + object pointers).

Backends (auto-selected by :func:`make_message_queue`):
  * ``local`` — a filesystem-backed queue (env ``VTECH_GPU_QUEUE_LOCAL`` or ``local_dir=``),
    for unit tests / dev without cloud. Implements visibility-timeout redelivery so the
    consume/ack/redeliver contract is testable offline.
  * ``oci``   — OCI Queue via the ``oci`` SDK (``OCI_QUEUE_ID`` + ``OCI_QUEUE_MESSAGES_ENDPOINT``).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_CHANNEL = "default"


@dataclass
class QueueMessage:
    """One delivered message. ``receipt`` is the token required to delete/extend it."""
    body: dict[str, Any]
    id: str
    receipt: str
    channel: str = _DEFAULT_CHANNEL


class MessageQueue:
    """Minimal message-queue interface: put / get / delete / extend."""

    backend_kind = "abstract"

    def put(self, body: dict[str, Any], *, channel: str = _DEFAULT_CHANNEL) -> None: ...

    def get(self, *, channel: str = _DEFAULT_CHANNEL, visibility_s: int = 300,
            max_msgs: int = 1, wait_s: int = 0) -> list[QueueMessage]: ...

    def delete(self, msg: QueueMessage) -> None: ...

    def extend(self, msg: QueueMessage, visibility_s: int) -> None: ...


# ── local backend (tests / dev) ──────────────────────────────────────────────

class LocalMessageQueue(MessageQueue):
    """Filesystem-backed queue with visibility-timeout redelivery, rooted at a directory.

    Layout: ``<root>/<channel>/<msg_id>.json`` (payload) + ``<msg_id>.lock`` (invisible-until
    epoch). A message is *visible* when it has no lock or its lock has expired. ``get`` claims
    up to ``max_msgs`` visible messages by writing a lock at ``now + visibility_s``; ``delete``
    removes both files; ``extend`` rewrites the lock expiry. With ``visibility_s=0`` a consumed
    message is immediately visible again (deterministic redelivery for tests).
    """

    backend_kind = "local"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _cdir(self, channel: str) -> Path:
        d = self._root / channel
        d.mkdir(parents=True, exist_ok=True)
        return d

    def put(self, body: dict[str, Any], *, channel: str = _DEFAULT_CHANNEL) -> None:
        d = self._cdir(channel)
        mid = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        tmp = d / f"{mid}.json.tmp"
        tmp.write_text(json.dumps({"body": body}))
        os.replace(tmp, d / f"{mid}.json")

    def get(self, *, channel: str = _DEFAULT_CHANNEL, visibility_s: int = 300,
            max_msgs: int = 1, wait_s: int = 0) -> list[QueueMessage]:
        d = self._cdir(channel)
        now = time.time()
        out: list[QueueMessage] = []
        for p in sorted(d.glob("*.json")):
            lock = p.with_suffix(".lock")
            if lock.exists():
                try:
                    if float(lock.read_text()) > now:
                        continue
                except ValueError:
                    pass
            lock.write_text(str(now + visibility_s))
            body = json.loads(p.read_text())["body"]
            out.append(QueueMessage(body=body, id=p.stem, receipt=p.stem, channel=channel))
            if len(out) >= max_msgs:
                break
        return out

    def delete(self, msg: QueueMessage) -> None:
        d = self._cdir(msg.channel)
        p = d / f"{msg.id}.json"
        lock = p.with_suffix(".lock")
        if p.exists():
            p.unlink()
        if lock.exists():
            lock.unlink()

    def extend(self, msg: QueueMessage, visibility_s: int) -> None:
        d = self._cdir(msg.channel)
        (d / f"{msg.id}.lock").write_text(str(time.time() + visibility_s))


# ── OCI Queue backend ────────────────────────────────────────────────────────

class OCIMessageQueue(MessageQueue):
    """OCI Queue via the ``oci`` SDK (lazy import; only needed on the real backend)."""

    backend_kind = "oci"

    def __init__(self, queue_id: str, *, endpoint: str, config_file: str | None = None) -> None:
        self._qid = queue_id
        self._endpoint = endpoint
        self._config_file = config_file or os.environ.get(
            "OCI_CONFIG_FILE", os.path.expanduser("~/.oci/config"))
        self._client: Any = None

    def _c(self) -> Any:
        if self._client is None:
            import oci  # noqa: PLC0415 (lazy: not needed for the local backend)
            config_path = self._config_file
            if config_path and os.path.exists(config_path):
                cfg = oci.config.from_file(config_path)
                self._client = oci.queue.QueueClient(cfg, service_endpoint=self._endpoint)
            else:
                # No key-file config — fall back to instance principal auth (OCI cloud VMs).
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                self._client = oci.queue.QueueClient(
                    config={}, signer=signer, service_endpoint=self._endpoint)
        return self._client

    def put(self, body: dict[str, Any], *, channel: str = _DEFAULT_CHANNEL) -> None:
        import oci  # noqa: PLC0415
        # The channel is carried in per-message metadata (channel_id), not a top-level field.
        entry = oci.queue.models.PutMessagesDetailsEntry(
            content=json.dumps(body, default=str),
            metadata=oci.queue.models.MessageMetadata(channel_id=channel))
        self._c().put_messages(
            self._qid, oci.queue.models.PutMessagesDetails(messages=[entry]))

    def get(self, *, channel: str = _DEFAULT_CHANNEL, visibility_s: int = 300,
            max_msgs: int = 1, wait_s: int = 0) -> list[QueueMessage]:
        resp = self._c().get_messages(
            self._qid, visibility_in_seconds=visibility_s,
            timeout_in_seconds=min(int(wait_s), 30),   # OCI long-poll cap is 30s
            limit=min(int(max_msgs), 20),              # OCI GetMessages limit cap is 20
            channel_filter=channel)
        msgs = getattr(resp.data, "messages", None) or []
        return [QueueMessage(body=json.loads(m.content), id=str(getattr(m, "id", "")),
                             receipt=m.receipt, channel=channel) for m in msgs]

    def delete(self, msg: QueueMessage) -> None:
        self._c().delete_message(self._qid, msg.receipt)

    def extend(self, msg: QueueMessage, visibility_s: int) -> None:
        import oci  # noqa: PLC0415
        self._c().update_message(
            self._qid, msg.receipt,
            oci.queue.models.UpdateMessageDetails(visibility_in_seconds=visibility_s))


def make_message_queue(
    *,
    local_dir: str | None = None,
    local_env: str | None = "VTECH_GPU_QUEUE_LOCAL",
    queue_id: str | None = None,
    queue_id_env: str = "OCI_QUEUE_ID",
    endpoint: str | None = None,
    endpoint_env: str = "OCI_QUEUE_MESSAGES_ENDPOINT",
    config_file: str | None = None,
) -> tuple[MessageQueue, str]:
    """Return ``(queue, kind)`` where kind is ``"local"`` or ``"oci"``.

    Selects the local backend when ``local_dir`` is given or ``local_env`` names a set env var;
    otherwise the OCI backend, which requires both the queue OCID (``queue_id``/``OCI_QUEUE_ID``)
    and its messages endpoint (``endpoint``/``OCI_QUEUE_MESSAGES_ENDPOINT``).
    """
    ld = local_dir or (os.environ.get(local_env) if local_env else None)
    if ld:
        return LocalMessageQueue(ld), "local"
    qid = queue_id or os.environ.get(queue_id_env)
    ep = endpoint or os.environ.get(endpoint_env)
    if not qid or not ep:
        raise RuntimeError(
            "OCI Queue not configured: set OCI_QUEUE_ID + OCI_QUEUE_MESSAGES_ENDPOINT "
            "(or VTECH_GPU_QUEUE_LOCAL for the local backend)")
    return OCIMessageQueue(qid, endpoint=ep, config_file=config_file), "oci"
