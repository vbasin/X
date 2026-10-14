"""GPU execution plane — ADR-022 (OCI Queue + content-addressed code bundles).

The CPU VM (orchestrator) publishes a content-addressed code bundle to Object Storage and enqueues
model-candidate jobs on the managed **OCI Queue** (jobs + results channels); the GPU VM's runner
long-polls, fetches the bundle by digest, runs each job in a subprocess, and publishes results
back on the queue. Lifecycle is autonomous: the CPU only ``ensure_running``s the VM (START iff
STOPPED); the GPU VM powers itself off when idle. Neither VM opens an inbound socket to the other.
"""
