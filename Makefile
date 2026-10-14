SHELL        := /bin/bash
VENV         := /opt/vtech/venv/bin/activate
MLFLOW_DB    := sqlite:////opt/vtech/data/mlflow.db
MLFLOW_PORT  := 5000
MLFLOW_PID   := /tmp/mlflow-ui.pid
MLFLOW_LOG   := /opt/vtech/logs/mlflow-ui.log

# ── Training monitor ──────────────────────────────────────────────────────────
.DEFAULT_GOAL := public-help
PYTHON ?= python
DEMO_OUTPUT ?= .demo-output/run
PUBLICATION_TESTS = tests/test_public_demo.py
PUBLIC_CONTRACT_TESTS = tests/architecture/test_causal.py \
	tests/architecture/test_economic_evaluation.py \
	tests/architecture/test_experiment_manifest.py \
	tests/architecture/test_schemas.py \
	tests/architecture/test_lifecycle.py

.PHONY: public-help demo test-publication
public-help:
	@printf 'Public showcase: make demo | make test-publication\n'
	@printf 'Activate your own venv first. Demo output: $(DEMO_OUTPUT) (must not exist).\n'
	@printf 'Other targets are historical operator recipes, not public setup commands.\n'

demo:
	$(PYTHON) -m src.demo --output "$(DEMO_OUTPUT)"

test-publication:
	$(PYTHON) -m pytest $(PUBLICATION_TESTS) $(PUBLIC_CONTRACT_TESTS) tests/test_scope_gate_consistency.py

.PHONY: ml-monitor hpo-monitor gpu-monitor gpu-follow gpu-watch

ml-monitor:
	@bash scripts/ops/monitor_training.sh

hpo-monitor:
	@watch -n 10 /opt/vtech/venv/bin/python3 /opt/vtech/scripts/search/hpo_monitor.py

# ADR-073: one-shot GPU job heartbeat (SSH progress.json + fold CM + health probe).
gpu-monitor:
	@bash scripts/ops/gpu_monitor.sh

# Live GPU runner journal (needs runner tee on the GPU VM). Ctrl+C to stop.
gpu-follow:
	@set -a && source /opt/vtech/.env && set +a && \
	bash deploy/gpu/ssh_gpu.sh 'journalctl -u vtech-gpu-runner -f --no-pager'

# Refresh gpu-monitor every 15s. Ctrl+C to stop.
gpu-watch:
	@watch -n 15 -- $(MAKE) --no-print-directory gpu-monitor

# ── MLflow UI ──────────────────────────────────────────────────────────────────
.PHONY: mlflow mlflow-stop mlflow-status

mlflow:
	@source $(VENV) && \
	nohup mlflow ui \
		--backend-store-uri $(MLFLOW_DB) \
		--host 127.0.0.1 \
		--port $(MLFLOW_PORT) \
		&> $(MLFLOW_LOG) & \
	echo $$! > $(MLFLOW_PID) && \
	echo "MLflow UI started on http://localhost:$(MLFLOW_PORT)  (pid=$$(cat $(MLFLOW_PID)), log=$(MLFLOW_LOG))"

mlflow-stop:
	@if [ -f $(MLFLOW_PID) ]; then \
		kill $$(cat $(MLFLOW_PID)) 2>/dev/null && echo "MLflow UI stopped" || echo "Process already gone"; \
		rm -f $(MLFLOW_PID); \
	else \
		echo "No PID file found — MLflow UI may not be running"; \
	fi

mlflow-status:
	@if [ -f $(MLFLOW_PID) ] && kill -0 $$(cat $(MLFLOW_PID)) 2>/dev/null; then \
		echo "MLflow UI running (pid=$$(cat $(MLFLOW_PID)))  http://localhost:$(MLFLOW_PORT)"; \
	else \
		echo "MLflow UI not running"; \
	fi

# ── Web console ────────────────────────────────────────────────────────────────
WEBUI_PORT  := 8700
WEBUI_LOG   := /opt/vtech/logs/webui.log
WEBUI_INDEX := src/webui/static/index.html
# Match the console by its unique module invocation, not a PID file: `nohup … &` records
# the nohup wrapper's PID (nohup forks python), so a PID-file kill orphans the real server.
# The leading [p] stops pgrep/pkill from matching this recipe's own shell.
WEBUI_MATCH := [p]ython -m src.webui.app

.PHONY: webui webui-stop webui-status webui-restart webui-bust webui-refresh

# Always start from the host checkout. `make webui` from a Grok worktree used to
# import that worktree's src/ while writing logs to /opt/vtech (Custom Test then
# ran as manual_override into research/search/ — ADR-036 miss).
REPO_ROOT   := /opt/vtech

webui:
	@cd $(REPO_ROOT) && source $(VENV) && \
	set -a && source $(REPO_ROOT)/.env && set +a && \
	PYTHONPATH=$(REPO_ROOT) nohup python -m src.webui.app --host 127.0.0.1 --port $(WEBUI_PORT) &> $(WEBUI_LOG) & \
	echo "VTech console started on http://localhost:$(WEBUI_PORT)  cwd=$(REPO_ROOT) (log=$(WEBUI_LOG); 'make webui-status' for pid)"

webui-stop:
	@pkill -f "$(WEBUI_MATCH)" && echo "Web console stopping…" || echo "Web console not running"
	@for i in $$(seq 1 40); do \
		ss -ltn 2>/dev/null | grep -q ':$(WEBUI_PORT) ' || break; \
		sleep 0.25; \
	done
	@if ss -ltn 2>/dev/null | grep -q ':$(WEBUI_PORT) '; then \
		echo "Port $(WEBUI_PORT) still busy after graceful stop — SIGKILL leftover python"; \
		pkill -9 -f "$(WEBUI_MATCH)" || true; \
		sleep 0.4; \
	fi
	@if ss -ltn 2>/dev/null | grep -q ':$(WEBUI_PORT) '; then \
		echo "ERROR: :$(WEBUI_PORT) still in use"; \
		ss -ltnp | grep '$(WEBUI_PORT)' || true; \
		exit 1; \
	fi

webui-status:
	@if pgrep -f "$(WEBUI_MATCH)" >/dev/null; then \
		echo "Web console running (pid=$$(pgrep -f '$(WEBUI_MATCH)' | tr '\n' ' '))  http://127.0.0.1:$(WEBUI_PORT)"; \
	else \
		echo "Web console not running"; \
	fi

# Restart the Python process only (picks up src/webui/*.py). Static assets may still
# be browser-cached until ?v= changes — use webui-refresh after UI edits.
# Stop waits until :8700 is actually free (lifespan + live-snapshot child can
# outlive the 2s window that used to race here → EADDRINUSE, Makefile still
# printed "started", browser tab hung on the dead tunnel). Start is not
# success until GET / returns 200.
webui-restart: webui-stop
	@$(MAKE) --no-print-directory webui
	@ok=0; \
	for i in $$(seq 1 40); do \
		if curl -sf -m 1 -o /dev/null http://127.0.0.1:$(WEBUI_PORT)/; then ok=1; break; fi; \
		sleep 0.25; \
	done; \
	if [ "$$ok" = 1 ]; then \
		echo "Web console healthy on http://127.0.0.1:$(WEBUI_PORT)"; \
	else \
		echo "ERROR: console launched but GET / did not return 200 within 10s"; \
		tail -n 40 $(WEBUI_LOG) || true; \
		exit 1; \
	fi

# Rewrite app.js / style.css ?v= query params so browsers fetch fresh assets.
webui-bust:
	@test -f $(WEBUI_INDEX) || { echo "missing $(WEBUI_INDEX)"; exit 1; }
	@v=$$(date +%Y%m%d-%H%M%S); \
	sed -i -E "s#(/style\.css\?v=)[^\"']+#\1$$v#; s#(/app\.js\?v=)[^\"']+#\1$$v#" $(WEBUI_INDEX); \
	echo "Static cache bust → ?v=$$v  ($(WEBUI_INDEX))"

# Full hard refresh: bust static ?v=, restart console. Then reload the browser tab
# (Ctrl+Shift+R / Cmd+Shift+R if the HTML shell itself still looks cached).
webui-refresh: webui-bust webui-restart
	@echo ""
	@echo "Console restarted with new asset versions."
	@echo "  → open/reload  http://localhost:$(WEBUI_PORT)"
	@echo "  → if UI still looks old: hard-reload the tab (Ctrl+Shift+R / Cmd+Shift+R)"

# ── SHAP on a serving-bundle folder or a search-run folder ────────────────────
# Writes CSVs/PNGs/SHAP_SUMMARY.md to <run>/SHAP (or <bundle>/SHAP).
# Usage:
#   make shap MODEL=/opt/vtech/data/models/model_registry/nq-50pt-econ/xgb_tbm_topgroups_20260715T234308Z
#   make shap MODEL=/opt/vtech/data/research/search/nq-50pt-econ/20260815T202407Z
#   make shap MODEL=/data/research/search/nq-50pt-econ/20260815T202407Z   # alias → $VTECH_DATA_DIR
#   make shap MODEL=/path/to/bundle SHAP_ARGS='--sample 20000 --tau 0.8 --max-days 60 --candidate xgboost'
# Module is src.evaluation.shap_explain (not src.evluation). Use `make shap` so venv+env load.
.PHONY: shap
shap:
	@test -n "$(MODEL)" || { \
	  echo "usage: make shap MODEL=/path/to/bundle-or-search-run [SHAP_ARGS='--sample 8000 --tau 0.8']"; \
	  exit 1; }
	@source $(VENV) && set -a && source /opt/vtech/.env && set +a && \
	PYTHONPATH=$(CURDIR) python -m src.evaluation.shap_explain --model "$(MODEL)" $(SHAP_ARGS)

# ── Common operator entrypoints (paths after ADR-028 script grouping) ─────────
.PHONY: rd-dry sync-oci promote search-econ-dry

rd-dry:
	@source $(VENV) && set -a && source /opt/vtech/.env && set +a && \
	PYTHONPATH=/opt/vtech python -m src.research_director.run --dry-run

sync-oci:
	@bash scripts/ops/sync_to_oci.sh

promote:
	@source $(VENV) && PYTHONPATH=/opt/vtech python scripts/serving/promote_model.py $(ARGS)

search-econ-dry:
	@source $(VENV) && set -a && source /opt/vtech/.env && set +a && \
	PYTHONPATH=/opt/vtech python -m src.search.controller --objective nq-50pt-econ \
	  --manual-spec config/specs/manual_candidates.nq-50pt-econ.yaml --dry-run --no-ensemble

# ── Scoped test gates ─────────────────────────────────────────────────────────
# The full suite has no green baseline and cannot gate a change, so gate on the
# tests that cover the component being changed.
#
# The path list is explicit on purpose. `pytest tests/ -m acquisition` selects
# the right tests but still *imports* all 147 test modules to collect them, so
# any module-level side effect in an unrelated test runs during a scoped gate —
# which is exactly how a narrow change can cause wide damage. Explicit paths
# never import what they do not run.
#
# tests/test_scope_gate_consistency.py asserts this list equals the set of
# modules carrying @pytest.mark.acquisition, so the two cannot drift.
ACQUISITION_TESTS = \
  tests/test_rotating_dbn_stream.py \
  tests/test_live_options_subscription.py \
  tests/test_live_streamer_eom_options.py \
  tests/test_mbo_resting_book.py \
  tests/test_globex_session_date.py \
  tests/test_logging_extras.py \
  tests/test_definition_gap_detection.py \
  tests/test_definition_acquisition.py \
  tests/test_weekly_session_lifecycle.py \
  tests/test_gap_recovery_acquisition.py \
  tests/test_gap_recovery_ingestion.py \
  tests/test_gap_recovery_validation.py \
  tests/test_gap_rebuild_outputs.py \
  tests/test_scope_gate_consistency.py

.PHONY: test-acquisition test-markers test-scope

# Databento raw-data acquisition: subscriptions, live streamer, DBN archive.
test-acquisition:
	@source $(VENV) && set -a && source /opt/vtech/.env && set +a && \
	PYTHONPATH=$(CURDIR) python -m pytest $(ACQUISITION_TESTS) --strict-markers

# Fails if the path list above and the marker set have drifted apart.
test-scope:
	@source $(VENV) && set -a && source /opt/vtech/.env && set +a && \
	PYTHONPATH=$(CURDIR) python -m pytest tests/test_scope_gate_consistency.py -q

# What each marker currently selects (wide collection; diagnostic only).
test-markers:
	@source $(VENV) && set -a && source /opt/vtech/.env && set +a && \
	for m in acquisition db_write real_data; do \
	  n=$$(PYTHONPATH=$(CURDIR) python -m pytest tests/ -m $$m --collect-only 2>/dev/null \
	       | sed -n 's|^\([0-9]*\)/[0-9]* tests collected.*|\1|p'); \
	  printf "  %-14s %s tests\n" "$$m" "$${n:-0}"; \
	done

# ── Git snapshot push ───────────────────────────────────────────────────────────
# Stages everything, force-moves <branch> to a new commit on top of its current
# tip (parent = <branch>'s current head), pushes it, then unstages (git reset).
# Usage: make push_git <branch>     e.g. make push_git Canonical_redesign/09-22-26
.PHONY: push_git
push_git:
	@branch="$(filter-out $@,$(MAKECMDGOALS))"; \
	test -n "$$branch" || { echo "usage: make push_git <branch>"; exit 1; }; \
	git add -A && \
	git branch -f "$$branch" "$$(git commit-tree "$$(git write-tree)" -p "$$branch" -m "Canonical redesign: updated snapshot")" && \
	git push origin "$$branch" && \
	git reset

# Swallow extra command-line args (e.g. the branch name above) so make doesn't
# try to interpret them as targets of their own.
%:
	@:
