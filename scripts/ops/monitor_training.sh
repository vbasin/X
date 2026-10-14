#!/usr/bin/env bash
# Live monitor for walk-forward training progress.
# Usage:  bash scripts/monitor_training.sh
# Press Ctrl+C to exit.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFRESH=5

# ── helpers ──────────────────────────────────────────────────────────────────

bar() {
    local pct=$1 s="" i
    local filled=$(( pct * 20 / 100 ))
    for (( i=0; i<filled; i++ ));  do s+="█"; done
    for (( i=filled; i<20; i++ )); do s+="░"; done
    printf '%s' "$s"
}

fmt_sec() {
    local s=${1:-0}
    local h=$(( s/3600 )) m=$(( (s%3600)/60 ))
    (( h > 0 )) && printf '%dh %02dm' $h $m && return
    (( m > 0 )) && printf '%dm %02ds' $m $((s%60)) && return
    printf '%ds' $s
}

# Box helpers — 60-char wide box
ROW()    { printf "║  %-54s  ║\n" "$1"; }
SEP()    { echo   "╠══════════════════════════════════════════════════════════╣"; }
# BARROW: ║  <label8>[bar20] <right25>║  = 60 chars total
BARROW() { printf "║  %-8s[%s] %-25s║\n" "$1" "$(bar "${2:-0}")" "$3"; }

get_ps_arg() {
    # handles both --arg value and --arg=value styles
    ps -p "$1" -o args= 2>/dev/null \
        | grep -oP "(?<=--${2} )\S+|(?<=--${2}=)\S+" | head -1
}

fold_test_label() {
    local pid="${1:-}" fd="${2:-}"
    if [[ -n "$pid" ]]; then
        local ts
        ts=$(get_ps_arg "$pid" "test-start" 2>/dev/null || true)
        if [[ -n "$ts" ]]; then
            python3 -c "
from datetime import datetime
print(datetime.strptime('${ts:0:7}','%Y-%m').strftime('%b %Y'))" 2>/dev/null && return
        fi
    fi
    if [[ -n "$fd" ]] && [[ -f "${fd}/training_result.json" ]]; then
        python3 - "${fd}/training_result.json" 2>/dev/null <<'PY'
import json, sys
from datetime import datetime
try:
    d = json.load(open(sys.argv[1]))
    ts = d.get("test_start","")
    if ts: print(datetime.strptime(ts[:7],'%Y-%m').strftime('%b %Y'))
except Exception: pass
PY
    fi
}

metrics_lines() {
    local result="${1}/training_result.json"
    [[ -f "$result" ]] || return 1
    python3 - "$result" 2>/dev/null <<'PY'
import json, sys, math
d = json.load(open(sys.argv[1]))
f   = d.get("test_f1_per_class", {})
bd  = f.get("BIG_DOWN", 0); bu = f.get("BIG_UP", 0)
tau = {(e["direction"], round(e["tau"], 2)): e for e in d.get("precision_at_tau", [])}
pf  = lambda v: f"{v*100:.0f}%" if isinstance(v,(int,float)) and not math.isnan(v) else "n/a"
sec = int(d.get("training_time_seconds", 0))
print(f"F1_macro={d.get('test_f1_macro',0):.4f}  BD={bd:.4f}  BU={bu:.4f}  {sec//3600}h{(sec%3600)//60:02d}m")
# P@tau rows for BU and BD across multiple thresholds
for direction, label in [("BIG_UP","BU"),("BIG_DOWN","BD")]:
    parts = []
    for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        e = tau.get((direction, t), {})
        n = e.get("n_signals", 0)
        if n > 0:
            parts.append(f"{int(t*100)}={pf(e.get('precision',float('nan')))}(n={n})")
    if parts:
        print(f"P@τ {label}: {' '.join(parts)}")
PY
}

live_epoch_metrics() {
    # Extracts latest per-epoch stats from the Keras log (for ACTIVE folds).
    # Outputs up to 2 lines: F1 line + loss line.
    # Handles both new format (f1_bd/f1_bu in log) and legacy (Keras summary only).
    local logfile="$1"
    [[ -f "$logfile" ]] || return 1
    python3 - "$logfile" 2>/dev/null <<'PY'
import sys, re
try:
    with open(sys.argv[1], 'rb') as fh:
        fh.seek(0, 2); fh.seek(max(0, fh.tell() - 131072))
        data = fh.read().decode('utf-8', errors='replace')
except Exception:
    sys.exit()
data = re.sub(r'\x1b\[[0-9;]*[mABCDHJKST]', '', data)
lines = re.split(r'[\r\n]', data)
df1_re  = re.compile(r'directional_f1\s+epoch=(\d+)\s+f1_bd=([\d.eE+-]+)\s+f1_bu=([\d.eE+-]+)\s+val_directional_f1=([\d.eE+-]+)')
keras_re = re.compile(r'^\s*\d+/\d+.*?\bloss:\s*([\d.eE+-]+).*?\bval_accuracy:\s*([\d.eE+-]+).*?\bval_loss:\s*([\d.eE+-]+).*?val_directional_f1:\s*([\d.eE+-]+)')
ig_re    = re.compile(r'ig_epoch\s+epoch=(\d+)\s+BU_top=(\S+)\s+BD_top=(\S+)')
tot_re   = re.compile(r'^\s*Epoch\s+(\d+)/(\d+)')
latest_df1 = None; latest_k = None; latest_ig = None; ep_total = 30; ep_count = 0
for l in lines:
    m = tot_re.search(l)
    if m: ep_total = int(m.group(2))
    m = df1_re.search(l)
    if m: latest_df1 = (int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
    m = keras_re.search(l)
    if m:
        ep_count += 1
        latest_k = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
    m = ig_re.search(l)
    if m: latest_ig = (m.group(2), m.group(3))
if latest_df1:
    ep, bd, bu, df1 = latest_df1
    print(f"Ep {ep}/{ep_total}: BD_F1={bd:.4f}  BU_F1={bu:.4f}  dir_F1={df1:.4f}")
    if latest_k:
        loss, vacc, vloss, vdf1 = latest_k
        print(f"loss={loss:.4f}  val_loss={vloss:.4f}  val_acc={vacc*100:.1f}%")
elif latest_k:
    loss, vacc, vloss, vdf1 = latest_k
    print(f"Ep {ep_count}/{ep_total}: dir_F1={vdf1:.4f}  val_loss={vloss:.4f}")
    print(f"loss={loss:.4f}  val_acc={vacc*100:.1f}%")
if latest_ig:
    def ig_pairs(s):
        out = []
        for p in s.split(','):
            if ':' in p:
                name, val = p.split(':', 1)
                try:
                    out.append((name, float(val)))
                except ValueError:
                    out.append((name, 0.0))
        return out[:5]
    def fmt_ig(pair):
        if not pair:
            return ''
        name, val = pair
        return f"{name:<18} {val:.4f}"[:26]
    bu_pairs = ig_pairs(latest_ig[0])
    bd_pairs = ig_pairs(latest_ig[1])
    print(f"{'IG_BU':<26}  {'IG_BD':<26}")
    for i in range(max(len(bu_pairs), len(bd_pairs))):
        bu = fmt_ig(bu_pairs[i] if i < len(bu_pairs) else None)
        bd = fmt_ig(bd_pairs[i] if i < len(bd_pairs) else None)
        print(f"{bu:<26}  {bd:<26}")
PY
}

ig_lines() {
    # Extracts top-5 IG feature names for BU and BD from training_result.json.
    # Displayed as a 2-column table (IG_BU | IG_BD) with full feature names.
    local result="${1}/training_result.json"
    [[ -f "$result" ]] || return 1
    python3 - "$result" 2>/dev/null <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ig = d.get("integrated_gradients", {})
    def _ig_entry(feat):
        name = feat.get('feature', '')
        val  = feat.get('mean_abs_ig', 0.0)
        return f"{name:<19} {val:.4f}"[:26]
    bu_list = ig.get("BIG_UP",   {}).get("top_features", [])[:5]
    bd_list = ig.get("BIG_DOWN", {}).get("top_features", [])[:5]
    if not bu_list and not bd_list:
        sys.exit()
    print(f"{'IG_BU':<26}  {'IG_BD':<26}")
    for i in range(max(len(bu_list), len(bd_list))):
        bu = _ig_entry(bu_list[i]) if i < len(bu_list) else ''
        bd = _ig_entry(bd_list[i]) if i < len(bd_list) else ''
        print(f"{bu:<26}  {bd:<26}")
except Exception:
    pass
PY
}

# Returns: ep_done ep_total st_total cur_step step_ms avg_ep_sec
# Parses Keras verbose=1 output redirected to file (contains \r + backspaces).
# Each step fragment looks like: "N/M ━━━… ETA Xs/step - accuracy: … [- val_…]"
fold_log_stats() {
    local logfile="$1"
    [[ -f "$logfile" ]] || { echo "0 30 0 0 0 0"; return; }
    python3 - "$logfile" 2>/dev/null <<'PY'
import sys, re
ansi    = re.compile(rb'\x1b\[[0-9;]*[mABCDHJKST]|\x08+')
epoch_re = re.compile(r'Epoch\s+(\d+)/(\d+)')
# "28/28 ━━━ 317s 11s/step - …"  or  "3/28 ━━━ 4:46 11s/step - …"
step_re  = re.compile(r'^(\d+)/(\d+).*?(\d+(?:\.\d+)?)(ms|s)/step')
val_re   = re.compile(r'val_(?:loss|directional_f1|accuracy)')

try:
    data = open(sys.argv[1], 'rb').read()
except Exception:
    print("0 30 0 0 0 0"); sys.exit()

ep_total = 30; ep_done = 0; st_total = 1; cur_step = 0; step_ms = 0.0
epoch_times = []
# Track the step-total established by the FIRST step line after each epoch
# header. Lines from a concurrent process with a different total are rejected.
epoch_established_total = None   # None = not yet locked for current epoch

for raw in re.split(rb'[\r\n]', data):
    try:
        l = ansi.sub(b'', raw).decode('utf-8', errors='replace').strip()
    except Exception:
        continue
    if not l:
        continue

    em = epoch_re.search(l)
    if em:
        ep_total = int(em.group(2))
        epoch_established_total = None   # reset; first step will re-lock
        continue

    sm = step_re.match(l)
    if not sm:
        continue

    step  = int(sm.group(1)); total = int(sm.group(2))
    ms    = float(sm.group(3)) * (1000 if sm.group(4) == 's' else 1)

    # Lock the expected step-total for this epoch on the first line seen.
    # Reject any line whose total doesn't match — it's from a different process.
    if epoch_established_total is None:
        epoch_established_total = total
    elif total != epoch_established_total:
        continue   # interleaved output from a concurrent process — ignore

    if step == total and val_re.search(l):
        # Epoch complete — record epoch duration, reset step cursor
        ep_done += 1
        if total > 0 and ms > 0:
            epoch_times.append(total * ms / 1000)
        cur_step = 0; st_total = total
        epoch_established_total = None   # unlock for next epoch
    elif ms > 0:
        # In-progress step
        cur_step = step; st_total = total; step_ms = ms

avg = int(sum(epoch_times) / len(epoch_times)) if epoch_times \
      else (int(st_total * step_ms / 1000) if step_ms else 0)
print(ep_done, ep_total, st_total, cur_step, int(step_ms), avg)
PY
}

post_train_progress() {
    # Returns: phase done total eta_sec
    # phase = "perm" | "ig" | "none"
    # eta_sec = -1 means "not enough data to estimate"
    local logfile="$1" ep_done="${2:-0}"
    [[ -f "$logfile" ]] || { echo "none 0 0 -1"; return; }
    python3 - "$logfile" "$ep_done" 2>/dev/null <<'PY'
import sys, re, time
from datetime import datetime

logfile = sys.argv[1]
ep_done = int(sys.argv[2]) if len(sys.argv) > 2 else 0

try:
    with open(logfile, 'r', errors='replace') as f:
        content = f.read()
    lines = content.splitlines()
except Exception:
    print("none 0 0 -1"); sys.exit()

perm_lines = [l for l in lines if 'perm_importance_group' in l]
ig_lines_  = [l for l in lines if 'ig_epoch' in l and 'ig_importance_start' not in l]

ts_re = re.compile(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})')
def parse_ts(line):
    m = ts_re.search(line)
    if not m: return None
    s = m.group(1).replace('T', ' ')
    try: return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timestamp()
    except: return None

now = time.time()

if any('ig_importance_start' in l for l in lines):
    # Post-training IG: progress tracked via ig_top_feature lines (10 per class = 20 total)
    ig_feat_lines = [l for l in lines if 'ig_top_feature' in l]
    n_done  = len(ig_feat_lines)
    total   = 20   # 10 features × 2 classes
    ts_start_line = next((l for l in lines if 'ig_importance_start' in l), None)
    ts_first = parse_ts(ts_start_line) if ts_start_line else None
    ts_last  = parse_ts(ig_feat_lines[-1]) if ig_feat_lines else None
    if ts_first and ts_last and n_done > 0:
        elapsed  = ts_last - ts_first
        rate     = elapsed / n_done
        left     = max(total - n_done, 0)
        eta_sec  = int(rate * left)
    elif ts_first and n_done == 0:
        # IG just started — no features logged yet; extrapolate from perm timing
        perm_ts  = [parse_ts(l) for l in perm_lines if parse_ts(l)]
        if len(perm_ts) > 1:
            avg_group_sec = (perm_ts[-1] - perm_ts[0]) / (len(perm_ts) - 1)
            eta_sec = int(avg_group_sec * 8)   # rough: IG ≈ 8 perm-group equivalents
        else:
            eta_sec = -1
    else:
        eta_sec = -1
    print("ig", n_done, total, eta_sec)
elif perm_lines:
    n_done = len(perm_lines)
    m = re.search(r'n_groups\s*=\s*(\d+)', content)
    total = int(m.group(1)) if m else max(21, n_done)
    ts_first = parse_ts(perm_lines[0])
    ts_last  = parse_ts(perm_lines[-1])
    if ts_first and ts_last and n_done > 1:
        rate     = (ts_last - ts_first) / (n_done - 1)
        left     = max(total - n_done, 0)
        eta_sec  = int(rate * left)
    else:
        eta_sec = -1
    print("perm", n_done, total, eta_sec)
else:
    print("none 0 0 -1")
PY
}

# ── main loop ─────────────────────────────────────────────────────────────────

while true; do
    clear

    # ── Step 1: seed FOLD_DIRS from the single most-recently-modified fold dir ──
    # Only the ONE newest fold is seeded here. Step 2 overlays any live process
    # onto that slot (or adds it as a new entry if its fold number differs).
    unset FOLD_PIDS FOLD_DIRS FOLD_LOGS
    declare -A FOLD_PIDS FOLD_DIRS FOLD_LOGS

    latest_ver=""; latest_mtime=0; latest_fold_dir=""
    for dir in "${REPO_DIR}/data/legacy/checkpoints/model_v"*"_fold"*/; do
        [[ -d "$dir" ]] || continue
        mt=$(stat -c '%Y' "$dir" 2>/dev/null || echo 0)
        if (( mt > latest_mtime )); then
            latest_mtime=$mt
            latest_ver=$(basename "${dir%/}" | grep -oP 'v\d+' || true)
            latest_fold_dir="${dir%/}"
        fi
    done
    if [[ -n "$latest_fold_dir" ]]; then
        sfn=$(basename "$latest_fold_dir" | grep -oP '(?<=fold)\d+' || true)
        if [[ -n "$sfn" ]]; then
            FOLD_DIRS[$sfn]="$latest_fold_dir"
            FOLD_LOGS[$sfn]="/tmp/train_${latest_ver}_fold${sfn}.log"
        fi
    fi

    # ── Step 2: overlay live process info (PID + correct log path) ────────────
    # Running processes take priority: their --save-dir is authoritative and
    # /proc/$pid/fd/1 gives the exact file they are writing to.
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        pid=$(echo "$line" | awk '{print $1}')
        sd=$(echo "$line" | grep -oP '(?<=--save-dir )\S+|(?<=--save-dir=)\S+' || true)
        [[ -z "$sd" ]] && continue
        [[ "${sd:0:1}" != "/" ]] && sd="${REPO_DIR}/${sd}"
        fn=$(basename "$sd" | grep -oP '(?<=fold)\d+' || echo "1")
        ver=$(basename "$sd" | grep -oP 'v\d+' || echo "")
        FOLD_PIDS[$fn]=$pid
        FOLD_DIRS[$fn]=$sd
        # Resolve the log the process is actually writing to via fd/1.
        actual_log=$(readlink "/proc/${pid}/fd/1" 2>/dev/null || true)
        if [[ -n "$actual_log" && "$actual_log" != "pipe:"* && -f "$actual_log" ]]; then
            FOLD_LOGS[$fn]="$actual_log"
        elif [[ -n "$ver" ]]; then
            repo_log="${REPO_DIR}/logs/${ver}_fold${fn}.log"
            tmp_log="/tmp/train_${ver}_fold${fn}.log"
            if [[ -f "$repo_log" ]]; then
                FOLD_LOGS[$fn]="$repo_log"
            else
                FOLD_LOGS[$fn]="$tmp_log"
            fi
        fi
    done < <(pgrep -a -f "src[.]training[.]trainer" 2>/dev/null || true)

    # ── derive N_FOLDS from the detected set ──────────────────────────────────
    N_FOLDS=0
    for fn in "${!FOLD_DIRS[@]}"; do
        (( fn > N_FOLDS )) && N_FOLDS=$fn
    done

    # ── system RAM ────────────────────────────────────────────────────────────
    read -r MEM_T MEM_U MEM_P < <(
        awk '/^MemTotal/{t=$2}/^MemAvailable/{a=$2}
             END{u=(t-a)/1024/1024; printf "%.1f %.1f %.0f",t/1024/1024,u,u*100/(t/1024/1024)}' \
        /proc/meminfo
    )

    # ── count active / done ───────────────────────────────────────────────────
    ACTIVE_CT=0; DONE_CT=0
    for fn in "${!FOLD_DIRS[@]}"; do
        if   [[ -n "${FOLD_PIDS[$fn]:-}" ]];                                  then (( ACTIVE_CT++ )) || true
        elif [[ -f "${FOLD_DIRS[$fn]}/training_result.json" ]];               then (( DONE_CT++   )) || true
        fi
    done

    # ── header ────────────────────────────────────────────────────────────────
    echo "╔══════════════════════════════════════════════════════════╗"
    printf "║       TRAINING MONITOR   %-32s║\n" "$(date '+%H:%M:%S %Z')"
    SEP
    ROW "$(printf "%d fold(s) active  •  %d done  •  %d detected" \
            "$ACTIVE_CT" "$DONE_CT" "$N_FOLDS")"
    printf "║  RAM    [%s]  %-25s║\n" "$(bar "${MEM_P%.*}")" "${MEM_P%.*}%  ${MEM_U}/${MEM_T} GB"
    SEP

    # ── no folds detected ─────────────────────────────────────────────────────
    if (( N_FOLDS == 0 )); then
        ROW "⚠  No trainer processes or checkpoint dirs detected."
        echo "╚══════════════════════════════════════════════════════════╝"
        echo "  Ctrl+C to exit  •  refresh ${REFRESH}s"
        sleep $REFRESH; continue
    fi

    # ── per-fold rows (sorted by fold number) ─────────────────────────────────
    first=1
    for i in $(echo "${!FOLD_DIRS[@]}" | tr ' ' '\n' | sort -n); do
        [[ "$first" -eq 0 ]] && SEP
        first=0

        FPID="${FOLD_PIDS[$i]:-}"
        fd="${FOLD_DIRS[$i]}"
        logfile="${FOLD_LOGS[$i]:-}"

        if   [[ -n "$FPID" ]];                        then st="ACTIVE"
        elif [[ -f "${fd}/training_result.json" ]];   then st="DONE"
        else                                               st="PENDING"
        fi

        tw=$(fold_test_label "$FPID" "$fd" 2>/dev/null || echo "?")
        [[ -z "$tw" ]] && tw="?"

        case "$st" in
        DONE)
            ROW "Fold $i  [DONE]  test=${tw}  dir=$(basename "$fd")"
            while IFS= read -r ml; do ROW "$ml"; done \
                < <(metrics_lines "$fd" 2>/dev/null || true)
            while IFS= read -r ml; do ROW "$ml"; done \
                < <(ig_lines "$fd" 2>/dev/null || true)
            ;;

        ACTIVE)
            F_RSS="--"; F_ELSEC=0
            RSS_KB=$(awk '/^VmRSS/{print $2}' "/proc/${FPID}/status" 2>/dev/null || echo 0)
            F_RSS=$(awk "BEGIN{printf \"%.1f\", ${RSS_KB:-0}/1024/1024}")
            F_ELSEC=$(ps -p "$FPID" -o etimes= 2>/dev/null | tr -d ' ' || echo 0)
            F_EL=$(fmt_sec "${F_ELSEC:-0}")

            # All progress metrics come from the log — nothing hardcoded
            read ep_done ep_total st_total cur_step step_ms avg_ep_sec \
                <<< "$(fold_log_stats "$logfile")"

            # Detect post-training phase: perm_importance or full IG running
            post_phase=""
            if [[ -f "$logfile" ]]; then
                if grep -q "ig_importance_start" "$logfile" 2>/dev/null; then
                    post_phase="ig"
                elif grep -q "perm_importance_group" "$logfile" 2>/dev/null; then
                    post_phase="perm"
                fi
            fi

            # Epoch-level progress bar
            ep_pct=0
            (( ep_total > 0 )) && ep_pct=$(( ep_done * 100 / ep_total ))
            [[ -n "$post_phase" ]] && ep_pct=100

            # Step-level progress bar (within current epoch)
            step_pct=0
            (( st_total > 0 && cur_step > 0 )) && step_pct=$(( cur_step * 100 / st_total ))

            # Step rate display
            if (( step_ms >= 1000 )); then
                rate_disp="$(awk "BEGIN{printf \"%.1f\", ${step_ms}/1000}")s/step"
            else
                rate_disp="${step_ms}ms/step"
            fi

            # ETA — estimate avg_ep_sec from current rate if no completed epochs yet
            eta_str="--"; eta_wall=""
            if (( step_ms > 0 && st_total > 0 )); then
                (( avg_ep_sec == 0 )) && avg_ep_sec=$(( st_total * step_ms / 1000 ))
                eta_cur=$(( (st_total - cur_step) * step_ms / 1000 ))
                ep_left=$(( ep_total - ep_done - 1 ))
                (( ep_left < 0 )) && ep_left=0
                eta_total=$(( eta_cur + ep_left * avg_ep_sec ))
                eta_str="$(fmt_sec $eta_total)"
                eta_wall=$(date -d "+${eta_total} seconds" '+%H:%M:%S' 2>/dev/null || true)
            fi

            if [[ -n "$post_phase" ]]; then
                # Training epochs done; running post-training importance analysis
                read pt_phase pt_done pt_total pt_eta_sec \
                    <<< "$(post_train_progress "$logfile" "$ep_done")"
                pt_done=${pt_done:-0}; pt_total=${pt_total:-0}; pt_eta_sec=${pt_eta_sec:--1}
                pt_pct=0
                (( pt_total > 0 && pt_done > 0 )) && pt_pct=$(( pt_done * 100 / pt_total ))
                pt_eta_str="--"; pt_eta_wall=""
                if (( pt_eta_sec >= 0 )); then
                    pt_eta_str="$(fmt_sec $pt_eta_sec)"
                    pt_eta_wall=$(date -d "+${pt_eta_sec} seconds" '+%H:%M:%S' 2>/dev/null || true)
                fi

                [[ "$post_phase" == "ig" ]] && phase_lbl="IG" || phase_lbl="Perm"

                if [[ -n "$pt_eta_wall" ]]; then
                    pt_right="$(printf "%d/%d → %s" "$pt_done" "$pt_total" "$pt_eta_wall")"
                    pt_eta_row="  ⏳ ${phase_lbl}: ETA ~${pt_eta_wall}  (in ${pt_eta_str})"
                elif (( pt_done > 0 )); then
                    pt_right="$(printf "%d/%d estimating..." "$pt_done" "$pt_total")"
                    pt_eta_row="  ⏳ ${phase_lbl}: ${pt_done}/${pt_total} done  ETA estimating..."
                else
                    pt_right="waiting for first group..."
                    pt_eta_row="  ⏳ ${phase_lbl}: waiting for first group..."
                fi

                ROW "$(printf "Fold %d  [POST-TRAIN]  test=%-8s  up %s  RSS %s GB" \
                        "$i" "$tw" "$F_EL" "$F_RSS")"
                BARROW "Epochs" "100" \
                    "$(printf "%d/%d ep (stopped)" "$ep_done" "$ep_total")"
                BARROW "$phase_lbl" "$pt_pct" "$pt_right"
                ROW "$pt_eta_row"
            else
                ROW "$(printf "Fold %d  [ACTIVE]  test=%-8s  up %s  RSS %s GB" \
                        "$i" "$tw" "$F_EL" "$F_RSS")"
                BARROW "Epochs" "$ep_pct" \
                    "$(printf "%d/%d done" "$ep_done" "$ep_total")"
                BARROW "Step  " "$step_pct" \
                    "$(printf "%d/%d  %s" "$cur_step" "$st_total" "$rate_disp")"
                if [[ -n "$eta_wall" ]]; then
                    ROW "$(printf "  ETA ~%s  (in %s)" "$eta_wall" "$eta_str")"
                else
                    ROW "  ETA: waiting for first step..."
                fi
            fi
            # Latest per-epoch metrics from live log
            _em=$(live_epoch_metrics "$logfile" 2>/dev/null || true)
            if [[ -n "$_em" ]]; then
                SEP
                while IFS= read -r ml; do ROW "$ml"; done <<< "$_em"
            fi
            ;;

        PENDING)
            ROW "Fold $i  [PENDING]  dir=$(basename "$fd")"
            ;;
        esac
    done

    SEP
    if (( ACTIVE_CT == 0 && DONE_CT > 0 )); then
        ROW "✓  All detected folds complete."
    elif (( ACTIVE_CT == 0 )); then
        ROW "⚠  No active trainer process."
    fi
    echo "╚══════════════════════════════════════════════════════════╝"
    echo "  Ctrl+C to exit  •  refresh ${REFRESH}s"

    sleep $REFRESH
done
