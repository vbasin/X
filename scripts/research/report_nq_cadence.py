#!/usr/bin/env python3
"""Export completed cadence research artifacts; never train or open raw data.

The report distinguishes discovery, historical validation, exposed compatibility
and an optional consulted reserve. Existing report directories are preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
import time
from typing import Any

import numpy as np
import pandas as pd


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def money(value: Any) -> str:
    return '—' if value is None or pd.isna(value) else f'${float(value):,.0f}'


def number(value: Any, decimals: int = 2) -> str:
    return '—' if value is None or pd.isna(value) else f'{float(value):,.{decimals}f}'


def readable_policy(policy: dict) -> str:
    direction={'long_only':'long only','both':'long and short'}.get(policy.get('direction_mode'),policy.get('direction_mode'))
    return (f"{direction}; prior 14-session same-clock bands × {policy.get('multiplier')}; "
            f"decisions every {policy.get('cadence_minutes')} minutes; "
            f"VWAP {'enabled' if policy.get('use_vwap') else 'disabled'}; "
            f"fixed opposite-band emergency stop {'enabled' if policy.get('emergency_stop') else 'disabled'}; "
            f"maximum {policy.get('max_entries')} entries/day, cease after two net losses; "
            "entries 09:00–14:00 CT, flatten 15:00 CT.")


def metric_row(label: str, scenario: str, metrics: dict) -> dict:
    ci=metrics.get('daily_mean_95ci',[None,None])
    row={'cohort':label,'scenario':scenario}
    for key in ['n_sessions','n_trades','net_usd','profit_factor','win_rate','loss_count',
                'total_loss_usd','max_drawdown_usd','mean_daily_usd','unresolved']:
        row[key]=metrics.get(key)
    row['daily_mean_ci_lower_usd'],row['daily_mean_ci_upper_usd']=ci
    return row


def nested_benchmarks(value: Any, prefix: str = '') -> list[tuple[str,dict]]:
    found=[]
    if isinstance(value,dict):
        if 'net_usd' in value:
            return [(prefix or 'benchmark',value)]
        for key,item in value.items():
            found.extend(nested_benchmarks(item,f'{prefix}/{key}'.strip('/')))
    elif isinstance(value,list):
        for i,item in enumerate(value):
            key=str(item.get('name',item.get('cohort',i))) if isinstance(item,dict) else str(i)
            found.extend(nested_benchmarks(item,f'{prefix}/{key}'.strip('/')))
    return found


def build_report(root: Path) -> Path:
    """Render only existing saved outcomes; return the new report directory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.dates as mdates

    root=root.resolve()
    frozen=read_json(root/'frozen_policy.json')
    trials=read_json(root/'discovery_trials.json')
    policy_id=frozen['selected']['policy_id']
    selected=next(t for t in trials if t['policy_id']==policy_id)
    policy=frozen['selected']['policy']
    protocol=read_json(root/'protocol.json')
    dest=root/'report'
    dest.mkdir(exist_ok=False)
    trial_rows=[]
    for trial in trials:
        metrics=trial['metrics']
        trial_rows.append(dict(policy_id=trial['policy_id'],**trial['policy'],
            net_usd=metrics.get('net_usd'),profit_factor=metrics.get('profit_factor'),
            n_trades=metrics.get('n_trades'),loss_count=metrics.get('loss_count'),
            total_loss_usd=metrics.get('total_loss_usd'),max_drawdown_usd=metrics.get('max_drawdown_usd'),
            unresolved=metrics.get('unresolved'),worst_half_daily_sharpe=trial.get('score'),
            positive_years=trial.get('positive_years'),qualifies=trial.get('qualifies'),
            selected=trial['policy_id']==policy_id,
            stress_first_half_net_usd=trial.get('discovery_stress_half_net',[None,None])[0],
            stress_second_half_net_usd=trial.get('discovery_stress_half_net',[None,None])[1]))
    pd.DataFrame(trial_rows).to_csv(dest/'discovery_trials.csv',index=False)
    cohorts=[];rows=[];ledgers=[];source_paths=[root/'frozen_policy.json',root/'discovery_trials.json',root/'protocol.json']
    gates={'discovery_selection':bool(frozen.get('selection_passed',False))}
    robustness_path=root/'robustness_diagnostics.json'
    robustness=read_json(robustness_path) if robustness_path.exists() else {}
    if robustness_path.exists():source_paths.append(robustness_path)

    def add(label: str, key: str, directory: Path, metrics: dict, scenario: str,
            base: bool = False) -> None:
        rows.append(metric_row(label,scenario,metrics))
        daily_path=directory/'daily.csv'
        ledger_path=directory/'trade_ledger.parquet'
        source_paths.extend([daily_path,ledger_path])
        if not daily_path.exists() or not ledger_path.exists():
            raise ValueError(f'Incomplete completed-cohort artifacts: {directory}')
        daily=pd.read_csv(daily_path,index_col=0)
        daily.index=daily.index.astype(str)
        daily=daily.sort_index()
        ledger=pd.read_parquet(ledger_path)
        ledger.to_csv(dest/f'trades_{key}_{scenario}.csv',index=False)
        ledgers.append((key,scenario,ledger))
        if base:
            cohorts.append(dict(label=label,key=key,metrics=metrics,daily=daily))

    discovery=root/'discovery'/f'policy_{policy_id}'
    add('2011–2020 TRAIN / selected discovery','discovery',discovery,selected['metrics'],'base',True)
    stress_path=discovery/'stress_2pt.parquet'
    if stress_path.exists():
        pd.read_parquet(stress_path).to_csv(dest/'trades_discovery_stress_2pt.csv',index=False)
        source_paths.append(stress_path)
    for scenario in ['1','2','4']:
        directory=root/'validation'/f'cost_{scenario}'
        if (directory/'metrics.json').exists():
            metrics=read_json(directory/'metrics.json');source_paths.append(directory/'metrics.json')
            add('2021–2024 historical validation','validation',directory,metrics,f'cost_{scenario}pt',scenario=='1')
    if (root/'validation_gate.json').exists():
        gates['historical_validation']=read_json(root/'validation_gate.json')
        source_paths.append(root/'validation_gate.json')

    for phase,label in [('compatibility','Apr 2025–Jun 2026 EXPOSED compatibility'),
                        ('holdout','Jul–Sep 2026 consulted reserve')]:
        phase_dir=root/phase
        if not (phase_dir/'base/metrics.json').exists():
            continue
        lock=read_json(phase_dir/'evaluation_lock.json')
        if phase=='holdout' and not lock.get('reserved_read',False):
            raise ValueError('Reserve metrics require an existing consultation marker')
        source_paths.append(phase_dir/'evaluation_lock.json')
        for scenario in ['base','slip_0.5','slip_1','latency_1000ms']:
            directory=phase_dir/scenario
            if (directory/'metrics.json').exists():
                source_paths.append(directory/'metrics.json')
                add(label,phase,directory,read_json(directory/'metrics.json'),scenario,scenario=='base')
        if (phase_dir/'gate.json').exists():
            gates[phase]=read_json(phase_dir/'gate.json');source_paths.append(phase_dir/'gate.json')

    reserve_present=any(c['key']=='holdout' for c in cohorts)
    reserve_marker=(root/'holdout/evaluation_lock.json').exists()
    reserve_status=('CONSULTED — results reported separately' if reserve_present else
                    'CONSULTATION MARKER EXISTS — completed base results absent' if reserve_marker else
                    'NOT CONSULTED by this run; no reserve result inferred')
    metrics_table=pd.DataFrame(rows)
    metrics_table.to_csv(dest/'metrics_table.csv',index=False)
    audits=[]
    for path in [root/'audit/audit.json',root/'compatibility/audit/audit.json',
                 root/'holdout/audit/audit.json',root/'compatibility/audit_marks/audit.json',
                 root/'holdout/audit_marks/audit.json']:
        if path.exists():
            audits.append((str(path.relative_to(root)),read_json(path)));source_paths.append(path)
    bench_path=root/'benchmarks/summary.json'
    benchmarks=nested_benchmarks(read_json(bench_path),'historical') if bench_path.exists() else []
    if bench_path.exists():source_paths.append(bench_path)
    comparisons=[]
    for phase in ['compatibility','holdout']:
        directory=root/phase/'benchmarks'
        for relative in ['base/metrics.json','summary.json']:
            path=directory/relative
            if path.exists():
                benchmarks.extend(nested_benchmarks(read_json(path),f'{phase}/benchmark'))
                source_paths.append(path)
                break
        for filename in ['comparison.json','paired_comparison.json']:
            paired=directory/filename
            if paired.exists():
                comparisons.append((phase,read_json(paired)));source_paths.append(paired)
                break

    monthly=[]
    for cohort in cohorts:
        if cohort['key'] not in ('compatibility','holdout'):continue
        daily=cohort['daily']
        for month,group in daily.groupby(daily.index.str[:7]):
            monthly.append(dict(cohort=cohort['key'],month=month,n_sessions=len(group),
                                net_usd=float(group.net_usd.sum())))
    monthly_frame=pd.DataFrame(monthly,columns=['cohort','month','n_sessions','net_usd'])
    monthly_frame.to_csv(dest/'monthly_recent.csv',index=False)

    # Separate origins and calendar axes prevent fabricated continuous capital.
    chart_rows=(len(cohorts)+1)//2
    chart,axes=plt.subplots(chart_rows,2,figsize=(13.5,4*chart_rows),squeeze=False)
    for ax,cohort in zip(axes.flat,cohorts):
        daily=cohort['daily'];dates=pd.to_datetime(daily.index)
        equity=daily.net_usd.cumsum()
        dd=equity-equity.cummax().clip(lower=0)
        ax.plot(dates,equity,color='#176875',lw=1.4,label='Cumulative net $')
        ax.fill_between(dates,dd,0,color='#b44747',alpha=.4,label='Daily closed drawdown $')
        ax.axhline(0,color='gray',lw=.6)
        ax.set_title(cohort['label'],fontsize=10)
        ax.set_ylabel('USD / one NQ contract')
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(alpha=.15);ax.legend(fontsize=8,loc='best')
    for ax in list(axes.flat)[len(cohorts):]:ax.set_visible(False)
    chart.suptitle('Independent cohort equity curves — each starts at $0; no compounding',fontsize=13)
    chart.tight_layout(rect=(0,0,1,.96))
    chart.savefig(dest/'equity_drawdown.png',dpi=170)

    def gate_text() -> str:
        return '; '.join(f'{name}: '+('PASS' if (value.get('passed',False) if isinstance(value,dict) else value) else 'FAIL')
                         for name,value in gates.items())

    limitations=("Historical bars: price-friction proxy, not bid/ask execution. Recent quotes: trade-sampled "
        "pretrade BBO; fills assume liquidity at those snapshots. Unobserved quote moves, depth, impact and live "
        "fill quality are not established. Closed-trade/daily drawdown excludes open-position peaks; any observed "
        "liquidation audit is reported separately. Discovery is selected training evidence; compatibility reuses "
        "exposed dates. Confidence intervals use five-session block resampling, not a guarantee. No account "
        "percentage return, deployment, globally optimal strategy, or future/live profit is claimed.")
    readable=readable_policy(policy)
    base_rows=[metric_row(c['label'],'base',c['metrics']) for c in cohorts]
    header=['Cohort','Net $','PF','Trades','Losing trades','Total loss $','Closed DD $','Daily mean [95% CI]']
    def display_row(row):
        ci='not estimated' if row['daily_mean_ci_lower_usd'] is None else f"[{money(row['daily_mean_ci_lower_usd'])}, {money(row['daily_mean_ci_upper_usd'])}]"
        return [row['cohort'],money(row['net_usd']),number(row['profit_factor']),number(row['n_trades'],0),
                number(row['loss_count'],0),money(row['total_loss_usd']),money(row['max_drawdown_usd']),
                f"{money(row['mean_daily_usd'])} {ci}"]
    displayed=[display_row(r) for r in base_rows]
    concentration=[];yearly_robustness=[]
    for label,diagnostic in robustness.items():
        if not isinstance(diagnostic,dict):continue
        best=diagnostic.get('best_days',{})
        best_day=max(best,key=best.get) if best else None
        concentration.append(dict(cohort=label,net_usd=diagnostic.get('net_usd'),
            best_day=best_day,best_day_usd=best.get(best_day),
            net_without_best_day_usd=diagnostic.get('net_without_best_day_usd'),
            net_without_best_5_days_usd=diagnostic.get('net_without_best_5_days_usd'),
            positive_months=diagnostic.get('positive_months'),negative_months=diagnostic.get('negative_months')))
        yearly_robustness.extend(dict(cohort=label,year=str(year),net_usd=net)
            for year,net in diagnostic.get('yearly',{}).items())
    if concentration:
        pd.DataFrame(concentration).to_csv(dest/'robustness_concentration.csv',index=False)
        pd.DataFrame(yearly_robustness).to_csv(dest/'robustness_yearly.csv',index=False)
    recent=robustness.get('recent_exposed_compatibility',{})
    lead_warning=''
    if recent:
        last_years=recent.get('yearly',{})
        period_loss=last_years.get('2026')
        lead_warning='Consistent-profit and low-drawdown goals are NOT established. '
        if period_loss is not None:
            lead_warning+=f'Recent Jan–Jun 2026: {money(period_loss)}. '
        lead_warning+=f"Recent net without its best day: {money(recent.get('net_without_best_day_usd'))}. "
        recent_audit=next((a for name,a in audits if name.startswith('compatibility/')),None)
        if recent_audit is not None:
            lead_warning+=f"Observed liquidation drawdown: {money(recent_audit.get('maximum_observed_liquidation_drawdown_usd',recent_audit.get('full_observed_liquidation_mark_drawdown_usd')))}."

    summary=['# NQ cadence-confirmed research report','',
             *([f'**{lead_warning}**',''] if lead_warning else []),
             '**Historical simulation; one NQ contract. Not a live-profit claim or globally optimal model.**','',
             f'Run: `{root}`',f'Policy {policy_id}: {readable}',f'Gates: {gate_text()}',
             f'Reserve: {reserve_status}','',
             '| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']
    summary.extend('| '+' | '.join(r)+' |' for r in displayed)
    if concentration:
        summary+=['','## Concentration and stability — descriptive after freeze','',
                  'Positive aggregate P&L does not establish consistent performance. Removing winners is a sensitivity calculation, '
                  'not an implementable trading rule or an input for retuning.','',
                  '| Cohort | Net USD | Best day | Best-day USD | Net without best day | Net without best five days | Positive / negative months |',
                  '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
        summary.extend(f"| {r['cohort']} | {money(r['net_usd'])} | {r['best_day']} | {money(r['best_day_usd'])} | "
                       f"{money(r['net_without_best_day_usd'])} | {money(r['net_without_best_5_days_usd'])} | "
                       f"{r['positive_months']} / {r['negative_months']} |" for r in concentration)
        summary+=['','| Cohort | Year / observed portion | Net USD |','| --- | --- | ---: |']
        summary.extend(f"| {r['cohort']} | {r['year']}"+
                       (' (Jan–Jun only)' if r['cohort']=='recent_exposed_compatibility' and r['year']=='2026' else '')+
                       f" | {money(r['net_usd'])} |" for r in yearly_robustness)
        summary+=['',f'[Full robustness diagnostics]({robustness_path})']
    summary+=['','Dollar totals use one contract; no continuous compounding across cohorts.',
              'Trade CSVs retain invalid/unresolved rows. Reported realized totals follow the original metrics; unresolved counts remain in metrics_table.csv.','',
              '## Gate records','','```json',json.dumps(gates,indent=2),'```','',
              '## Cost sensitivity','', '| Cohort | Scenario | Net USD | Unresolved |','| --- | --- | ---: | ---: |']
    summary.extend(f"| {r['cohort']} | {r['scenario']} | {money(r['net_usd'])} | {r['unresolved']} |" for r in rows)
    summary+=['','## Recent monthly results','', '| Cohort | Month | Sessions | Net USD |','| --- | --- | ---: | ---: |']
    summary.extend(f"| {r['cohort']} | {r['month']} | {r['n_sessions']} | {money(r['net_usd'])} |" for r in monthly)
    if not monthly:summary.append('No completed recent cohort is present.')
    summary+=['','## Audits and benchmarks','']
    for name,audit in audits:
        summary.append(f"- `{name}`: passed={audit.get('audit_passed','not supplied')}; "
                       f"trades audited={audit.get('trades_audited',audit.get('n_trades_verified','not supplied'))}; "
                       f"observed liquidation drawdown={money(audit.get('maximum_observed_liquidation_drawdown_usd',audit.get('full_observed_liquidation_mark_drawdown_usd')))}.")
    if not audits:summary.append('No independent audit artifact was available when this report was generated.')
    for name,stats in benchmarks:
        summary.append(f"- Benchmark `{name}`: net {money(stats.get('net_usd'))}, "
                       f"closed drawdown {money(stats.get('max_drawdown_usd'))}, "
                       f"observed liquidation drawdown {money(stats.get('maximum_observed_liquidation_drawdown_usd'))}, "
                       f"trades {stats.get('n_trades','—')}.")
    if not benchmarks:summary.append('No benchmark summary was present; no incremental timing advantage is inferred.')
    for phase,comparison in comparisons:
        summary.extend(['',f'Paired comparison, {phase}:','','```json',json.dumps(comparison,indent=2),'```'])
    summary+=['','## Complete discovery search','',
              f'All {len(trial_rows)} evaluated configurations are disclosed below and in discovery_trials.csv. '
              'These are selected training comparisons, not independent validation.',
              '', '| ID | Multiplier | Cadence | Direction | Emergency | Net USD | PF | Worst-half daily Sharpe | Qualified | Selected |',
              '| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |']
    summary.extend(f"| {r['policy_id']} | {r.get('multiplier')} | {r.get('cadence_minutes')} | "
                   f"{r.get('direction_mode')} | {r.get('emergency_stop')} | {money(r['net_usd'])} | "
                   f"{number(r['profit_factor'])} | {number(r['worst_half_daily_sharpe'])} | "
                   f"{r['qualifies']} | {r['selected']} |" for r in trial_rows)
    summary+=['','## Execution and interpretation','',
              f"Historical friction: 1/2/4 points round trip plus ${protocol.get('commission',3.36):.2f} commission. "
              'Recent base: recorded spread, 0.25 point per side, 250ms latency; stresses 0.5/1 point per side and 1000ms latency.',
              limitations,'','## Sources and reproducibility','',
              f'Command: `venv/bin/python -m scripts.research.report_nq_cadence {root}`',
              'The exporter refuses to overwrite an existing report directory. It reads saved artifacts only; no raw market data or fitting is performed.','']
    summary.extend(f'- [{p.relative_to(root)}]({p})' for p in dict.fromkeys(source_paths))
    summary+=['',f'- [Protocol source]({root}/code/docs/research/NQ_CADENCE_AREA_PROTOCOL.md)',
              f'- [Run manifest]({root}/manifest.json)',f'- [Source hashes]({root}/source_hashes.json)']
    (dest/'summary.md').write_text('\n'.join(summary)+'\n')

    def page(title):
        fig=plt.figure(figsize=(11.69,8.27),facecolor='white')
        fig.text(.055,.95,title,fontsize=19,weight='bold',color='#173c46')
        fig.text(.055,.025,f'NQ research | {root.name} | simulated one-contract USD',fontsize=8,color='#666')
        return fig

    def wrapped(fig,x,y,text,width=125,size=10,lineheight=.025,color='#222222'):
        lines=textwrap.wrap(text,width=width,break_long_words=True,break_on_hyphens=False)
        fig.text(x,y,'\n'.join(lines),va='top',fontsize=size,linespacing=1.45,color=color)
        return y-len(lines)*lineheight

    def table(fig,headers,data,bounds,colwidths=None,size=8.5):
        ax=fig.add_axes(bounds);ax.axis('off')
        tab=ax.table(cellText=data,colLabels=headers,cellLoc='left',colLoc='left',
                     loc='upper left',colWidths=colwidths)
        tab.auto_set_font_size(False);tab.set_fontsize(size);tab.scale(1,1.55)
        for (r,c),cell in tab.get_celld().items():
            cell.set_edgecolor('#d9e1e4')
            if r==0:cell.set_facecolor('#e7f0f2');cell.set_text_props(weight='bold')
        for r,values in enumerate([headers,*data]):
            lines=max(str(v).count('\n')+1 for v in values)
            height=(lines*(size+3)+3)/(fig.get_figheight()*72*bounds[3])
            for c in range(len(headers)):tab[(r,c)].set_height(height)
        return ax

    with PdfPages(dest/'NQ_Cadence_Research_Report.pdf') as pdf:
        fig=page('NQ cadence-confirmed model: evidence and limits')
        if lead_warning:wrapped(fig,.055,.9,lead_warning,width=138,size=11,color='#9e2626')
        wrapped(fig,.055,.80 if lead_warning else .895,readable,width=148,size=9.5)
        wrapped(fig,.055,.695 if lead_warning else .79,'Gates: '+gate_text(),width=150,size=9.5)
        wrapped(fig,.055,.64 if lead_warning else .735,'Reserve: '+reserve_status,width=150,size=9.5)
        pdf_rows=[]
        for r in displayed:
            short=r.copy();short[0]='\n'.join(textwrap.wrap(short[0],26));short[-1]='\n'.join(textwrap.wrap(short[-1],22));pdf_rows.append(short)
        table(fig,header,pdf_rows,[.035,.385 if lead_warning else .44,.93,.20],
              [.215,.085,.055,.07,.10,.10,.10,.21],8)
        wrapped(fig,.055,.32 if lead_warning else .37,f"Search: {len(trial_rows)} discovery configurations, {sum(bool(t.get('qualifies')) for t in trials)} qualified. "
                'Full configurations/results: discovery_trials.csv. Discovery is TRAIN; recent compatibility is EXPOSED. '
                'Each cohort starts separately; no compounding.',width=145,size=10)
        wrapped(fig,.055,.225 if lead_warning else .275,f"Costs: historical 1/2/4 points round trip + ${protocol.get('commission',3.36):.2f}. Recent base: recorded spread + 0.25 point/side, 250ms; stresses 0.5/1 point/side and 1000ms. Full scenario metrics and unresolved counts: metrics_table.csv.",width=145,size=10)
        wrapped(fig,.055,.16,'Research only. Positive results do not establish a globally optimal model, consistent future profits, or live execution quality. Percentage account returns are not reported.',width=145,size=10)
        pdf.savefig(fig);plt.close(fig)
        pdf.savefig(chart);plt.close(chart)

        fig=page('Recent monthly evidence, audits and provenance')
        if lead_warning:wrapped(fig,.055,.895,lead_warning,width=145,size=10.5,color='#9e2626')
        month_rows=[[r['cohort'],r['month'],str(r['n_sessions']),money(r['net_usd'])] for r in monthly]
        if month_rows:
            table(fig,['Cohort','Month','Days','Net USD'],month_rows[:22],
                  [.045,.10,.45,.68 if lead_warning else .77],[.36,.22,.14,.27],8)
            if len(month_rows)>22:
                fig.text(.06,.07,'Additional months: monthly_recent.csv and summary.md',fontsize=8)
        else:
            wrapped(fig,.055,.84,'No completed recent cohort is present.',width=58)
        y=.785 if lead_warning else .875
        for name,audit in audits:
            y=wrapped(fig,.55,y,f"Audit {name}: passed={audit.get('audit_passed','not supplied')}; "
                f"audited trades={audit.get('trades_audited',audit.get('n_trades_verified','not supplied'))}; observed liquidation DD "
                f"{money(audit.get('maximum_observed_liquidation_drawdown_usd',audit.get('full_observed_liquidation_mark_drawdown_usd')))}.",width=62,size=9,lineheight=.022)-.025
        if not audits:y=wrapped(fig,.55,y,'No independent audit artifact available at export.',width=62,size=9)-.02
        pdf_benchmarks=[]
        for group in ['historical','compatibility','holdout']:
            chosen=next(((name,stats) for name,stats in benchmarks if name.startswith(group)),None)
            if chosen is not None:pdf_benchmarks.append(chosen)
        for name,stats in pdf_benchmarks:
            y=wrapped(fig,.55,y,f"{name.split('/')[0]} long benchmark: net {money(stats.get('net_usd'))}, "
                      f"closed DD {money(stats.get('max_drawdown_usd'))}, "
                      f"observed mark DD {money(stats.get('maximum_observed_liquidation_drawdown_usd'))}.",
                      width=62,size=9,lineheight=.022)-.015
        for phase,comparison in comparisons:
            ci=comparison.get('paired_mean_daily_95ci')
            interval=f"[{money(ci[0])}, {money(ci[1])}]" if ci is not None else 'unavailable'
            y=wrapped(fig,.55,y,f"{phase} model minus benchmark/day: "
                      f"{money(comparison.get('model_minus_benchmark_mean_daily_usd'))}; 95% CI {interval}.",
                      width=65,size=8.5,lineheight=.020)-.01
        concise_limits=('Trade-sampled BBO does not establish live liquidity or unobserved market paths. '
                        'Chart drawdown is closed equity. Discovery is TRAIN; compatibility is EXPOSED. '
                        'No account return, guaranteed profit, or global optimum is claimed. '
                        'Full costs, audit details and benchmarks: summary.md.')
        y=wrapped(fig,.55,y,concise_limits,width=66,size=8.5,lineheight=.019)-.02
        wrapped(fig,.55,y,'Source run: '+str(root)+' | Full artifact paths and gate records are listed in summary.md. Source snapshots: code/. Frozen specification: protocol.json; policy: frozen_policy.json. Trade CSVs preserve unresolved rows.',width=69,size=8,lineheight=.018)
        pdf.savefig(fig);plt.close(fig)
    return dest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args();start=time.monotonic()
    print('[████░░░░░░░░░░░░░░░░] Export saved research evidence',flush=True)
    output=build_report(args.run)
    print(f'[████████████████████] ✅ Done in {time.monotonic()-start:.1f}s: {output}',flush=True)


if __name__=='__main__':main()
