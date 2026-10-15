/* Artifact-backed console. No external scripts, tracking, training or live API. */
'use strict';
const $ = s => document.querySelector(s);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, digits=4) => typeof v === 'number' ? v.toLocaleString('en-US', {maximumFractionDigits:digits}) : (v ?? '—');
const pct = v => typeof v === 'number' ? `${(100*v).toFixed(2)}%` : '—';
const pretty = v => esc(JSON.stringify(v, null, 2));
const colors = ['#37b6ff','#7ee0a3','#ffc857','#b59eff','#ff8787'];
const splitColors = {train:'#143246', earlystop:'#183a2a', calibration:'#3f3521', validation:'#312647', holdout:'#442a2c'};
let example;
let state = {tab:'dashboard', feature:'wf_context', split:'all', outcomes:false, model:'', predictionSplit:'validation', page:0};
const view = () => $('#view');
const heading = (title, text) => `<h1>${esc(title)}</h1><p class="lead">${esc(text)}</p>`;
const badge = (text, ok=true) => `<span class="pill ${ok?'ok':'warn'}">${esc(text)}</span>`;
const card = (label, value, sub='') => `<div class="card"><h3>${esc(label)}</h3><div class="metric">${esc(value)}</div><small>${esc(sub)}</small></div>`;
const details = (name, value) => `<details class="details"><summary>${esc(name)}</summary><pre>${pretty(value)}</pre></details>`;
const table = (columns, rows, limit=Infinity) => `<div class="table-wrap"><table><thead><tr>${columns.map(c=>`<th>${esc(c[0])}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,limit).map(r=>`<tr>${columns.map(c=>`<td>${esc(fmt(typeof c[1]==='function'?c[1](r):r[c[1]]))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const selection = () => example.models.find(m=>m.design_id===state.model) || example.models.find(m=>m.selected) || example.models[0];
const modelName = m => `Depth ${m.parameters.n_layers ?? m.depth ?? '?'} · ${m.design_id}${m.selected?' · selected':''}`;
const modelSelect = () => `<label>Transformer design<select id="model-choice">${example.models.map(m=>`<option value="${esc(m.design_id)}" ${m.design_id===selection().design_id?'selected':''}>${esc(modelName(m))}</option>`).join('')}</select></label>`;
const wireModel = render => { $('#model-choice')?.addEventListener('change', e => {state.model=e.target.value; render();}); };
const time = value => { const n = Date.parse(value); return Number.isFinite(n) ? n : 0; };
const splitNames = () => Object.keys(example.data.splits).sort((a,b)=>time(example.data.splits[a].start)-time(example.data.splits[b].start));
const splitRows = () => state.split==='all' ? example.data.rows : example.data.rows.filter(r=>time(r.ts)>=time(example.data.splits[state.split].start)&&time(r.ts)<=time(example.data.splits[state.split].end));

function chart(rows, series, {title='', shading=false, markers=[], range=null}={}) {
  const W=1100,H=285,left=78,right=18,top=22,bottom=42;
  if (!rows.length) return '<p class="muted">No observations in this population.</p>';
  let min=Infinity,max=-Infinity;
  for (const row of rows) for(const s of series) { const v=Number(row[s.key]); if(row[s.key]!==null&&row[s.key]!==undefined&&Number.isFinite(v)){min=Math.min(min,v);max=Math.max(max,v);} }
  for(const m of markers) { min=Math.min(min,m.value);max=Math.max(max,m.value); }
  if(range) [min,max]=range;
  if(!Number.isFinite(min)) return '<p class="muted">No numeric observations available.</p>';
  if(max===min){min-=.5;max+=.5;}
  const x0=time(rows[0].ts), x1=time(rows[rows.length-1].ts);
  const x = ts => left+(time(ts)-x0)/(x1-x0||1)*(W-left-right);
  const y = value => H-bottom-(Number(value)-min)/(max-min)*(H-top-bottom);
  let body='';
  if(shading) for(const [name,span] of Object.entries(example.data.splits)) {
    const a=Math.max(left,x(span.start)), b=Math.min(W-right,x(span.end));
    if(b>a) body+=`<rect x="${a}" y="${top}" width="${b-a}" height="${H-top-bottom}" fill="${splitColors[name]}" opacity=".7"><title>${esc(name)}</title></rect>`;
  }
  for(let i=0;i<5;i++) {const value=min+(max-min)*i/4;body+=`<line class="grid-line" x1="${left}" x2="${W-right}" y1="${y(value)}" y2="${y(value)}"/><text x="${left-9}" y="${y(value)+4}" text-anchor="end">${esc(fmt(value,Math.abs(max-min)<.1?5:2))}</text>`;}
  series.forEach((s,i)=>{
    const points=rows.filter(r=>r[s.key]!==null&&r[s.key]!==undefined&&Number.isFinite(Number(r[s.key]))).map(r=>`${x(r.ts).toFixed(2)},${y(r[s.key]).toFixed(2)}`).join(' ');
    body+=`<polyline class="chart-line" stroke="${s.color||colors[i%colors.length]}" points="${points}"><title>${esc(s.label||s.key)}</title></polyline>`;
  });
  markers.forEach(m=>{body+=`<circle cx="${x(m.ts)}" cy="${y(m.value)}" r="4" fill="${m.color||colors[1]}" stroke="#10161d"><title>${esc(m.label)}</title></circle>`;});
  for(let i=0;i<3;i++) {const row=rows[Math.floor((rows.length-1)*i/2)];body+=`<text x="${x(row.ts)}" y="${H-12}" text-anchor="${i===0?'start':i===2?'end':'middle'}">${esc(new Date(row.ts).toISOString().slice(5,16).replace('T',' '))}</text>`;}
  return `<svg class="chart" role="img" aria-label="${esc(title)}" viewBox="0 0 ${W} ${H}"><title>${esc(title)}</title>${body}</svg><div class="chart-caption">${rows.length.toLocaleString()} actual observations · UTC · ${esc(title)}</div>`;
}
function dashboard() {
  const hold=example.evaluation.gates.holdout;
  const winner=example.models.find(m=>m.selected);
  view().innerHTML=heading('Robin: a Transformer, end to end','A small artificial price series teaches a model to recognize down, flat and up regimes. Explore the actual retained data, comparison, predictions and promotion evidence.')+
  `<div class="cards">${card('Synthetic minutes',example.data.rows.length.toLocaleString(),'Exact retained source dataset')}${card('Designs compared',example.models.length,'Same split and frozen evaluation gates')}${card('Holdout accuracy',pct(hold.metrics.accuracy),`${hold.n_rows} unseen prediction windows`)}${card('Original promotion',hold.passed?'Passed':'Rejected','Synthetic classification service only')}</div>`+
  '<div class="workflow"><a href="#/data">1 · Data & features</a>→<a href="#/models">2 · Train & compare</a>→<a href="#/backtest">3 · Replay</a>→<a href="#/evaluation">4 · Evaluate & promote</a>→<a href="#/artifacts">5 · Inspect evidence</a></div>'+
  `<div class="card"><h2>Three invented regimes, one causal prediction task</h2>${chart(example.data.rows,[{key:'price'}],{title:'Synthetic price across chronological partitions',shading:true})}<div class="legend split-key">${splitNames().map(s=>`<span class="split-${esc(s)}">${esc(s)}</span>`).join('')}</div><p>The model sees five features available at the decision time, over a 12-minute window, and predicts the next four minutes. Forward outcomes are labels, not model inputs.</p></div>`+
  `<div class="grid cols-2"><div class="card"><h2>What the result demonstrates</h2><p>The selected depth-${esc(winner.parameters.n_layers)} design passed eight fixed holdout checks. Both deliberately weak controls were rejected. This demonstrates that the data, training, evaluation, tracking and serving workflow connects correctly.</p><p class="warn-text">Synthetic replay economics are diagnostic only. The persistence baseline matches holdout accuracy; this is not evidence of real-market trading edge.</p></div><div class="card"><h2>Recorded evidence, local exploration</h2><p>Preparation imports the original records. Reopening this viewer reads that saved state; it does not run training or newly promote a model. MLflow keeps original run IDs as source tags beside its new local IDs.</p><a id="dashboard-mlflow" href="${esc(example.viewer.mlflow_url)}" target="_blank" rel="noopener">Open local MLflow ↗</a>${details('Source run and provenance',example.summary)}</div></div>`;
}
function dataPage() {
  const rows=splitRows(), cols=[['Time (UTC)','ts'],['Price','price'],...example.data.features.map(f=>[f,f])];
  const start=state.page*40;
  view().innerHTML=heading('Data / Features','All 3,000 synthetic minutes are retained. Features are causal; the optional outcome panel contains information that becomes known later.')+
  `<div class="controls"><label>Partition<select id="split-choice"><option value="all">All partitions</option>${splitNames().map(k=>`<option value="${esc(k)}" ${state.split===k?'selected':''}>${esc(k)}</option>`).join('')}</select></label><label>Causal feature<select id="feature-choice">${example.data.features.map(f=>`<option ${state.feature===f?'selected':''}>${esc(f)}</option>`).join('')}</select></label><label class="checkbox"><input id="outcomes-choice" type="checkbox" ${state.outcomes?'checked':''}>Show separate future-outcome panel</label></div>`+
  `<div class="card"><h2>Synthetic price</h2>${chart(rows,[{key:'price'}],{title:'Price known at decision time',shading:true})}</div><div class="card"><h2>${esc(state.feature)}</h2>${chart(rows,[{key:state.feature,color:colors[1]}],{title:`Causal feature: ${state.feature}`,shading:true})}</div>`+
  (state.outcomes?`<div class="card"><h2 class="warn-text">Future outcomes · never model inputs</h2><p>These labels and forward prices describe what happened after the decision. Class 0 = down, 1 = flat, 2 = up.</p>${chart(rows,[{key:'label',color:colors[2]}],{title:'Forward outcome labels (future information)'})}${table([['Decision time','ts'],['Label','label'],['Future price','future_price'],['Outcome available','future_end']],rows,30)}</div>`:'')+
  `<div class="card"><h2>Inspectable observations</h2><p class="muted">${rows.length.toLocaleString()} rows in this selection. Purged boundary rows are omitted when filtering a named partition.</p>${table(cols,rows.slice(start,start+40))}<div class="pagination"><button id="rows-prev" ${state.page===0?'disabled':''}>Previous</button><span>Rows ${rows.length?start+1:0}–${Math.min(start+40,rows.length)} of ${rows.length}</span><button id="rows-next" ${start+40>=rows.length?'disabled':''}>Next</button></div></div>${details('Chronological partitions and label boundaries',example.data.splits)}`;
  $('#split-choice').onchange=e=>{state.split=e.target.value;state.page=0;dataPage();};
  $('#feature-choice').onchange=e=>{state.feature=e.target.value;dataPage();};
  $('#outcomes-choice').onchange=e=>{state.outcomes=e.target.checked;dataPage();};
  $('#rows-prev').onclick=()=>{state.page--;dataPage();};$('#rows-next').onclick=()=>{state.page++;dataPage();};
}
function modelsPage() {
  const model=selection();
  if(!model.selected) state.predictionSplit='validation';
  const predictions=state.predictionSplit==='holdout'?example.predictions.holdout:example.predictions.validation[model.design_id];
  const hold=example.evaluation.gates.holdout;
  view().innerHTML=heading('Model design comparison','Two small Transformer designs share the same data, chronological partitions and training recipe. The winner was selected on validation loss before the locked holdout was consulted.')+
  `<div class="card">${table([['Design',modelName],['Validation accuracy',m=>pct(m.validation.accuracy)],['Validation macro F1',m=>m.validation.macro_f1],['Validation log loss',m=>m.validation.log_loss],['Selected',m=>m.selected?'Yes':'No']],example.models)}</div>`+
  `<div class="controls">${modelSelect()}<label>Prediction population<select id="prediction-split"><option value="validation" ${state.predictionSplit==='validation'?'selected':''}>Validation · 435 rows</option>${model.selected?`<option value="holdout" ${state.predictionSplit==='holdout'?'selected':''}>Locked holdout · ${hold.n_rows} rows</option>`:''}</select></label></div>`+
  `<div class="card"><h2>Actual saved class probabilities</h2><div class="legend"><span class="down">Down</span><span class="flat">Flat</span><span class="up">Up</span></div>${chart(predictions,[{key:'p_down',color:'#ff8787'},{key:'p_flat',color:'#ffc857'},{key:'p_up',color:'#7ee0a3'}],{title:`${state.predictionSplit} probabilities · ${model.design_id}`,range:[0,1]})}${table([['Time','ts'],['Prediction','direction'],['True outcome','direction_true'],['Confidence','confidence']],predictions,40)}</div>`+
  `<div class="grid cols-2"><div class="card"><h2>Exact model parameters</h2><pre>${pretty(model.parameters)}</pre></div><div class="card"><h2>Holdout comparison</h2>${table([['Model/baseline','name'],['Accuracy',r=>pct(r.accuracy)],['Log loss','log_loss']],[{name:'Selected Transformer',...hold.metrics},...Object.entries(hold.baselines).map(([name,m])=>({name,...m}))])}<p class="muted">The persistence rule is a strong baseline at this small synthetic scale. Equal accuracy is not statistical superiority.</p></div></div>`;
  wireModel(modelsPage);$('#prediction-split').onchange=e=>{state.predictionSplit=e.target.value;modelsPage();};
}
function backtestPage() {
  const model=selection(), run=example.backtest[model.design_id], trades=run.trades;
  const span=example.data.splits.validation, rows=example.data.rows.filter(r=>time(r.ts)>=time(span.start)&&time(r.ts)<=time(span.end));
  let cumulative=0;
  const equity=trades.map(t=>({ts:t.exit_ts,equity:(cumulative+=t.net_dollars)}));
  const markers=trades.map(t=>({ts:t.entry_ts,value:t.entry_price,color:t.direction>0?'#7ee0a3':'#ff8787',label:`${t.direction>0?'Long':'Short'} ${t.entry_ts}; net ${fmt(t.net_dollars,2)} synthetic dollars`}));
  view().innerHTML=heading('Synthetic backtest replay','Replay the saved validation trade ledger. These diagnostic economics use invented executable quotes and explicit costs; they are not a financial promotion gate.')+
  `<div class="controls">${modelSelect()}</div><div class="cards">${card('Recorded trades',trades.length,'Saved validation population')}${card('Synthetic net P/L',fmt(run.economics.net_dollars,2),'Dollars under stated cost assumptions')}${card('Commission',fmt(run.economics.total_commission,2),'Already included in net results')}${card('Hit rate',pct(run.economics.hit_rate),'Synthetic ledger only')}</div>`+
  `<div class="card"><h2>Price and recorded entries</h2><div class="legend"><span class="up">Long entry</span><span class="down">Short entry</span></div>${chart(rows,[{key:'price'}],{title:'Validation price with actual ledger entries',markers})}</div><div class="card"><h2>Cumulative synthetic net P/L</h2>${chart(equity,[{key:'equity',color:colors[1]}],{title:'Cumulative saved net_dollars at each exit'})}${details('Execution and cost assumptions',run.economics.assumptions)}</div>`+
  `<div class="card"><h2>Full saved trade ledger</h2>${table([['Trade','trade_id'],['Entry (UTC)','entry_ts'],['Exit (UTC)','exit_ts'],['Side',t=>t.direction>0?'Long':'Short'],['Entry price','entry_price'],['Exit price','exit_price'],['Commission','commission'],['Net synthetic dollars','net_dollars']],trades)}</div>`;
  wireModel(backtestPage);
}
function gateTable(gate) {
  const p=gate.policy,m=gate.metrics;
  const rules={accuracy:[m.accuracy,`≥ ${p.min_accuracy}`],macro_f1:[m.macro_f1,`≥ ${p.min_macro_f1}`],class_recall:[(m.class_recall||[]).map(v=>fmt(v)).join(', '),`each ≥ ${p.min_class_recall}`],maximum_log_loss:[m.log_loss,`≤ ${p.max_log_loss}`],minimum_rows:[gate.n_rows,`≥ ${p.min_rows}`],minimum_class_rows:[(gate.class_counts||[]).join(', '),`each ≥ ${p.min_class_rows}`],prior_log_loss_improvement:[m.prior_log_loss_improvement,`≥ ${p.min_prior_log_loss_improvement}`],persistence_accuracy_noninferiority:[m.persistence_accuracy_difference,`≥ −${p.persistence_accuracy_margin}`]};
  return table([['Check','name'],['Observed','value'],['Frozen threshold','threshold'],['Result','result']],Object.entries(gate.checks).map(([name,passed])=>({name,value:rules[name]?.[0]??'—',threshold:rules[name]?.[1]??'See policy',result:passed?'PASS':'FAIL'})));
}
function evaluationPage() {
  const g=example.evaluation.gates, controls=example.evaluation.negative_controls;
  view().innerHTML=heading('Evaluation / Promotion','The original model legitimately passed the frozen synthetic-service gates. Importing its records into this viewer does not constitute a new evaluation or promotion.')+
  `<div class="cards">${card('Validation',g.validation.passed?'8 / 8 passed':'Rejected',`${g.validation.n_rows} predictions`)}${card('Locked holdout',g.holdout.passed?'8 / 8 passed':'Rejected',`${g.holdout.n_rows} predictions`)}${card('Negative controls',`${Object.values(controls).filter(v=>!v.passed).length} rejected`,'Prior-only and shuffled predictions')}</div>`+
  `<div class="grid cols-2"><div class="card"><h2>Validation gates</h2>${gateTable(g.validation)}</div><div class="card"><h2>Locked holdout gates</h2>${gateTable(g.holdout)}</div></div>`+
  `<div class="card"><h2>Controls must fail</h2>${table([['Control','name'],['Accuracy',r=>pct(r.metrics.accuracy)],['Log loss',r=>r.metrics.log_loss],['Outcome',r=>r.passed?'UNEXPECTED PASS':'Rejected as expected']],Object.entries(controls).map(([name,v])=>({name,...v})))}<p>These controls confirm that merely emitting probabilities or rearranging correct predictions is insufficient to pass the same evaluation rules.</p></div>`+
  `<div class="card"><h2>Recorded promotion receipt</h2>${badge('Original run: synthetic service promoted')} ${badge('This viewer: recorded example',false)}<p>The original test used the <code>test-serving</code> alias and verified loaded-model and HTTP inference. “PRODUCTION” in that receipt refers only to the isolated synthetic classification service. Trading execution is not enabled.</p>${details('Promotion evidence and source hashes',example.evaluation.promotion)}${details('Baselines and gate policies',example.evaluation.baselines||g.holdout.baselines)}</div>`;
}
function artifactsPage() {
  const rows=example.artifacts;
  view().innerHTML=heading('Artifacts & tracking','The fixture is checksummed before it is loaded. Original evidence is preserved; portability projections identify their source hashes.')+
  `<div class="card"><h2>MLflow lineage</h2><p>The source parent run and both model-design runs are imported with new local IDs. Original IDs remain searchable in <code>source_run_id</code> tags. The imported run is explicitly labelled <code>recorded_example</code>.</p><a href="${esc(example.viewer.mlflow_url)}" target="_blank" rel="noopener">Open local MLflow ↗</a>${details('Local import and original run mapping',example.tracking||example.mlflow||{})}</div>`+
  `<div class="card"><h2>${rows.length} verified artifacts</h2><div class="table-wrap"><table><thead><tr><th>Artifact</th><th>Bytes</th><th>Kind</th><th>SHA-256</th></tr></thead><tbody>${rows.map(r=>`<tr><td><a href="/api/showcase/artifact/${r.path.split('/').map(encodeURIComponent).join('/')}">${esc(r.path)}</a></td><td>${esc(fmt(r.size))}</td><td>${esc(r.kind)}</td><td class="mono">${esc(r.sha256)}</td></tr>`).join('')}</tbody></table></div></div>`;
}
function livePage() {
  view().innerHTML=heading('Live platform capabilities','This synthetic example demonstrates the research workflow. These operational applications require their own data and infrastructure.')+
  `<div class="grid cols-2"><div class="card"><h2>Live feed / order book</h2>${badge('Unavailable in this example',false)}<p>The dataset has synthetic minute prices, five features and future labels. It contains no trade prints, volume, OHLC candles or market-depth events. The viewer cannot reconstruct them.</p><button disabled>Connect market feed</button></div><div class="card"><h2>Cloud jobs / trading deployment</h2>${badge('Unavailable in this example',false)}<p>The recorded console has no cloud, vendor, scheduler or broker connections. Use the documented training CLI for a fresh local Transformer run and its own evaluation records.</p><button disabled>Launch operational job</button></div></div>`;
}
const renderers={dashboard,data:dataPage,models:modelsPage,backtest:backtestPage,evaluation:evaluationPage,artifacts:artifactsPage,live:livePage};
function render(){state.tab=location.hash.replace(/^#\//,'')||'dashboard';document.querySelectorAll('[data-nav]').forEach(a=>a.classList.toggle('active',a.dataset.nav===state.tab));(renderers[state.tab]||dashboard)();}
async function boot(){try{const r=await fetch('/api/showcase');if(!r.ok)throw new Error(`HTTP ${r.status}`);example=await r.json();state.model=example.models.find(m=>m.selected).design_id;$('#mlflow-link').href=example.viewer.mlflow_url;window.addEventListener('hashchange',render);render();}catch(e){view().innerHTML=heading('Example unavailable','The retained artifacts could not be loaded.')+`<pre>${esc(e.message)}</pre><p>Run the documented prepare command and inspect its integrity-check output.</p>`;}}
boot();
