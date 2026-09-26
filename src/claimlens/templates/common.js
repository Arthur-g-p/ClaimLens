'use strict';
// Shared by every report template: the embedded documents, the roster island
// the template ships, the readers, and the run-screen building blocks. What
// differs per report type stays in that type's template.
const R = JSON.parse(document.getElementById('record').textContent);
const F = JSON.parse(document.getElementById('findings').textContent);
const NRUNS = R.runs.length;
const ROSTER = (el => el ? JSON.parse(el.textContent) : {groups: [], behavior: [], health: [], directions: {}, universes: {}})(document.getElementById('roster'));
const DIR = ROSTER.directions || {};
const SEP = " || ";
const VC = {Entailment: "ent", Contradiction: "con", Neutral: "neu"};

const arrow = k => { const d = DIR[k]; return d === "higher is better" ? "↑" : d === "lower is better" ? "↓" : d === "depends on goals" ? "↕" : ""; };
const LOWER = k => DIR[k] === "lower is better";
const FLAT = k => /distribution/.test(DIR[k] || "");

// Old reports stored a verdict as a bare string; new ones as {verdict, explanation}.
function V(x){ if(x == null) return {verdict: null}; return typeof x === "string" ? {verdict: x} : x; }
function vcls(v){ return VC[v] || "unj"; }
function vname(v){ return v || "unjudged"; }
function fmt(x, d){ d = (d == null) ? 2 : d; return (x == null || Number.isNaN(x)) ? '<span class="na">n/a</span>' : Number(x).toFixed(d); }
function ct(c){ return typeof c === "string" ? c : (c.subject + " " + c.predicate + " " + c.object); }
function esc(s){ return String(s == null ? "" : s).replace(/[&<>"]/g, m => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[m])); }
function support(run, k){ const s = run.counts && run.counts.support; return (s && k in s) ? s[k] : null; }
function fl(fr, b){ return (fr && fr[b]) || []; }
function findingsOf(ri){ return (F.runs && F.runs[ri] && F.runs[ri].findings) || {}; }
function ctxOf(it, j){ const c = (it.retrieved_context || [])[j]; return typeof c === "string" ? {doc_id: "", text: c} : (c || {doc_id: "", text: ""}); }
function shortChunk(j){ return "c" + (j + 1); }
// Everything that points at a chunk is keyed by position, as the pipeline
// keys its matrices; doc_ids are labels and may repeat.
function isRel(it, j){ return (it.retrieved2answer || []).some(row => V((row || [])[j]).verdict === "Entailment"); }
function positions(it, i, verdict){ const out = []; (((it.retrieved2response || [])[i]) || []).forEach((c, j) => { if(V(c).verdict === verdict) out.push(j); }); return out; }
function metaLine(){ const a = R._args || {}, m = R._meta || {}; return esc(a.extractor_model || "") + " → " + esc(a.checker_model || "") + " · " + m.evaluated_items + " of " + m.total_items + " items · " + NRUNS + " run" + (NRUNS > 1 ? "s" : "") + " · " + esc(m.timestamp || ""); }
function titleFor(type){ const name = String((R._args || {}).input_file || "").split("/").pop(); return "claimlens · " + type + (name ? " · " + name : ""); }

// ── ragcheck findings branches as pills ─────────────────────────────────
// Map from query + SEP + claim text (or query + SEP for item-level branches)
// to the findings branches that list it, so a claim can show why it was flagged.
function findingTags(ri){
  const fr = findingsOf(ri); const m = new Map();
  const add = (branch, key, e) => { if(!m.has(key)) m.set(key, []); m.get(key).push({branch, e}); };
  for(const b of ["hallucination", "noise_sensitivity_in_relevant", "noise_sensitivity_in_irrelevant", "self_knowledge"])
    for(const e of fl(fr, b)) add(b, e.query + SEP + e.claim, e);
  for(const e of fl(fr, "recall_misses")) add("recall_miss", e.query + SEP + e.gt_claim, e);
  for(const b of ["unjustified_abstention", "unwarranted_answer", "extraction_failed"])
    for(const e of fl(fr, b)) add(b, e.query + SEP, e);
  return m;
}
const TAG_LABEL = {hallucination: "hallucination", noise_sensitivity_in_relevant: "noise (relevant)", noise_sensitivity_in_irrelevant: "noise (irrelevant)", self_knowledge: "self-knowledge", recall_miss: "recall miss", unjustified_abstention: "unjustified abstention", unwarranted_answer: "unwarranted answer", extraction_failed: "extraction failed"};
const TAG_CLS = {hallucination: "bad", noise_sensitivity_in_relevant: "bad", noise_sensitivity_in_irrelevant: "bad", self_knowledge: "warn", recall_miss: "warn", unjustified_abstention: "bad", unwarranted_answer: "bad", extraction_failed: "bad"};
// What each branch means, in the record's own terms; the entry fills in the chunks involved.
function tagDesc(t, e){
  const ids = a => (Array.isArray(a) && a.length) ? a.join(", ") : "";
  switch(t){
    case "hallucination": return "Not entailed by the GT answer and not entailed by any retrieved chunk: the response made it up.";
    case "noise_sensitivity_in_relevant": return "Not entailed by the GT answer, but entailed by a relevant chunk (one that entails at least one GT claim)" + (ids(e && e.grounded_by) ? ": " + ids(e.grounded_by) : "") + ". The generator picked up noise from relevant context.";
    case "noise_sensitivity_in_irrelevant": return "Not entailed by the GT answer, entailed only by irrelevant chunks (none of them entails a GT claim)" + (ids(e && e.grounded_by) ? ": " + ids(e.grounded_by) : "") + ". The generator picked up noise from irrelevant context.";
    case "self_knowledge": return "Entailed by the GT answer but by no retrieved chunk: correct without the context, from the model's own knowledge.";
    case "recall_miss": return "A GT claim the response never states. " + (ids(e && e.retrieved_in) ? "It was retrieved (" + ids(e.retrieved_in) + "), so the generator dropped evidence it had." : "No chunk contains it, so the retriever never brought it.");
    case "unjustified_abstention": return "The response declined to answer although a GT answer exists." + (ids(e && e.relevant_chunks) ? " Relevant chunks were retrieved: " + ids(e.relevant_chunks) + "." : "");
    case "unwarranted_answer": return "The response answered a question annotated as unanswerable (empty gt_answer).";
    case "extraction_failed": return "Claim extraction failed for this item (a tooling error); it is excluded from every quality metric.";
    default: return "";
  }
}
function tagHtml(t, e){ return '<span class="tag ' + (TAG_CLS[t] || "") + '" title="' + esc(tagDesc(t, e)) + '">' + esc(TAG_LABEL[t] || t) + '</span>'; }

// ── run screen ──────────────────────────────────────────────────────────
// One dot per run on a 0..1 axis, a bracket for [min, max], a tick for the mean.
function dotsHtml(vr, mean){
  if(!vr || !Array.isArray(vr.values)) return "";
  const pct = x => Math.max(0, Math.min(1, x)) * 100;
  let h = '<div class="dots" title="one dot per run, tick = mean, bracket = [min, max]">';
  if(vr.min != null && vr.max != null) h += '<b style="left:' + pct(vr.min) + '%;width:' + (pct(vr.max) - pct(vr.min)) + '%"></b>';
  for(const v of vr.values) if(v != null) h += '<i style="left:' + pct(v) + '%"></i>';
  if(mean != null) h += '<s style="left:' + pct(mean) + '%"></s>';
  return h + "</div>";
}
// The console's rule: rates over a universe (roster.universes: [numerator,
// denominator, optional subtrahend] in counts.abstention) always show their
// fraction; paper metrics show support only when it is short of the item count.
function supportNote(run, k){
  const u = ROSTER.universes && ROSTER.universes[k];
  if(u){ const ab = (run.counts || {}).abstention || {}; const den = (ab[u[1]] || 0) - (u[2] ? (ab[u[2]] || 0) : 0); return (ab[u[0]] || 0) + " of " + den + " " + u[1]; }
  const rl = (run.counts || {}).reliability || {};
  if(k === "extraction_error_rate" && rl.extraction) return rl.extraction.failed + " of " + rl.extraction.items + " items failed";
  if(k === "checker_failure_rate" && rl.checking) return rl.checking.unjudged + " of " + rl.checking.issued + " verdicts unjudged";
  const s = support(run, k), n = run._meta.evaluated_items;
  return (s == null || s === n) ? "" : s + " of " + n + " items";
}
// A cell is skipped when the source has no such key at all (not computed at
// that level); null is n/a and still shown. vr adds ± std and the run dots;
// run adds the support note from its counts.
// roster.labels (optional) puts a name before the key, for templates whose
// metrics are named by the run (compare under a ground truth).
function metricCell(k, vals, vr, run){
  const m = vals[k]; if(m === undefined) return "";
  const v = vr && vr[k], lab = ROSTER.labels && ROSTER.labels[k];
  return '<div class="cell" title="' + esc(DIR[k] || "") + '"><div class="k">' + (lab ? '<span class="nm">' + esc(lab) + "</span> · " : "") + '<span class="mono">' + k + "</span> " + arrow(k) + "</div>"
    + '<div class="v">' + (m == null ? '<span class="na">n/a</span>' : m.toFixed(3)) + (v && v.std != null ? "<small>± " + v.std.toFixed(3) + "</small>" : "") + "</div>"
    + (m == null ? "" : '<div class="bar"><i class="' + (LOWER(k) ? "bad" : FLAT(k) ? "flat" : "") + '" style="width:' + (m * 100) + '%"></i></div>')
    + (v ? dotsHtml(v, m) : "")
    + '<div class="n">' + (run ? supportNote(run, k) : "") + "</div></div>";
}
function cluster(name, desc, keys, vals, vr, run){
  const cells = keys.map(k => metricCell(k, vals, vr, run)).join("");
  if(!cells) return "";
  if(!name) return '<div class="band">' + cells + "</div>";
  return '<details class="cl" open><summary>' + esc(name) + (desc ? " <span>— " + esc(desc) + "</span>" : "") + '</summary><div class="band">' + cells + "</div></details>";
}
// One structure for every level: vals is a metrics dict (the mean over runs,
// one run, or one item), vr its variance when it is a mean, run the run whose
// counts annotate the cells. Clusters collapse on their own.
function metricsBlock(title, vals, vr, run, open){
  let h = "";
  for(const [name, desc, keys] of ROSTER.groups) h += cluster(name, desc, keys, vals, vr, run);
  if(ROSTER.behavior && ROSTER.behavior.length) h += cluster("Abstention Behavior", null, ROSTER.behavior, vals, vr, run);
  if(ROSTER.health && ROSTER.health.length) h += cluster("Reliability", "tooling, excluded from all metrics", ROSTER.health, vals, vr, run);
  return h ? '<details class="mb"' + (open ? " open" : "") + '><summary>' + esc(title) + "</summary>" + h + "</details>" : "";
}
function itemRows(items, cfg){
  let h = "";
  items.forEach((it, i) => {
    const dots = it.is_abstention ? '<span class="tag warn" title="' + esc(cfg.abstainTitle || "The response declined to answer.") + '">abstained</span>'
      : (it.response_claims.length ? it.response_claims.map((_, ci) => '<i class="' + cfg.dot(it, ci) + '"></i>').join("") : '<span class="na">no claims</span>');
    h += '<div class="row" data-i="' + i + '"><span class="muted">#' + (i + 1) + '</span><span class="q">' + esc(it.query) + '</span><span class="dots2">' + dots + '</span><span class="f">' + cfg.right(it) + "</span></div>"; });
  return h;
}
// Over N runs the final result comes first, mean ± std with one dot per run,
// then each run's own block; one run shows its macro block. Every report's
// run screen opens with it.
function runMetrics(run){
  return NRUNS > 1
    ? metricsBlock("Metrics · mean ± std over " + NRUNS + " runs", R.metrics, R.variance, null, true)
      + R.runs.map((r, i) => metricsBlock("Run " + (i + 1) + (r._meta && r._meta.duration_seconds != null ? " · " + r._meta.duration_seconds + "s" : ""), r.metrics || {}, null, r, false)).join("")
    : metricsBlock("Metrics · macro over " + run._meta.evaluated_items + " items", run.metrics || {}, null, run, true);
}
function runScreen(el, run, cfg){
  const runIdx = R.runs.indexOf(run);
  let h = runMetrics(run);
  h += '<div class="items"><h2><span class="hd">Items</span><span>· ' + (NRUNS > 1 ? "run " + (runIdx + 1) + " · " : "") + "one dot per response claim · " + esc(cfg.dotLegend || "") + "</span></h2>" + itemRows(run.items, cfg) + "</div>";
  el.innerHTML = h;
  el.onclick = e => { const r = e.target.closest(".row"); if(r) cfg.onOpen(+r.dataset.i); };
}
function runSeg(el, runIdx){ el.innerHTML = NRUNS > 1 ? R.runs.map((_, i) => '<button class="' + (i === runIdx ? "on" : "") + '" data-r="' + i + '">Run ' + (i + 1) + "</button>").join("") : ""; }
function navHtml(){ return '<button class="btn" id="back">‹ run</button><button class="btn" id="prev" title="previous item">‹</button><button class="btn" id="next" title="next item">›</button>'; }

// ── tooltip ─────────────────────────────────────────────────────────────
function placeTip(x, y){ const tip = document.getElementById("tip"); const w = tip.offsetWidth, hh = tip.offsetHeight; let L = x + 14, T = y + 14; if(L + w > innerWidth - 8) L = x - w - 14; if(T + hh > innerHeight - 8) T = Math.max(8, y - hh - 14); tip.style.left = L + "px"; tip.style.top = T + "px"; }
function showTip(html, x, y){ const tip = document.getElementById("tip"); tip.innerHTML = html; tip.hidden = false; placeTip(x, y); }
function hideTip(){ const tip = document.getElementById("tip"); if(tip) tip.hidden = true; }
function tipHidden(){ const tip = document.getElementById("tip"); return !tip || tip.hidden; }
