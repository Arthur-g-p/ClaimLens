'use strict';
// Shared by every report template: the two embedded documents and the
// helpers that read them. Anything specific to one report type lives in
// that type's template.
const R = JSON.parse(document.getElementById('record').textContent);
const F = JSON.parse(document.getElementById('findings').textContent);
const NRUNS = R.runs.length;
const SEP = " || ";
const VC = {Entailment: "ent", Contradiction: "con", Neutral: "neu"};

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

function ctxOf(it, j){ const c = (it.retrieved_context || [])[j]; return typeof c === "string" ? {doc_id: "", text: c} : (c || {doc_id: "", text: ""}); }
function shortChunk(j){ return "c" + (j + 1); }
function isRel(it, j){ const c = ctxOf(it, j); return !!(c.doc_id && it.relevant_chunks && it.relevant_chunks.includes(c.doc_id)); }

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
function metaLine(){ const a = R._args || {}, m = R._meta || {}; return esc(a.extractor_model || "") + " → " + esc(a.checker_model || "") + " · " + m.evaluated_items + " of " + m.total_items + " items · " + NRUNS + " run" + (NRUNS > 1 ? "s" : "") + " · " + esc(m.timestamp || ""); }
