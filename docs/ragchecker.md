# RagChecker Pipeline

`claimlens ragcheck` — full RAGChecker-style RAG evaluation:
2 extractions + 4 checking directions + 11 metrics, one self-contained
report file. Implements the methodology of RAGChecker (Ru et al., 2024)
with modernized data contracts, explicit error handling, and additional
reliability metrics.

## Input

A JSON list of items, or the original RAGChecker envelope
`{"results": [...]}` (accepted at the boundary, never emitted).

Required per item (hard drop when missing: absent or `null` for the two
strings, absent or empty for the chunk list, because nothing can be
evaluated against no context):

| Key | Meaning |
| --- | --- |
| `response` | The RAG system's answer under evaluation. `""` is data, a full abstention, never missing. |
| `gt_answer` | The ground truth answer. `""` is data, an unanswerable question, never missing. |
| `retrieved_context` | Chunks: `[{doc_id, text}]` or bare strings (ids synthesized as `000`, `001`, ...). Empty = missing. |

Optional: `query_id` (falls back to `id`), `query` (canonicalized to
`question` internally, emitted back as `query`).

Items without ground truth belong in the faithfulness pipeline
(docs/faithfulness.md), not in a degraded ragcheck.

**Unanswerable questions: the blank-GT convention.** Some questions have
no answer. The annotator says so with an explicit empty string:

> **`"gt_answer": ""` means no answer exists.** Absent or `null` means the
> ground truth is missing and the item is dropped. Present but empty is a
> deliberate annotation: the question is unanswerable.

This comes from the annotation, never from the tool. SQuAD 2.0, Natural
Questions and MS MARCO ("No Answer Present.", the source of this
project's eval data) all mark unanswerable questions the same way. A
ground truth that is present but yields no claims after extraction is
ambiguous (did the ground truth abstain, or did extraction fail?) and is
reported as `uncategorized`. What the empty string makes measurable,
justified silence and unwarranted answers, is explained under
[Abstention](#abstention).

## What runs

```
response  --extract-->  {ext}_response_kg
gt_answer --extract-->  {ext}_gt_answer_kg     (mark_abstention=False:
                                                an empty GT extraction is a
                                                data-quality signal, not a
                                                response abstention)

answer2response      response claims vs gt_answer          (flat)
response2answer      gt claims       vs response           (flat)
retrieved2response   response claims vs each chunk         (matrix)
retrieved2answer     gt claims       vs each chunk         (matrix)
```

Each direction gets its own CheckingService with a direction-specific
verdict namespace (`{checker}_answer2response_verdict`, ...) so verdicts
over the same triplets never collide. Matrix directions run one joint
check per (item, chunk) and fold back `{chunk_index: verdict}` dicts —
keyed by position, because a corpus chunked from one document repeats the
same doc_id across its chunks.

There is no item-level skipping in v1 (planned for 2.0 together with
report re-ingestion for manually corrected ground truth). A run that dies
part-way is re-run from the start.

## Output: record and findings

The CLI writes two files from `pipeline.last_report` and
`pipeline.last_findings` (default `results/{input_stem}_ragcheck[_{runs}].json`
and its `_findings.json` sibling). One skeleton, shared with `faithcheck`,
`eval checker` and `eval extractor`:

```
record    {_args, _meta, metrics, variance, runs: [{_meta, metrics, counts, items}]}
findings  {_args, _meta, runs: [{_meta, findings}]}
```

`runs` is a list at `--runs 1` too: `metrics` holds the mean over runs (the
run's own values at N = 1), `variance` the spread, `runs` one complete entry
per run. `--runs N` adds entries and reshapes nothing.

- `metrics` — the 11 paper metrics plus `unjustified_abstention_rate`,
  `refused_with_relevant_chunks_rate`, `justified_abstention_rate`,
  `unwarranted_answer_rate`, `extraction_error_rate`, `checker_failure_rate`:
  exactly the variance roster. Rates live here and nowhere else.
- `counts` — every number the console prints: `support` (items behind each
  macro average), `pipeline` (requests and tallies per phase, failure
  numbers when something failed), `abstention` (the ⚪ tree incl. the cause
  split), `reliability` (the 💥 rows: extraction failed / items / by side,
  verdicts unjudged / issued).
- `items` — the RAGChecker structure per item: `{subject, predicate, object}`
  claims, four directional arrays parallel to the claims with verdict
  objects `{"verdict", "explanation"}` (plus a sparse `error` cause),
  explicit `is_abstention` and `gt_no_answer`, `relevant_chunks` (the
  doc_ids that entail at least one gt claim — exposed so consumers never
  re-derive it from the matrix), sparse `extraction_errors`, per-item
  `metrics`.
- `findings` — the review queue, one list per branch: the claims counted by
  `hallucination`, `noise_sensitivity_in_relevant`,
  `noise_sensitivity_in_irrelevant` and `self_knowledge` (each with the GT
  verdict, the checker's explanation and, for noise, `grounded_by`),
  `recall_misses` (GT claims the response never states; `retrieved_in`
  non-empty = the generator dropped retrieved evidence, empty = the
  retriever never brought it), `unjustified_abstention`,
  `unwarranted_answer`, `extraction_failed`, `unjudged`. Empty branches stay
  present.

The CLI also writes `{report_stem}.html` unless `--no-html` is given: a
self-contained viewer rendered from the record and the findings by
`claimlens.viewer.render_html`. It shows nothing the two JSON files do not
contain — the run's metrics with their spread, then per item the GT claims,
chunks and response claims with every verdict as a line between them. The
template lives in `src/claimlens/templates/ragcheck.html` and declares the
`schema_version` it reads; a schema bump fails the viewer test until the
template follows.

Consumers reading both old paper outputs and these reports need one
canonicalization rule: string verdict entry = old format, object = new;
missing field = "information not provided", never a crash.

## The 11 paper metrics

Formulas follow the original RAGChecker; they are anchored by a unit test
reproducing the original implementation's reference output value-for-value
(`tests/unit/test_ragchecker.py::TestMetricsReference`).

Notation: R = response claims, G = gt claims, C = chunks. "Correct" =
entailed by gt_answer (`answer2response`). "Grounded" = entailed by at
least one chunk.

| Metric | Definition |
| --- | --- |
| precision | correct response claims / R |
| recall | gt claims entailed by response / G |
| f1 | harmonic mean of the two |
| claim_recall | gt claims grounded in any chunk / G (retriever) |
| context_precision | chunks entailing >=1 gt claim / C (retriever) |
| faithfulness | grounded response claims / R |
| hallucination | ungrounded AND incorrect response claims / R |
| self_knowledge | ungrounded AND correct response claims / R |
| context_utilization | gt claims grounded in chunks AND entailed by response / gt claims grounded in chunks (paper definition, GT-side) |
| noise_sensitivity_in_relevant | incorrect response claims entailed by a relevant chunk / R |
| noise_sensitivity_in_irrelevant | incorrect response claims entailed only by irrelevant chunks / R |

Identity (enforced by a shared denominator and asserted in tests):
`faithfulness = 1 - hallucination - self_knowledge`.

### Semantics that make the numbers trustworthy

- **Only judged data enters a metric.** Every claim and every item ends
  up in one of three states: judged correct, judged wrong, or not judged
  at all. Correct and wrong both count. Not judged is left out of the
  metric and reported separately as a failure rate, so a tooling failure
  can never look like a system failure. The rules below all follow from
  this.
- **None-verdict propagation.** A failed check is *unknown*, never "not
  entailed". Unknown claims leave numerator AND denominator, so checker
  failures cannot inflate hallucination. In matrix rows, known cells decide
  when they can: one Entailment makes a claim grounded regardless of unknown
  cells; no Entailment plus an unknown cell makes the claim undecidable and
  excluded.
- **Zero denominators are `null`, never `0.0`.** "Not computable" is not a
  score.
- **Gating.** Extraction error (either side) → the item is fully excluded:
  every metric `null`, counted in the error rate instead. Abstention → the
  generator family is `null`, but the retrieval metrics (claim_recall,
  context_precision) are still computed — `retrieved2answer` does not
  involve the response, and an abstention says nothing about the retriever.
- **Aggregation is macro** (per paper): per-item metrics averaged over
  contributing items, nulls skipped. `counts.support` reports how
  many items actually contributed to each average — exclusions shrink N
  invisibly otherwise. Micro aggregation is recomputable from the report
  (all verdict arrays are preserved) without any LLM calls.

## Abstention

An abstention is a response from which no claims are extracted, for
example "I don't know". Anything short of that is an ordinary answer and
is scored claim by claim. Refusals that also state facts are an open
problem, see [Known methodology limitations](#known-methodology-limitations).
A partial answer gets partial recall.
A hedged estimate ("the context suggests 600 to 650 ly") is one hedged
claim and is checked like any other. Detecting an abstention is the
extractor's job (docs/extractor.md, Bucket 2). This section is about what
an abstention means and costs once it has been detected.

### Was the abstention right?

Two facts decide that. The ground truth says whether an answer was
expected at all. The retrieval verdicts (`retrieved2answer`, summarized
in `claim_recall`) say whether the retriever delivered the evidence. The
second fact only matters when the first says an answer was expected:

```
abstained
├─ GT ""  (no answer exists)           → justified abstention
└─ GT present (an answer was expected) → unjustified abstention, recall 0
     ├─ refused with relevant chunks     (claim_recall > 0)   generator fault
     ├─ refused without relevant chunks  (claim_recall == 0)  retriever fault
     └─ refused, relevant chunks unknown (claim_recall None)  GT extracted to zero claims, or retrieval unjudged
answered + GT ""                       → unwarranted answer, precision 0
```

### Attribution

If the ground truth is present, an answer was expected, and every
abstention counts against the system. That is `unjustified_abstention_rate`
under Overall. It is a system number, not a generator number, because the
generator cannot be blamed for a claim it never received. To find the
generator's part, the refusals are split by what the retriever delivered:

```
unjustified abstentions = refused with relevant chunks
                        + refused without relevant chunks
                        + refused, relevant chunks unknown
```

Only the first part is the generator's fault: the evidence was retrieved
and the model still refused. That is `refused_with_relevant_chunks_rate`
under Generator. The second part is the retriever's fault and is already
scored by `claim_recall`. The third part cannot be attributed to anyone.
Both rates use the same denominator, the answerable items, so the
difference between the Overall row and the Generator row is exactly the
refusals the generator is not charged for.

On the unanswerable side, `justified_abstention_rate` (abstained, correct)
and `unwarranted_answer_rate` (answered anyway) add up to one.

All four rates print as fractions with their universe, for example
`1 of 8 answerable`. The denominator comes from the annotation (minus any
items where extraction failed), so under `--runs` only the numerator
moves.

The extractor eval has its own view of abstention (docs/eval_extractor.md,
Step 3). Faithcheck cannot judge abstentions (docs/faithfulness.md). The
checker eval has no abstention concept. It compares verdicts to labels,
claim by claim.

### What an abstention costs

- **Recall.** A refusal delivers nothing, so recall is 0 and F1 follows.
  This needs no special rule. Recall is judged from the response text
  (`response2answer`), and a refusal entails no ground truth claim. When
  the ground truth is `""` there are no claims to miss, so justified
  silence costs no recall. Abstentions are never dropped from recall.
  Dropping them would let the model pick its battles.
- **Precision.** `null` for a refusal (no claims to judge). 0 for an
  unwarranted answer (claims, but no reference to check them against).
- **Generator metrics.** `null` for an abstained item. Zero claims means
  0/0, which is undefined, not a score.
- **Retrieval metrics.** Still computed. `retrieved2answer` does not
  involve the response, and an abstention says nothing about the
  retriever.
- **The abstained count.** Distribution information, never a quality
  score. It appears as counts in the Abstention Behavior block, not as a
  rate.

The metrics stay standard. What is new is that an abstention is visible
and attributed: a labeled outcome instead of an anonymous recall of zero,
split into justified, unjustified and unwarranted, with the unjustified
ones further split by whose fault they were. Where a metric is undefined
the report says `null`.

## New metrics (beyond the paper)

| Metric | Definition | Why it exists |
| --- | --- | --- |
| *abstained* (count, `counts.abstention`) | abstained items among the evaluated, extraction-errored excluded. Distribution information, never a quality score — printed as the ⚪ tree's counts, not as a rate. | The paper punishes "I don't know" as a wrong answer, tanking generator metrics. Here abstentions are excluded from the generator family — which would be gameable (abstain on everything, look perfect) unless the refusals themselves are visible and attributed. |
| `justified_abstention_rate` | abstained / **unanswerable** items (those annotated `"gt_answer": ""` — no answer exists) | Correct silence, judged by the annotation (SQuAD 2.0's NoAns). |
| `unjustified_abstention_rate` | abstained / **answerable** items (a GT answer is present) | An answer was expected and the system refused — charged in recall. A **system** outcome, printed under Overall: it includes refusals the generator could not avoid. The *cause* is apportioned by retrieval evidence (`counts.abstention`: `relevant_chunk_present` / `all_chunks_irrelevant` / `relevance_unknown`), computed for free from `retrieved2answer`, which runs for abstained items regardless. The cause never softens the verdict. |
| `unwarranted_answer_rate` | answered / **unanswerable** items | Answered where no answer exists — the failure mode of systems that never shut up. Precision is 0 by necessity (there is no reference to check the claims against; no request is spent). |
| `refused_with_relevant_chunks_rate` | refusals with a chunk that entails a GT claim / **answerable** items | The generator's share of the unjustified rate — the evidence was retrieved and it declined anyway. Printed under Generator, lower is better. Same denominator as the unjustified rate, so the two rows differ by exactly the refusals the generator is *not* charged for: no relevant chunk (retriever fault, `claim_recall` already scores it) or relevance unknown. Those stay counts in the ⚪ tree — see [Abstention](#abstention). |
| `extraction_error_rate` (+ per-side counts) | items with tooling failures / evaluated | The report is honest about its own tooling. These items are excluded from every quality metric — our parse failure must never masquerade as the evaluated system's abstention or hallucination. |
| `checker_failure_rate` | None verdicts / all issued checks | Checker reliability per run: a run with 4% failed judgments deserves less trust than one with 0.1%. Catches loud failures (parse/context); silent checker degradation (all-Entailment bias) is a known open problem — see the drawbacks backlog. |

Behavior-rate denominators exclude extraction-errored items (no-results
leave the denominator): a tooling failure is charged exactly once, in
`extraction_error_rate` — never by diluting a behavior rate.

## Repeated runs: `--runs N` (variance measurement)

Single-run numbers are point samples of a noisy process — a 25% vs 30% F1
across two runs decides nothing. `--runs N` (also on `faithcheck`,
`eval extractor`, `eval checker`) repeats the whole experiment N times and
reports `mean ± std [min, max]` per metric; the [min, max] band shows where
the results "stuck". N=3 is the floor, N=5 when a decision rides on it, and
the cost is N x the LLM bill. A `± 0.000` result against a deterministic
endpoint is a real finding, not a bug.

The feature belongs to the pipeline/evaluator (`runs` constructor
parameter) — the CLI only passes the number through. Mechanics:

- Every run starts from a pristine deep copy of the input, snapshotted
  BEFORE run 1 (run 1 mutates in place per the run() contract; a later
  copy of the mutated data would carry run 1's claims and verdicts, the
  skip logic would no-op runs 2..N, and the variance would read a fake
  zero).
- All N runs print symmetrically: a Run n/N header, progress bars, one
  summary line with duration. The full phase narrative belongs to
  `--runs 1` only; validation + config print once (they are run-invariant).
  One VARIANCE block and one cumulative token table land at the end.

The multi-run file is the single-run file with more entries:

- `_meta.runs` = N and `duration_seconds` the wall-clock total, usage summed;
- `metrics` holds the **means** — a variance-unaware reader parses a
  multi-run record exactly like a single-run one;
- `variance` per metric `{n, std, min, max, values}` — `n` says how many
  runs contributed, a metric that was `null` in every run keeps its key
  with `n: 0`;
- `runs` holds N complete entries `{_meta, metrics, counts, items}`; the
  findings document mirrors them one to one.

## Known methodology limitations

Beyond the metric definitions themselves: claim granularity acts as an
invisible denominator (duplicate facts double-weight recall; extraction
variance masquerades as metric variance), decontextualized triplets can be
unanswerable in isolation, context_precision is partly a chunking artifact,
and precision cannot distinguish hallucinated from true-but-not-in-GT
claims. These are properties of the RAGChecker methodology; the toolkit's
orthogonal axes (atomicity, duplicates, error rates) exist to expose rather
than blend them.

**The generator/retriever split of abstentions** depends on the checker's
per chunk verdicts, and its known weaknesses all push the same way. They
lower `claim_recall`, which moves a refusal from the generator's column to
the retriever's. A ground truth claim that needs two chunks combined shows
no Entailment in any single chunk. A checker that reads too strictly
produces false Neutrals. Both excuse the generator. Read
`refused_with_relevant_chunks_rate` under `--runs`, never from a single
run.

### Abstention handling in verbose systems (unsolved)

Detection rests on the extractor producing no claims. That holds for short
refusals and for long ones that only say the context is insufficient. It
does not hold for refusals that also state facts, which talkative systems
produce all the time:

| case | response | claims extracted | detected as abstention | how it is scored |
| --- | --- | --- | --- | --- |
| simple abstention | "I don't know." | none | yes | justified or unjustified by the gt, generator metrics `null` |
| partial abstention | "I can't say what the surface is made of, but Kepler-22b is about 600 ly away." | the distance | no | as an answer. The claim is not in the gt, so precision 0 and recall 0. It is either grounded in a chunk (noise sensitivity) or not (hallucination). The refusal is invisible. |
| verbose abstention | "The passages mention Kepler-22b's distance and its constellation, but not its surface composition." | none, or two facts about the passages, depending on the extractor | sometimes | as the simple case when no claims come out, as the partial case when they do |
| no abstention | "Kepler-22b's surface composition is unknown. It may be rocky or an ocean world." | two | no | normally |

The second and third rows are the problem. A helpful refusal is scored as
a wrong answer, its refusal never reaches the Abstention Behavior block,
and its volunteered facts land in noise sensitivity or hallucination.

This could be tackled with abstention detection as its own step: an
explicit refusal flag from the extractor, independent of the claim list.
But that cuts against the method. This is fine grained hallucination
detection: every verdict is about one claim. An item level refusal flag
that drops the claims in a flagged response throws that granularity away.
In the third row, "the passages mention Kepler-22b's distance and its
constellation" would be lost. Whether volunteered facts should be
punished, ignored or only flagged is not decided. Until it is, read
`noise_sensitivity` and `hallucination` with this in mind when the system
under test is a talkative one.
