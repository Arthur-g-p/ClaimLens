# Extractor Evaluator (`ExtractorEvaluator`)

The Extractor Evaluator measures **extraction quality**: it runs extraction live
on a labelled dataset, then reconciles the predicted triplets against the
ground-truth (GT) triplets in both directions using an LLM checker.

There is no gold extraction to compare against — the GT triplets in
`eval_data/msmarco/` were human-*verdicted*, never proven complete (see
[eval_data/msmarco/README.md](../eval_data/msmarco/README.md)). So this is
reconciliation, not scoring against a key: each side has to find its
counterpart in the other.

> **Qualify your checker first.** Every number below is produced by an LLM
> checker making entailment judgements. A checker that fails to parse, or that
> collapses to answering "Entailment" for everything, produces extractor scores
> that are partial or meaningless. Run [`eval checker`](eval_checker.md) on the
> same model before trusting `eval extractor` output, and read the
> `checker_failures` block in every report.

## The matcher at a glance

The "checker" of this eval is a matcher: it compares claims with claims. Read
its prompt before reading any number.

| | |
| --- | --- |
| prompt | `claim_match_prompt` in the prompt map, `src/claimlens/prompt_map.json`. Its twin `claim_match_prompt_plain` is sent only to a model that accepts no `response_format` |
| both directions | the **same** prompt for pass 1 and pass 2 |
| rule | one-to-one: one claim of the other side must make the claim true on its own; several taken together never count. The explanation names that claim |
| roles in the prompt | none: the prompt speaks of "one triplet (the Claim)" and "a list of other triplets (the Reference)", since the same words serve both passes |
| one call | set by `--match-mode` (see [The three modes](#the-three-modes)). `joint` (default): up to `--joint-num` claims (default 10) of one item against one reference claim. `strict`: one claim against one reference claim. `many-to-many`: up to `--joint-num` claims against the whole claim list of the other side. A bundle shrinks only when a call would pass the word budget (`DEFAULT_MAX_WORDS` 6,000 × 0.75) |
| what the model reads | every triplet flattened to `subject predicate object`; the claims numbered `[1]`, `[2]`, …; the reference claims labelled with their real number in the other side's list, `(R1)`, `(R2)`, …, also when a call holds only one: the third is `(R3)` |
| never sees | the response and the question |
| answer per claim | an explanation, then `Entailment`, `Neutral` or `Contradiction` |
| verdicts per item | `joint` and `strict`: one per pair (claim, reference claim), folded to one per claim. `many-to-many`: one per GT claim (pass 1) plus one per predicted claim (pass 2) |

---

## Pipeline Overview

```
Input (list[dict])
  │
  ├─ _canonicalize_keys() →  Normalizes key aliases in-place ('context' → 'reference')
  ├─ _validate()   →  Drops items without a response. Canonicalizes GT triplets.
  ├─ extract       →  Runs `ExtractionService` (quiet) → predicted triplets
  ├─ 2b atomicity  →  OPTIONAL orthogonal axis, measured on a deep copy
  ├─ 2c duplicates →  Orthogonal axis, read-only
  ├─ _classify()   →  Five buckets: to_compare, abstention_recognized,
  │                   answer_missed, abstention_misread, extraction_error
  ├─ Match (LLM)   →  2-pass entailment check over the `to_compare` items:
  │                   _match_all_llm() builds synthetic items per pass
  │                   (_build_pass_items) and runs `CheckingService` twice;
  │                   in joint/strict it folds the pairs (_fold_pairs)
  └─ Metrics       →  Precision / Recall / F1 + orthogonal axes + tooling rates
```

Three families of number come out, and they are never mixed:

| family | what it measures | examples |
| --- | --- | --- |
| **Coverage** | how well the extractor found the facts | precision, recall, f1 |
| **Orthogonal axes** | independent properties of the extraction | atomicity, duplicates, abstention behavior |
| **Reliability** | how well *this evaluator* worked | extraction errors, checker failures, atomization failures |

Tooling failures are excluded from every quality metric. Our parse failure must
never be reported as the extractor's mistake.

## Step 1: Validation (`_validate`)

- **Keeps** items with a `response` and a GT key. Absent or null GT key
  = missing data (dropped, reported); an explicit empty list is data —
  nothing to extract: the response abstained, and that is what makes a
  misread abstention detectable.
- **Drops** items without a `response`; there is nothing to extract from.
- **Drops** items marked by `--question-if` (`"needs_question": true` by
  default) that have no `question`: they cannot get the question prompt.
  Counted as `dropped_no_question` in `counts.data` and on its own 📂 Data
  line. Every item extracted with its question carries a sparse
  `"question_given": true` in the record's items, and ⚙️ Config prints how
  many there are.
- **Canonicalizes** key aliases (`context` → `reference`) and GT triplets
  (`{"triplet": [s, p, o]}` → `{"subject", "predicate", "object"}`).
- **Fails fast** (`InvalidInputError`) if no item survives, or if *no* item
  carries the `--gt-key` at all — that means the wrong file or the wrong key,
  and it is caught before a single LLM call is paid for.

GT presence is judged on the **key**, not its contents: an item carrying
`"claude2_response_kg": []` is a deliberate abstention trap, not missing data.

## Step 2: Extraction

Delegates to `ExtractionService` with the configured `--extractor-model`,
running quiet (the evaluator owns the logging). Predictions are always written
under `{extractor_model}_response_kg` — see the collision guard below.

Deduplication is **off**: the eval measures duplicates as an independent axis,
so predictions must stay raw. The eval never runs on deduplicated data.

## Step 2b: Atomicity axis (optional, orthogonal)

Runs the atomizer over the predictions to ask how *atomic* they are: does one
triplet carry exactly one fact? Measured on a **deep copy**, so coverage
matching and the output file always see the raw predictions.

Requires `--atomizer-model` **and** `ATOMIZER_API_KEY`. If either is missing the
axis is skipped and the console says why (`⏭️ Atomicity skipped (...)`); the
`atomicity` field is then `null`.

- `atomicity_rate` — kept claims / evaluated claims (1.0 = every triplet atomic)
- `information_density` — atomic units / evaluated claims (facts per triplet)
- Full split detail rides on its item in the record (`atomicity_splits`).

## Step 2c: Duplicate axis (orthogonal, read-only)

Counts exact (string-equal) duplicate claims in the predictions. Never mutates
anything. Reported as `duplicate_rate` plus the offending triplets per item.

## Step 3: Classification (`_classify`)

After extraction, every item lands in exactly one of **five** buckets. The
abstention is always the *response's* (GT has no claims because the response
asserted nothing); the extractor never abstains — it either *recognizes* an
abstaining response or *misreads* it into invented claims. This differs from
`ragcheck`, whose abstention judges are about the question.

Abstention detection is a quality of the extractor, until detection
becomes its own pipeline step. The three rates in the Abstention Handling
block measure it. `abstention recognized`: a refusal yielded no claims,
correct. `abstention misread`: a refusal yielded invented claims, charged
in precision. `answer missed`: a real answer yielded no claims, charged in
recall. This is independent of ragcheck's abstention handling, which asks
a different question. Not whether the extractor detected the refusal, but
whether the system should have refused.

| bucket | GT | predictions | effect |
| --- | --- | --- | --- |
| `to_compare` | yes | yes | sent to LLM matching |
| `abstention_recognized` | no | no | abstaining response, nothing extracted — correct, no penalty |
| `answer_missed` | yes | no | answering response, nothing extracted — every GT claim charged as a recall miss |
| `abstention_misread` | no | yes | abstaining response, claims invented — every predicted claim charged as a precision miss |
| `extraction_error` | any | — | tooling failure → **excluded from all metrics**, counted in `extraction_errors` |

The `extraction_error` bucket exists so a parse failure can never masquerade as
an abstention. An empty prediction list after a crashed extraction says nothing
about the extractor's willingness to answer.

## Step 4: Matching (`_match_all_llm`)

For `to_compare` items, two independent passes through `CheckingService`, with
the `claim_match_prompt` prompt in every match mode (the service's call with
numbered claims; its single-claim call would send `checker_prompt`):

1. **Pass 1 — GT → Pred (recall).** GT triplets are the claims, predictions are
   the reference. Entailment = the extractor found this fact.
2. **Pass 2 — Pred → GT (precision).** Roles reversed. Entailment = this
   prediction is supported by ground truth.

| pass | each claim of | judged against | question | counts for |
| --- | --- | --- | --- | --- |
| 1 | the GT | the predictions of the item | does one prediction make this GT claim true? | recall |
| 2 | the predictions | the GT claims of the item | does one GT claim make this prediction true? | precision |

Each pass is its own `CheckingService` run. `_build_pass_items` turns every
compared item into synthetic items for it: the claims side goes under a
placeholder extraction key (`_exteval_response_kg`), the other side becomes the
reference text, one `(R1)`, `(R2)`, … line per claim. In `joint` and `strict`
there is one synthetic item per (item, reference claim), with that one line as
its reference; in `many-to-many` one per compared item, with the whole list as
its reference. In pass 2 the two sides swap. This is reshaping only: nothing is
extracted again, and no call is spent on it.

Two passes, two independent counts — following RAGChecker, there is **no shared
TP and no `min()`**. Each pass counts its own hits, and the two counts need not
agree. Example: the GT holds three claims A, B and C; the extractor wrote one
claim "A, B and C". Pass 1 finds each of A, B and C covered, since the one
prediction says each of them (recall 3 / 3); pass 2 finds no GT claim that says
all of "A, B and C" on its own (precision 0 / 1). A shared TP would count matched
pairs once for both ratios; a `min()` would take the smaller of the two hit
counts. Neither is done.

**One triplet must be enough.** A claim counts as found only when one triplet of
the other side makes it true on its own; several triplets taken together never
count, and the explanation names the triplet. So a claim that joins facts the
other side states separately, or one that needs a neighbouring triplet to be
read ("the league that season"), comes back as a miss. A claim that says less
than one triplet is still found: "kills ants" in "kills ants, roaches and fleas".
In `joint` and `strict` the matcher sees one reference claim per call, so it
cannot combine claims.

### The three modes

`--match-mode` decides what one matcher call holds. Words: **the claim** is the
claim being judged ("the Claim" in the prompt), **the reference claim** the claim
of the other side that has to make it true ("the Reference"). For one item and
one pass, with m claims to judge and n reference claims:

| mode | one call holds | calls |
| --- | --- | --- |
| `joint` (default) | up to `--joint-num` claims (default 10), each judged against the same one reference claim | n × ⌈m / joint-num⌉ |
| `strict` | one claim against one reference claim | m × n |
| `many-to-many` | up to `--joint-num` claims against the whole list of reference claims | ⌈m / joint-num⌉ |

A run costs the sum over its compared items and both passes: in pass 1 m is the
item's GT claims and n its predictions, in pass 2 the other way round. `strict`
costs 2 × m × n calls per item. Retry rounds add requests, not calls: the
console's `tasks sent to LLM` is the formula's number, `HTTP requests` adds the
retries.

- **In `joint` the bundle is always on the side of the claims being judged**, in
  both passes; the reference side always holds one claim. Several reference
  claims in one call are what lets the matcher combine them.
- **`strict` and `joint` ask the same pairs.** `joint` is the cheaper way to ask
  them: the instructions are sent once per call, not once per pair.
- **`many-to-many` is not one-to-one.** The matcher sees the whole list of
  reference claims, so only the prompt keeps it from combining several claims
  or from reading a claim with its neighbour. How much that changes the verdicts
  is not measured yet.

### The fold (`joint`, `strict`)

Every pair (claim, reference claim) gets its own verdict. The claim's verdict is
folded from its pairs, as ragcheck folds a claim's verdicts over its chunks, with
`Contradiction` kept apart from `Neutral`:

| the pairs of one claim | the claim's verdict |
| --- | --- |
| at least one pair is Entailment | `Entailment`: found, whatever the other pairs say, also when some pair has no verdict |
| no Entailment, every pair has a verdict, at least one is Contradiction | `Contradiction` |
| no Entailment, every pair has a verdict, none is Contradiction | `Neutral` |
| no Entailment, and at least one pair has no verdict | no verdict (`null`): the claim is unjudged, never a miss |

The scores, the buckets and the findings are computed from the folded verdicts
exactly as in `many-to-many`: one folded verdict is one claim in the funnels below.

### Why the matcher never sees the response

Matching is claim against claim. The matcher once also got the response text, to
tell whether two phrasings name the same thing. It used the text as evidence
instead: a claim with no counterpart on the other side was matched because the
response states it. That inflates both ratios, and in pass 2 it hides exactly
the facts the ground truth is missing. On `msmarco_claude2` (707 predicted
claims, the same predictions for every run), withholding the response dropped
recall from 0.88 to 0.65 and precision from 0.89 to 0.81; in a hand sample of
the lost matches, about 9 in 10 had no counterpart at all. The price is a few
real paraphrases with a framing word ("HP insists X works best" against "X works
best") now judged a miss.

### Three verdict outcomes, not two

The matcher can return four things (three verdicts, or none at all), and they
have three effects. Each claim comes back as exactly one of:

| verdict | meaning | effect on metrics |
| --- | --- | --- |
| `Entailment` | counterpart found | numerator **and** denominator |
| `Contradiction` / `Neutral` | judged, no counterpart | denominator only (a real miss) |
| `None` | the checker never returned a verdict (parse failure after all retries) | **neither** — the claim leaves the ratio entirely |

`Contradiction` and `Neutral` share a row because they have the same effect:
both mean no counterpart. What they say differs: `Contradiction` means the other
side states the opposite, `Neutral` that it does not state the claim.

A claim nobody judged is not evidence about the extractor. Unjudged claims are
excluded from both sides of the fraction, counted in `checker_failures`, and
listed per item in the findings file as `unjudged`. This mirrors the
None-propagation rule `ragcheck` follows.

Note the sampling bias this leaves behind: bundles do not fail at random, so a
high `checker_failure_rate` means the numbers are not merely less precise, they
may be biased. That is why the console prints a "run `eval checker`" pointer
whenever the rate is above zero.

## Step 5: Metrics

Both ratios are computed from **exhaustive partitions** of the issued claims —
every claim of a side lands in exactly one bucket:

```
recall_counts:
  total_gt_claims = covered + missed + answer_missed_penalty + unjudged
  denominator     = total_gt_claims - unjudged
  recall          = covered / denominator

precision_counts:
  total_pred_claims = supported + unsupported + abstention_misread_penalty + unjudged
  denominator       = total_pred_claims - unjudged
  precision         = supported / denominator

f1 = harmonic mean of precision and recall
```

The penalties are *synthetic*: no checker ever saw those claims; the item was
excluded wholesale and its claims charged. Keeping them separate from judged
misses is the point of the partition — "the extractor missed 48 claims" reads
very differently when it is 12 real misses plus 36 claims from 5 skipped items.

**The null rule.** A metric whose denominator is zero is `null`, never `0.0`.
Nothing was judged, and "not computable" is not a score. `f1` is `null` when
either side is. A `0.0` therefore always means a real, measured zero.

**Tooling rates** are reported alongside but never mixed in:

- `extraction_errors.rate` — items whose extraction failed / all extracted items
- `checker_failures.rate` — verdicts the checker never returned / verdicts issued
  (penalty claims were never sent, so they do not count as issued). In `joint`
  and `strict` a verdict here is a claim's folded verdict, not a pair's: a claim
  is unjudged only when no pair found it and some pair has no verdict

## Repeated runs extract again, on purpose

With `--runs N`, every run extracts again and matches again; no run reuses an
earlier run's predictions. With an extraction temperature above 0 the claims
differ from run to run, and at 0 the provider's nondeterminism still makes them
vary. That spread is the extractor's own behaviour and belongs to its score: a
single run is one sample of it, and the mean over runs is the score to report.
The spread also holds the matcher's noise, and this eval cannot tell the two
apart.

---

## Running the Evaluator (CLI)

```bash
claimlens eval extractor <input_file> [options]
```

> Requires **two** models: one performs the live extraction
> (`--extractor-model`), one acts as the checker in the 2-pass matching
> (`--checker-model`). They may be the same model — but then the thing being
> measured and the ruler measuring it are correlated. Both flags are required.

### Arguments and Options

| Parameter | Flag | Default | Description |
|---|---|---|---|
| **Input File** | `input_file` *(argument)* | *Required* | Eval JSON: items with a `response` and GT triplets. Positional — no flag in front of it. |
| **Extractor Model** | `--extractor-model`, `-e` | *Required* | Model used to extract triplets live. Also the prefix of the predicted-triplet key. |
| **Checker Model** | `--checker-model` | *Required* | LLM checker for the 2-pass matching. |
| **Extractor API Base** | `--extractor-base-api` | `None` | Base URL for the extractor (OpenAI-SDK route). Unset → LiteLLM provider routing. |
| **Checker API Base** | `--checker-base-api` | `None` | Base URL for the checker. |
| **GT Key** | `--gt-key` | `claude2_response_kg` | Item key holding the ground-truth triplets. |
| **Atomizer Model** | `--atomizer-model` | `None` | Enables the atomicity axis. Needs `ATOMIZER_API_KEY`; silently skipped if unset. |
| **Atomizer API Base** | `--atomizer-base-api` | `None` | Base URL for the atomizer. |
| **Output File** | `--output`, `-o` | `results/{input_stem}_extractor_eval[_{runs}].json` | Summary JSON. The `_{runs}` suffix appears only for `--runs > 1`. A `*_findings.json` (the review queue) is written alongside. |
| **Match Mode** | `--match-mode` | `joint` | What one matcher call holds: `joint` (up to `--joint-num` claims against one reference claim), `strict` (one claim against one reference claim), `many-to-many` (up to `--joint-num` claims against the whole list of reference claims). `joint` and `strict` ask the same pairs and fold them; `many-to-many` is not one-to-one. See [The three modes](#the-three-modes). |
| **Joint Bundle Size** | `--joint-num` | `10` | Max claims bundled per matching call in `joint` and `many-to-many`; `strict` sends one. |
| **Word Budget** | `--max-words` | `None` | Word budget per matching call. Unset, a matching call uses `DEFAULT_MAX_WORDS` (6,000), of which 75 % goes to the reference and the claims; a call over it gets fewer claims. |
| **Runs** | `--runs` | `1` | Repeat the whole eval N times and report variance (N × LLM cost). Every run extracts again (see Repeated runs). |
| **Skip Marked** | `--skip-marked` | `None` | Leave out every item that carries this key, e.g. `cut`: not extracted, not matched, counted as `skipped_marked` in `counts.data`. The item stays in the input file. |
| **Question If** | `--question-if` | `needs_question` | An item whose value under this key is `true` is extracted with its question, through `extractor_prompt_question`; every other item from its text alone. A marked item without a `question` is dropped (`dropped_no_question`). `""` turns it off. |
| **Concurrency** | `--concurrency` | `10` | Max simultaneous LLM requests per client. |
| **Debug Mode** | `--debug` | `False` | Timestamps + module names in logs. |

> **The predicted-triplet key is derived, not configurable.** Live extraction
> always writes under `{extractor_model}_response_kg`, and that is where the
> evaluator reads predictions from. If that key would equal `--gt-key` (e.g.
> `--extractor-model claude2` with the default GT key) the evaluator raises
> `InvalidInputError` immediately — otherwise extraction would overwrite the GT
> slot and the eval would match ground truth against itself, reporting a
> flawless and completely fictional score.

### Required environment (`.env`)

| Variable | Used for |
|---|---|
| `EXTRACTOR_API_KEY` | Auth for the extractor LLM endpoint. |
| `CHECKER_API_KEY` | Auth for the checker/matching LLM endpoint. |
| `ATOMIZER_API_KEY` | Auth for the atomizer. Without it the atomicity axis is skipped. |
| `LLM_TIMEOUT` *(optional)* | Per-call timeout in seconds (default `120.0`). |

### Worked example

```bash
claimlens eval extractor eval_data/msmarco/msmarco_gpt4_5.json \
  --extractor-model gemini-3.1 \
  --checker-model claude-haiku-5.5 \
  --extractor-base-api http://localhost:4000/v1 \
  --checker-base-api http://localhost:4000/v1
```

Two different models: a model that matches its own extraction is no independent
check.

Reads GT from `claude2_response_kg`, extracts live into
`gemini-3.1_response_kg`, runs the 2-pass match, then writes
`eval_data/msmarco/results/msmarco_gpt4_5_extractor_eval.json` plus its
`_findings.json` sibling. Output always lands in a `results/` directory
beside the input, so an input already inside `results/` nests one level deeper —
pass `-o` to control this.

---

## Output Format

Two files are written. The **record** holds everything: metrics, every count
the console prints, and the complete per-item claim lists. The **findings**
file is the review queue — only what went wrong, each entry tagged with a
`kind` — derived from the record's items and never a source of its own. Both
open with `_args` (what you asked for) and `_meta` (what the run turned out
to be; see docs/output_conventions.md).

The skeleton is the same at `--runs 1` and `--runs N`: `runs` is always a
list, `metrics` is the mean over runs (the run's own values at N = 1), and
`variance` the spread. `--runs` adds entries, it never reshapes.

### Record — `<filename>_extractor_eval[_N].json`

```json
{
  "_args": { "...": "..." },
  "_meta": { "...": "...", "pred_key": "gemini-3.1_response_kg", "matching": "llm-2-pass",
             "match_mode": "joint", "joint_num": 10, "runs": 1 },
  "metrics": {
    "recall": 0.871, "precision": 0.991, "f1": 0.927,
    "atomicity_rate": 1.0, "claim_density": 1.0, "duplicate_rate": 0.0,
    "abstention_recognized_rate": 0.0, "abstention_misread_rate": 1.0, "answer_missed_rate": 0.0,
    "extraction_error_rate": 0.0, "checker_failure_rate": 0.0, "atomization_failure_rate": 0.0
  },
  "variance": { "recall": { "n": 1, "std": 0.0, "min": 0.871, "max": 0.871, "values": [0.871] }, "...": "..." },
  "runs": [
    {
      "_meta": { "...": "...", "run": 1, "duration_seconds": 90.6 },
      "metrics": { "...same keys, this run's values..." },
      "counts": {
        "data":      { "dropped_no_response": 0, "dropped_no_gt_key": 0, "gt_empty": 1 },
        "recall":    { "total_gt_claims": 116, "covered": 101, "missed": 15, "answer_missed_penalty": 0, "unjudged": 0, "denominator": 116 },
        "precision": { "total_pred_claims": 105, "supported": 104, "unsupported": 0, "abstention_misread_penalty": 1, "unjudged": 0, "denominator": 105 },
        "extraction_stats": { "gt": { "claims": 116, "avg_per_item": 4.83 }, "pred": { "claims": 105, "avg_per_item": 4.38 } },
        "abstention_handling": { "answers": 24, "answers_extracted": 24, "answers_missed": 0,
                                 "abstentions": 1, "abstentions_recognized": 0, "abstentions_misread": 1 },
        "reliability": {
          "extraction":       { "failed": 0, "items": 25, "by_cause": {} },
          "matching_checker": { "unjudged": 0, "issued": 221, "items_affected": 0, "unjudged_gt": 0, "unjudged_pred": 0 },
          "atomization":      { "measured": false, "reason": "no --atomizer-model" }
        },
        "atomicity":  null,
        "duplicates": { "predicted_claims": 105, "unique_claims": 105, "duplicate_claims": 0 }
      },
      "items": [
        { "id": "12345", "question": "...", "response": "...", "bucket": "compared",
          "gt_claims":   [ { "subject": "...", "predicate": "...", "object": "..." },
                           { "subject": "...", "predicate": "...", "object": "..." } ],
          "pred_claims": [ { "subject": "...", "predicate": "...", "object": "..." } ],
          "pred2gt":     [ { "verdict": "Entailment", "explanation": "...", "matched_by": [0], "pairs": [ ... ] },
                           { "verdict": "Neutral",    "explanation": "...", "matched_by": [],  "pairs": [ ... ] } ],
          "gt2pred":     [ { "verdict": "Entailment", "explanation": "...", "matched_by": [0], "pairs": [ ... ] } ] },
        { "id": "1088355", "question": "...", "response": "The passages do not provide ...",
          "bucket": "abstention_misread", "gt_claims": [],
          "pred_claims": [ { "subject": "...", "predicate": "...", "object": "..." } ] }
      ]
    }
  ]
}
```

- `metrics` is the variance roster: the numbers the console's Metrics
  rows and the run line show. Rates live here and nowhere else.
- `counts` mirrors the console blocks one to one: 📂 Data, the two
  🔎 Matching Quality funnels, 📊 Extraction Stats, ⚪ Abstention Handling,
  the three 💥 Reliability rows, 🧬 Atomicity and 🔁 Duplicates (numbers
  only). An axis that did not run says so: `"measured": false` with the
  reason, and `atomicity: null`.
- `items` lists every attempted item with its `bucket` (`compared`,
  `answer_missed`, `abstention_recognized`, `abstention_misread`,
  `extraction_error`) and both claim lists as clean `{subject, predicate,
  object}`, the shape ragcheck writes. Compared items add the verdict cells
  `{verdict, explanation}` (sparse `error`, `retry`; `matched_by` and `pairs`
  in `joint` and `strict`, see below) in two arrays parallel
  to the claims, named like ragcheck's directions, reference first:
  `pred2gt` (pass 1, GT claims against the predictions) and `gt2pred`
  (pass 2). The other buckets never reached the matcher, so they have none.
  A prediction can be taken into the ground truth exactly as the extractor
  wrote it.
  Split decisions (`atomicity_splits`) and `duplicate_claims` ride on the
  item they belong to; an errored item carries its `cause`.
- `_meta.match_mode` names the mode, `_meta.joint_num` the claims per call
  the run used (`1` in `strict`); the ⚙️ Config block prints both in its
  `Match mode:` line, with the word budget in `joint` and `many-to-many`, where a
  bundle can shrink:
  `Match mode:  joint · up to 10 claims against one reference claim per call · 6000 max words`.

#### The cells in `joint` and `strict`

A cell keeps `{verdict, explanation}` (sparse `error`, `retry`) with the
folded verdict, so every reader of `pred2gt` and `gt2pred` reads it as in
`many-to-many`. It adds the pairs:

```json
"pred2gt": [
  { "verdict": "Entailment", "explanation": "... R2 states the same ...",
    "matched_by": [1],
    "pairs": [ { "index": 0, "verdict": "Neutral",    "explanation": "..." },
               { "index": 1, "verdict": "Entailment", "explanation": "... R2 states the same ..." } ] },
  { "verdict": "Neutral", "explanation": "No single prediction makes the claim true (2 Neutral).",
    "matched_by": [],
    "pairs": [ { "index": 0, "verdict": "Neutral", "explanation": "..." },
               { "index": 1, "verdict": "Neutral", "explanation": "..." } ] }
]
```

- `pairs` — one entry per reference claim, in the order of the other side's
  list: `index` is its position there, 0-based (in `pred2gt` an index into
  `pred_claims`, in `gt2pred` into `gt_claims`), and the matcher saw it as
  `R{index + 1}`: index 2 is `(R3)`. Then the pair's own `verdict` and
  `explanation`, sparse `error` and `retry`, as a ragcheck matrix cell has.
- `matched_by` — the indexes of the reference claims that made the claim true;
  empty for a miss and for an unjudged claim.
- `explanation` is always a string. For an Entailment it is the explanation of
  the first pair that made the claim true; for a miss one sentence that no
  single claim of the other side makes it true, with the count of the pair
  verdicts; for an unjudged claim one sentence that says how many pairs gave
  no verdict. The pairs carry the detail.
- `error` (unjudged claims only) is the cause of the first pair without a
  verdict. `retry` is the round of the pair whose explanation the cell
  carries, so only an Entailment can have one; a miss is decided by every
  pair, and each pair keeps its own round.

The pairs of a run are one row each of (`pred`, `gt`, direction, verdict): in
`pred2gt` the claim is a GT claim and the reference a prediction, in `gt2pred`
the other way round.

### Findings — `<filename>_extractor_eval[_N]_findings.json`

The console branches opened up: one list per branch, every entry names its
item. Empty branches stay present — hidden is not zero.

```json
{
  "_args": { "...": "..." },
  "_meta": { "...": "..." },
  "runs": [
    { "_meta": { "...": "...", "run": 1 },
      "findings": {
        "missed":             [ { "id": "12345", "question": "...",
                                  "claim": { "subject": "...", "predicate": "...", "object": "..." },
                                  "verdict": "Neutral", "explanation": "...",
                                  "response": "...", "gt_claims": [ { "subject": "...", ... } ], "pred_claims": [ ... ] } ],
        "unsupported":        [ { "...": "same shape as missed" } ],
        "answer_missed":      [ { "id": "77", "question": "...", "response": "...", "claims": [ { "subject": "...", ... } ] } ],
        "abstention_misread": [ { "id": "1088355", "question": "...", "response": "...", "claims": [ { "subject": "...", ... } ] } ],
        "unjudged":           [ { "id": "...", "question": "...", "claim": { "subject": "...", ... }, "side": "gt",
                                  "cause": "checker_failure", "response": "...", "gt_claims": [ ... ], "pred_claims": [ ... ] } ],
        "extraction_failed":  [ { "id": "857956", "question": "...", "cause": "parse_failure" } ]
      } }
  ]
}
```

- `missed` / `unsupported` — the Matching Quality misses, each with the
  **judged** verdict, the checker's explanation, and what it is judged
  against: the response and both claim lists (`gt_claims`, `pred_claims`),
  as `eval checker` entries carry their chunks. Every claim is
  `{subject, predicate, object}`, not text as in ragcheck's findings: this
  queue is reviewed to repair the ground truth, and a missing claim goes in
  as the triplet the finding names.
- `pairs` (`joint` and `strict`) — a `missed`, `unsupported` or `unjudged`
  entry shows each reference claim with its verdict and explanation, so the
  reviewer sees why no single claim matched:
  `{ "index": 1, "reference_claim": { "subject": "...", ... }, "verdict": "Neutral", "explanation": "..." }`
  (sparse `error`, `retry`). In `many-to-many` the entries have no `pairs`.
- `unjudged` — a claim the checker never returned a verdict for (`side`
  names the pass). Not a disagreement, but the item stays traceable.
- `answer_missed` / `abstention_misread` — the ⚪ Abstention Handling
  branches: the GT claims lost to an empty extraction, the claims invented
  from a refusal.
- `extraction_failed` — the 💥 row's items, with their cause.
- `_meta` is a copy of the record's, so the two files identify each other.

With `--runs N`, the variance block aggregates the twelve `metrics` keys as
`mean ± std [min, max]` across runs (see docs/output_conventions.md rule
set 4); `variance[key].n` says how many runs contributed, and an axis that
never ran keeps its key with `n: 0`.

---

## Reading the console output

Each matching pass prints its 🌐 API & Parsing block and its 🔎 Checking tree.
`tasks sent to LLM` is the number of calls the mode's formula gives. In `joint`
and `strict` the tree counts pairs (`🔎 Checking — 52 pairs`), one verdict per
(claim, reference claim); the funnels below count claims, one folded verdict
each.

```
 🔎 Matching Quality  (LLM 2-pass)
    Recall — 27 total GT claims
     ├─ ✅ 15 covered by predictions  (judged)
     ├─ ❌ 0 missed  (judged)
     ├─ ⚪ 12 answer-missed penalty  (1 items, 0 predictions for 12 claims)
     ├─ 💥 0 unjudged by checker
     └─ → Recall 0.556  (15 / 27)
```

Each funnel accounts for every issued claim: the first three branches sum to the
denominator, and the 💥 branch sits explicitly outside it. The same partition is
in the JSON, so print and file can never disagree.

Below the funnels, the orthogonal sections always print (a hidden row is
indistinguishable from an unmeasured one): `⚪ Abstention Handling` (two
response kinds with their outcomes as sub-branches — answers: extracted /
nothing extracted; abstentions: recognized / misread — footer rates
`abstention recognized` and `abstention misread` over the abstaining
responses, `answer missed` over the answering ones), `💥 Reliability` (extraction + matching checker
+ atomization, each row `count of denominator → rate_key rate`; an
unconfigured axis prints `not measured`), `🧬 Atomicity` and
`🔁 Duplicates`. See docs/output_conventions.md for the display rules.
