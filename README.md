# onesystem model A/B/C

Compares the three onesystem decision models — **laya**, **julia**, **rizzo** — by asking
them identical questions and scoring the answers.

## Run it

```bash
onesystem start                      # all three must be enabled in the config
bun run src/main.ts --corpus corpus/broad.json --repeat 2
```

Flags: `--corpus <path>` (default `corpus/default.json`), `--models laya julia rizzo`,
`--repeat N` (ask each case N times, to verify determinism).

Outputs land in `out/`: a timestamped `.json` report and a `.md` summary.

## The page

```bash
bun run src/page.ts out/*.json --metrics data/metrics.json --findings data/findings.json > out/comparison.html
```

One self-contained HTML file — no scripts, fonts or network. It shows the measurements,
the per-domain and per-type breakdowns, the confidence analysis, every disagreement
expanded with the question and the reasoning behind the ground truth, and a written
strengths/weaknesses assessment.

`data/metrics.json` holds the two things a run cannot measure about itself: cold-start
latency (needs a restarted daemon) and VRAM (needs a live card). `data/findings.json` is
hand-written judgement, deliberately kept out of the generator so it is never mistaken for
something measured.

## Corpora

| file | cases | what it is |
|---|---|---|
| `default.json` | 20 | Software-engineering decisions (`choice` only) |
| `wide.json` | 44 | Wider engineering: refactors, deploys, schemas, oncall, security |
| `broad.json` | 36 | New: support triage, bug diagnosis, Linux/Omarchy desktop, judgement |

Corpora are hashed into every report, so a changed corpus shows up in the output instead of
silently comparing different things. `broad.json` carries a `why` on every case defending its
ground truth, which the page surfaces so a disagreement can be checked rather than trusted.

## What it will not tell you

It does not rank, and it does not average across runs. It prints counts, per-corpus rows, an
oracle ceiling, and every disagreement, and leaves the reading to you. All three models are
deterministic — they return a distribution and the harness reads the argmax, with no sampling
and no seed — so `--repeat` confirms rather than establishes that, and a single run's
differences are real rather than noise.

Note that `state` is typed differently per model (laya and rizzo reject a string, julia
rejects an object) and the *answers* diverge too: `noul` has no explicit polarity field, and
confidence is named three different ways. The input side is held identical; the output side
needs a per-model reader. See the module comment in `src/ab.ts` for the full list — that
incompatibility is itself one of the findings.
