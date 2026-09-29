/**
 * A/B/C across the onesystem decision models.
 *
 * "Deterministic" here means three specific things, because a comparison you cannot repeat
 * is an anecdote:
 *
 *   1. The same questions, in the same order, every run. The corpus is data, not code, and
 *      it is hashed — if it changes, the run says so instead of silently comparing
 *      different things.
 *   2. No sampling is requested anywhere. Every model returns a distribution and we read
 *      its argmax. There is no temperature to set and no seed to fix, because none of them
 *      generate. `--repeat` verifies that rather than assuming it.
 *   3. Every model sees the identical question payload, byte for byte. Only the `state`
 *      wrapper differs, and only because laya rejects a string while julia rejects an
 *      object — see `STATE_SHAPE`.
 *
 * ## Where the models are NOT interchangeable
 *
 * The inputs match; the *answers* do not, and the divergence is in the contract rather than
 * in the judgement. Measured on this machine:
 *
 *   - `noul` means opposite things. laya's `noul` is P(true) and its `confidence` is
 *     P(false); julia's `noul` is P(true) with no confidence field; rizzo's `noul` is
 *     P(true) with no confidence either. julia additionally answers ~0.0001 for both
 *     polarities of the same question, i.e. it does not respond to the question at all.
 *   - `score` is a soft position in laya (1.28 for an argmax of 1) and a near-integer index
 *     in julia and rizzo.
 *   - Confidence is named three different ways and, for laya's `score`, two of them appear
 *     at once with different values (`confidence` 0.28, `answer_confidence` 0.63).
 *
 * So each model gets a reader that knows its own shape. That is not the thing the harness
 * refuses to do — a per-model *input* adapter is where a fair comparison quietly stops
 * being fair, and there is exactly one of those, for `state`, and it is documented. Reading
 * three different outputs is unavoidable if the models are to be compared at all, and
 * doing it here rather than in a caller is the point: the incompatibility is a finding to
 * report, not a bug to hide in someone else's integration.
 *
 * ## What this does not do
 *
 * It does not pick a winner and it does not average across runs. Three models separated by
 * a few points on 36 questions are not distinguishable at that sample size, so a script
 * that prints a confident ranking anyway is worse than one that prints the counts and lets
 * the overlap show. Every headline here is reported with the number it came from.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { probeHealth, type HealthReport } from "../../../Projects/onesystem/src/health.ts"

const ONESYSTEM = process.env.ONESYSTEM_URL ?? "http://127.0.0.1:7331"

export type QuestionType = "choice" | "noul" | "score"

export interface Question {
  id: string
  type: QuestionType
  /** Domain tag, so results can be read per category rather than as one blended number. */
  task: string
  instructions: string
  /** Object for choice/noul, ordered array for score. See the module comment. */
  criteria: Record<string, string> | string[]
  /** Key for choice, "true"/"false" for noul, index for score. */
  answer: string | number
  /** Why the ground truth is what it is. Carried into the report so it can be challenged. */
  why?: string
  note?: string
}

export interface Corpus {
  name: string
  questions: Question[]
}

export interface CaseResult {
  id: string
  type: QuestionType
  task?: string
  /** What the model picked, normalised to a string so all three types compare. */
  answer: string
  expected: string
  correct: boolean
  /** The model's own confidence in its pick, or null when it reports none for this type. */
  confidence: number | null
  /** For score: the model's soft position, which need not be an integer index. */
  position?: number | null
  error?: string
  ms: number
  /** Only rizzo reports these today. Null rather than 0 for the others. */
  inferenceMs?: number | null
  queueMs?: number | null
  inTokens?: number | null
  outTokens?: number | null
  /** The process's peak device allocation as the driver reported it. */
  peakBytes?: number | null
}

export interface ModelResult {
  model: string
  cases: CaseResult[]
  correct: number
  total: number
  /** Fraction correct. Null when nothing ran, so an empty run is not reported as 0%. */
  accuracy: number | null
  meanMs: number
  /** Cold-inclusive and warm-only, because the first call is not the steady state. */
  warmMeanMs: number
  wrongConfidence: number | null
  /** Per task and per type, because a blended number hides where a model is weak. */
  byTask: Record<string, { correct: number; total: number }>
  byType: Record<string, { correct: number; total: number }>
  errors: number
}

export interface PairAnalysis {
  a: string
  b: string
  agree: number
  total: number
  observed: number
  /** What agreement would be if the two were independent draws of their own accuracy. */
  independent: number
  /** Observed minus independent. Near zero means the two carry the same information. */
  lift: number
  bothRight: number
  bothWrong: number
  onlyA: number
  onlyB: number
}

export interface RunReport {
  corpus: string
  corpusSha: string
  startedAt: string
  models: ModelResult[]
  /** Every model pair, not just the first two. */
  pairs: PairAnalysis[]
  /** Cases where the models disagree — the only ones worth reading. */
  disagreements: { id: string; task?: string; type: QuestionType; expected: string; byModel: Record<string, string> }[]
  /** How often all three landed on the same answer. */
  unanimous: { count: number; total: number }
  /**
   * How often at least one model was right. The ceiling for *any* strategy that picks
   * between these models — routing, voting, escalation. It is the number that decides
   * whether ensembling is worth building: headroom is the gap between this and the best
   * single model, and on a set where the models share biases that gap can be tiny while a
   * realistic vote loses a lot.
   */
  oracle: { correct: number; total: number; headroom: number }
  /**
   * What a 3-model majority would score. Meaningful where a 2-model vote is a coin flip:
   * with three, the modal answer is a real decision rather than a tie to break.
   */
  majority: { accuracy: number | null; correct: number; total: number; note: string } | null
  /** Whether repeating a case gives the same answer. Null unless --repeat was above 1. */
  stability: { model: string; identical: number; total: number }[] | null
  repeat: number
}

/**
 * How each model wants `state` typed.
 *
 * A hard incompatibility discovered by running them, not a preference:
 *
 *   laya 0.3.21  `state` must be an object — a string is a pydantic validation error
 *   Julia-1      `state` must be a string  — a dict is a validation error
 *   rizzo        `state` accepts both, and is given the object form to match laya
 *
 * So the models are not drop-in interchangeable for a caller: a payload written for one is
 * rejected by the other at the type checker. The questions stay byte-identical and only
 * this wrapper differs, with the same text either way.
 */
const STATE_SHAPE: Record<string, "object" | "string"> = {
  laya: "object",
  julia: "string",
  rizzo: "object",
}

function stateText(q: Question): string {
  return `You are choosing between labelled options for: ${q.instructions}`
}

/** Every model answers `state` + named questions. Only the state wrapper differs. */
function payload(q: Question, model: string) {
  const shape = STATE_SHAPE[model] ?? "string"
  return {
    state: shape === "object" ? { situation: stateText(q) } : stateText(q),
    questions: { [q.id]: { type: q.type, instructions: q.instructions, criteria: q.criteria } },
  }
}

/** Options in the order a score question ranks them. */
function levels(q: Question): string[] {
  return Array.isArray(q.criteria) ? q.criteria : Object.values(q.criteria)
}

/** The ground truth as a string, so all three types compare the same way. */
export function expectedOf(q: Question): string {
  if (q.type === "score") return String(q.answer)
  return String(q.answer)
}

/** The level text a score question's ground truth refers to, for the report. */
export function expectedLabel(q: Question): string {
  return q.type === "score" ? (levels(q)[Number(q.answer)] ?? "?") : String(q.answer)
}

/**
 * Read one model's answer into a common shape.
 *
 * Per model *and* per type, because the models genuinely disagree about their own contract
 * — see the module comment. `probabilities` is read where present so a `choice` can be
 * scored by argmax rather than by a label the model chose to print.
 *
 * Never throws. A shape change should produce one row with a visible reason, not a thrown
 * error that loses the whole run.
 */
export function readAnswer(
  body: string,
  q: Question,
  model: string,
): { answer: string; confidence: number | null; position?: number | null; error?: string; usage?: RawUsage } {
  let envelope: any
  try {
    envelope = JSON.parse(body)
  } catch {
    return { answer: "", confidence: null, error: `not JSON: ${body.slice(0, 120)}` }
  }
  if (envelope?.error) {
    return { answer: "", confidence: null, error: `rpc error: ${JSON.stringify(envelope.error).slice(0, 160)}` }
  }

  // Two layers of JSON. The wire format is an MCP envelope whose `result.content[0].text`
  // is itself a JSON *string*. Parsing the outer object and looking for `answers` there
  // finds nothing and reports it as "the model did not answer", which reads like a model
  // problem and is actually a reader problem.
  const text = envelope?.result?.content?.[0]?.text
  let parsed: any
  try {
    parsed = typeof text === "string" ? JSON.parse(text) : (text ?? envelope)
  } catch {
    return { answer: "", confidence: null, error: `content was not JSON: ${String(text).slice(0, 120)}` }
  }

  const entry = parsed?.answers?.[q.id]
  if (!entry) {
    return {
      answer: "",
      confidence: null,
      error: `no answer for ${q.id}; got keys: ${Object.keys(parsed?.answers ?? {}).join(",") || "none"}`,
    }
  }

  const usage = readUsage(parsed)
  const probs: Record<string, number> | undefined = entry.probabilities

  if (q.type === "choice") {
    // `act_probability` is 1.0 on every laya answer ever returned and carries no
    // information, so it is never read. Anything exactly 1.0 is treated as the placeholder.
    const candidates = [entry.max_probability, entry.answer_confidence, entry.confidence]
    const raw = candidates.find((c) => typeof c === "number" && c < 1)
    return {
      answer: String(entry.choice ?? ""),
      confidence: typeof raw === "number" ? raw : null,
      ...(usage ? { usage } : {}),
    }
  }

  if (q.type === "noul") {
    // All three put P(true) in `noul`, but only laya also reports a confidence and that
    // field is P(false) — the complement, not a second opinion. Reading it as a
    // confidence would invert the signal, so it is dropped and noul is the confidence.
    const pTrue = typeof entry.noul === "number" ? entry.noul : null
    const fromProbs = typeof probs?.["true"] === "number" ? probs["true"] : null
    const truth = pTrue ?? fromProbs
    if (truth === null) return { answer: "", confidence: null, error: "noul had no probability" }
    return {
      answer: truth >= 0.5 ? "true" : "false",
      confidence: truth,
      ...(usage ? { usage } : {}),
    }
  }

  // score. laya returns a soft position that need not be an integer; julia and rizzo return
  // something very close to one. The argmax of the level probabilities is the pick, so a
  // soft position is scored the same way a hard one is.
  const keys = Object.keys(probs ?? {})
  let best = -1
  for (const k of keys) if ((probs![k] ?? -1) > (probs![String(best)] ?? -1)) best = Number(k)
  const position = typeof entry.score === "number" ? entry.score : null
  const answer = best >= 0 ? String(best) : position !== null ? String(Math.round(position)) : ""
  if (!answer) return { answer: "", confidence: null, error: "score had no probabilities and no position" }
  const conf = [entry.answer_confidence, entry.max_probability, entry.confidence].find(
    (c) => typeof c === "number" && c < 1,
  )
  return {
    answer,
    confidence: typeof conf === "number" ? conf : typeof probs?.[answer] === "number" ? probs![answer] : null,
    position,
    ...(usage ? { usage } : {}),
  }
}

interface RawUsage {
  inTokens?: number
  outTokens?: number
  inferenceMs?: number
  queueMs?: number
  peakBytes?: number
}

/**
 * Only rizzo reports timing and tokens today. laya and julia return neither, so their
 * fields stay null and the report says "not reported" rather than implying zero — a model
 * that took 0ms of inference would be a claim, not a measurement.
 */
function readUsage(parsed: any): RawUsage | undefined {
  const usage = parsed?.usage
  const timing = parsed?.x_rizzo?.timing
  if (!usage && !timing) return undefined
  return {
    inTokens: typeof usage?.input_tokens === "number" ? usage.input_tokens : undefined,
    outTokens: typeof usage?.output_tokens === "number" ? usage.output_tokens : undefined,
    inferenceMs: typeof timing?.inference_seconds === "number" ? timing.inference_seconds * 1000 : undefined,
    queueMs: typeof timing?.queue_seconds === "number" ? timing.queue_seconds * 1000 : undefined,
    peakBytes: typeof timing?.peak_device_bytes === "number" ? timing.peak_device_bytes : undefined,
  }
}

/**
 * Ask one model one question.
 *
 * A fresh MCP session per model, not per question: the session carries the cold load, and
 * reusing it is the only fair comparison of warm latency. The first case therefore pays the
 * cold cost for that model and every later one does not — which is why mean and warmMean are
 * both reported rather than just the first.
 */
async function runModel(model: string, corpus: Corpus, repeat: number): Promise<ModelResult> {
  const base = `${ONESYSTEM}/mcp/${model}`
  const headers = { "content-type": "application/json", accept: "application/json, text/event-stream" }

  const initRes = await fetch(base, {
    method: "POST",
    headers,
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "ab", version: "2" } },
    }),
  })
  const session = initRes.headers.get("mcp-session-id")
  if (!session) throw new Error(`${model}: no MCP session (is the backend enabled?)`)
  const sid = { ...headers, "mcp-session-id": session }
  await initRes.text()
  await fetch(base, {
    method: "POST",
    headers: sid,
    body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
  })

  const cases: CaseResult[] = []
  // The first pass of each case is the one scored. Later passes only check that asking
  // twice gives the same answer, which is the claim the whole "deterministic" heading rests
  // on and is worth verifying rather than asserting.
  const firstAnswers = new Map<string, string>()
  let identical = 0
  let stabilityTotal = 0

  for (const q of corpus.questions) {
    let row: CaseResult | null = null
    for (let attempt = 0; attempt < repeat; attempt++) {
      const started = Date.now()
      let got: ReturnType<typeof readAnswer>
      let body = ""
      try {
        const res = await fetch(base, {
          method: "POST",
          headers: sid,
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: 2 + attempt,
            method: "tools/call",
            params: { name: "predict", arguments: payload(q, model) },
          }),
        })
        body = await res.text()
        const dataLine = body.split("\n").find((l) => l.startsWith("data: "))?.slice(6) ?? body
        got = readAnswer(dataLine, q, model)
      } catch (err) {
        got = { answer: "", confidence: null, error: String(err) }
      }
      const ms = Date.now() - started
      if (attempt === 0) {
        const expected = expectedOf(q)
        row = {
          id: q.id,
          type: q.type,
          task: q.task,
          answer: got.answer,
          expected,
          correct: got.answer === expected,
          confidence: got.confidence,
          position: got.position ?? null,
          error: got.error,
          ms,
          inferenceMs: got.usage?.inferenceMs ?? null,
          queueMs: got.usage?.queueMs ?? null,
          inTokens: got.usage?.inTokens ?? null,
          outTokens: got.usage?.outTokens ?? null,
          peakBytes: got.usage?.peakBytes ?? null,
        }
        firstAnswers.set(q.id, got.answer)
      } else {
        stabilityTotal++
        if (firstAnswers.get(q.id) === got.answer) identical++
      }
    }
    if (row) cases.push(row)
  }

  const result = summarise(model, cases)
  if (repeat > 1) (result as ModelResult & { _stability?: [number, number] })._stability = [identical, stabilityTotal]
  return result
}

function summarise(model: string, cases: CaseResult[]): ModelResult {
  const correct = cases.filter((c) => c.correct).length
  const wrongConf = cases.filter((c) => !c.correct && c.confidence !== null).map((c) => c.confidence!)
  const warm = cases.slice(1)
  const tally = (key: "task" | "type"): Record<string, { correct: number; total: number }> => {
    const out: Record<string, { correct: number; total: number }> = {}
    for (const c of cases) {
      const k = String(c[key] ?? "?")
      out[k] ??= { correct: 0, total: 0 }
      out[k].total++
      if (c.correct) out[k].correct++
    }
    return out
  }
  return {
    model,
    cases,
    correct,
    total: cases.length,
    accuracy: cases.length === 0 ? null : correct / cases.length,
    meanMs: cases.length === 0 ? 0 : Math.round(cases.reduce((a, c) => a + c.ms, 0) / cases.length),
    warmMeanMs: warm.length === 0 ? 0 : Math.round(warm.reduce((a, c) => a + c.ms, 0) / warm.length),
    wrongConfidence: wrongConf.length === 0 ? null : wrongConf.reduce((a, b) => a + b, 0) / wrongConf.length,
    byTask: tally("task"),
    byType: tally("type"),
    errors: cases.filter((c) => c.error).length,
  }
}

/**
 * Compare every model against every other.
 *
 * The independent rate is the whole point. If two models agree about as often as their own
 * accuracies predict, they are carrying the same information and running both buys nothing
 * however clever the policy is. Above it, there is something to combine; below it, they are
 * systematically wrong in the same direction, which is worse than useless because it looks
 * like agreement.
 */
function analysePairs(models: ModelResult[]): PairAnalysis[] {
  const out: PairAnalysis[] = []
  for (let i = 0; i < models.length; i++) {
    for (let j = i + 1; j < models.length; j++) {
      const a = models[i]!
      const b = models[j]!
      const byId = new Map(b.cases.map((c) => [c.id, c]))
      const pairs = a.cases.map((x) => ({ x, y: byId.get(x.id) })).filter((p) => p.y) as {
        x: CaseResult
        y: CaseResult
      }[]
      const bothRight = pairs.filter((p) => p.x.correct && p.y.correct).length
      const bothWrong = pairs.filter((p) => !p.x.correct && !p.y.correct).length
      const onlyA = pairs.filter((p) => p.x.correct && !p.y.correct).length
      const onlyB = pairs.filter((p) => !p.x.correct && p.y.correct).length
      const agree = bothRight + bothWrong
      const pa = a.accuracy ?? 0
      const pb = b.accuracy ?? 0
      const independent = pa * pb + (1 - pa) * (1 - pb)
      out.push({
        a: a.model,
        b: b.model,
        agree,
        total: pairs.length,
        observed: pairs.length === 0 ? 0 : agree / pairs.length,
        independent,
        lift: pairs.length === 0 ? 0 : agree / pairs.length - independent,
        bothRight,
        bothWrong,
        onlyA,
        onlyB,
      })
    }
  }
  return out
}

/**
 * The 3-model majority.
 *
 * Worth measuring precisely because a 2-model vote is a coin flip — with two there is no
 * majority, only a tie to break arbitrarily. With three the modal answer is a real decision,
 * and it costs three calls per question.
 */
function analyseMajority(models: ModelResult[]): RunReport["majority"] {
  if (models.length < 3) return null
  const ids = models[0]!.cases.map((c) => c.id)
  let correct = 0
  let tie = 0
  for (const id of ids) {
    const answers = models.map((m) => m.cases.find((c) => c.id === id)!).filter(Boolean)
    const counts = new Map<string, number>()
    for (const a of answers) counts.set(a.answer, (counts.get(a.answer) ?? 0) + 1)
    const ranked = [...counts.entries()].sort((p, q2) => q2[1] - p[1])
    if (ranked.length > 1 && ranked[0]![1] === ranked[1]![1]) tie++
    const expected = answers[0]!.expected
    if (ranked[0]?.[0] === expected) correct++
  }
  return {
    accuracy: ids.length === 0 ? null : correct / ids.length,
    correct,
    total: ids.length,
    note:
      `modal answer wins, ${tie} of ${ids.length} cases three-way tied` +
      (models.length > 3 ? `, and only the first three of ${models.length} models voted` : ""),
  }
}

export interface RunOptions {
  repeat?: number
}

export async function run(corpusPath: string, models: string[], options: RunOptions = {}): Promise<RunReport> {
  const repeat = options.repeat ?? 1
  const raw = await readFile(corpusPath, "utf8")
  const corpus = JSON.parse(raw) as Corpus
  // Hashed so a changed corpus is visible in the output rather than silently different.
  const sha = createHash("sha256").update(raw).digest("hex").slice(0, 12)

  const results: ModelResult[] = []
  for (const m of models) results.push(await runModel(m, corpus, repeat))

  const byId = new Map<string, Record<string, string>>()
  for (const r of results) {
    for (const c of r.cases) {
      const row = byId.get(c.id) ?? {}
      row[r.model] = c.answer || `error: ${c.error ?? "?"}`
      byId.set(c.id, row)
    }
  }
  const disagreements = corpus.questions
    .filter((q) => new Set(Object.values(byId.get(q.id) ?? {})).size > 1)
    .map((q) => ({
      id: q.id,
      task: q.task,
      type: q.type,
      expected: expectedLabel(q),
      byModel: byId.get(q.id)!,
    }))

  const total = corpus.questions.length
  const unanimous = {
    count: corpus.questions.filter((q) => new Set(Object.values(byId.get(q.id) ?? {})).size === 1).length,
    total,
  }

  const stability =
    repeat > 1
      ? results.map((m) => {
          const s = (m as ModelResult & { _stability?: [number, number] })._stability ?? [0, 0]
          return { model: m.model, identical: s[0], total: s[1] }
        })
      : null

  return {
    corpus: corpus.name,
    corpusSha: sha,
    startedAt: new Date().toISOString(),
    models: results,
    pairs: analysePairs(results),
    disagreements,
    unanimous,
    oracle: oracleOf(results),
    majority: analyseMajority(results),
    stability,
    repeat,
  }
}

/**
 * The best any chooser could do: right whenever at least one model was.
 *
 * Reported because it bounds the entire ensembling question. A strategy cannot beat this,
 * and if this is close to the best single model then no amount of clever routing helps —
 * the models fail together, not independently.
 */
function oracleOf(models: ModelResult[]): RunReport["oracle"] {
  const ids = models[0]?.cases.map((c) => c.id) ?? []
  const byModel = models.map((m) => new Map(m.cases.map((c) => [c.id, c])))
  const correct = ids.filter((id) => byModel.some((m) => m.get(id)?.correct)).length
  const best = Math.max(0, ...models.map((m) => m.correct))
  return { correct, total: ids.length, headroom: correct - best }
}

export async function write(report: RunReport, dir: string): Promise<{ json: string; md: string }> {
  await mkdir(dir, { recursive: true })
  const stamp = report.startedAt.replace(/[:.]/g, "-").slice(0, 19)
  const json = join(dir, `ab-${stamp}.json`)
  const md = join(dir, `ab-${stamp}.md`)
  await writeFile(json, JSON.stringify(report, null, 2))
  await writeFile(md, render(report))
  return { json, md }
}

const pct = (n: number | null): string => (n === null ? "—" : `${(n * 100).toFixed(1)}%`)

/** The human half. Counts first, then the disagreements, because that is the order they matter in. */
function render(r: RunReport): string {
  const L: string[] = []
  L.push(`# A/B/C — ${r.corpus}`)
  L.push("")
  L.push(`Corpus \`${r.corpusSha}\`, ${r.startedAt}`)
  L.push("")
  L.push(
    "> `state` is typed differently per model and they are not interchangeable: laya needs an " +
      "object, julia needs a string, rizzo takes either. The questions are byte-identical; only the " +
      "state wrapper differs. The *answers* diverge too — see the wire-shape note in src/ab.ts.",
  )
  L.push("")
  L.push("| model | correct | accuracy | mean ms | warm ms | confidence when wrong | errors |")
  L.push("|---|---|---|---|---|---|---|")
  for (const m of r.models) {
    const wc = m.wrongConfidence === null ? "—" : m.wrongConfidence.toFixed(2)
    L.push(`| ${m.model} | ${m.correct}/${m.total} | ${pct(m.accuracy)} | ${m.meanMs} | ${m.warmMeanMs} | ${wc} | ${m.errors} |`)
  }

  L.push("")
  L.push("## Per task")
  L.push("")
  const tasks = [...new Set(r.models.flatMap((m) => Object.keys(m.byTask)))].sort()
  L.push("| task | " + r.models.map((m) => m.model).join(" | ") + " |")
  L.push("|---|" + r.models.map(() => "---").join("|") + "|")
  for (const t of tasks) {
    L.push(
      `| ${t} | ` +
        r.models
          .map((m) => {
            const s = m.byTask[t]
            return s ? `${s.correct}/${s.total}` : "—"
          })
          .join(" | ") +
        " |",
    )
  }

  L.push("")
  L.push("## Per question type")
  L.push("")
  const types = [...new Set(r.models.flatMap((m) => Object.keys(m.byType)))].sort()
  L.push("| type | " + r.models.map((m) => m.model).join(" | ") + " |")
  L.push("|---|" + r.models.map(() => "---").join("|") + "|")
  for (const t of types) {
    L.push(
      `| ${t} | ` +
        r.models
          .map((m) => {
            const s = m.byType[t]
            return s ? `${s.correct}/${s.total}` : "—"
          })
          .join(" | ") +
        " |",
    )
  }

  L.push("")
  L.push("## Do they carry different information?")
  L.push("")
  L.push(
    "Agreement against what independence would predict. Near the independent rate means running both " +
      "buys nothing however clever the policy; above it, there is something to combine.",
  )
  L.push("")
  L.push("| pair | agree | observed | if independent | lift | both right | both wrong | only A | only B |")
  L.push("|---|---|---|---|---|---|---|---|---|")
  for (const p of r.pairs) {
    L.push(
      `| ${p.a} / ${p.b} | ${p.agree}/${p.total} | ${pct(p.observed)} | ${pct(p.independent)} | ` +
        `${p.lift >= 0 ? "+" : ""}${(p.lift * 100).toFixed(1)}pp | ${p.bothRight} | ${p.bothWrong} | ${p.onlyA} | ${p.onlyB} |`,
    )
  }

  if (r.unanimous.total > 0) {
    L.push("")
    L.push(
      `All ${r.models.length} models gave the same answer on ${r.unanimous.count}/${r.unanimous.total} ` +
        `(${pct(r.unanimous.count / r.unanimous.total)}) of cases. The rest are below.`,
    )
  }

  L.push("")
  L.push("## Is there anything left to win?")
  L.push("")
  L.push(
    `Oracle (right whenever \*any\* model was): **${r.oracle.correct}/${r.oracle.total}** ` +
      `(${pct(r.oracle.total === 0 ? null : r.oracle.correct / r.oracle.total)}). That is the ceiling for any ` +
      `routing or voting scheme. Headroom over the best single model: **${r.oracle.headroom} case${r.oracle.headroom === 1 ? "" : "s"}.**`,
  )

  if (r.majority) {
    L.push("")
    L.push("## Would a majority vote help?")
    L.push("")
    L.push(
      "With two models a vote is a coin flip — there is no majority. With three the modal answer is a " +
        "real decision, at the cost of a call to every model.",
    )
    L.push("")
    L.push(
      `Three-model majority: **${pct(r.majority.accuracy)}** (${r.majority.correct}/${r.majority.total}). ${r.majority.note}.`,
    )
  }

  if (r.stability) {
    L.push("")
    L.push("## Is it actually deterministic?")
    L.push("")
    L.push("Every case asked `--repeat` times. Same answer every time, or the comparison is a coin flip.")
    L.push("")
    L.push("| model | identical | asked |")
    L.push("|---|---|---|")
    for (const s of r.stability) L.push(`| ${s.model} | ${s.identical} | ${s.total} |`)
  }

  if (r.disagreements.length > 0) {
    L.push("")
    L.push("## Where they disagree")
    L.push("")
    L.push(
      "| case | task | type | expected | " + r.models.map((m) => m.model).join(" | ") + " |",
    )
    L.push("|---|---|---|---|" + r.models.map(() => "---").join("|") + "|")
    for (const d of r.disagreements) {
      L.push(
        `| \`${d.id}\` | ${d.task ?? ""} | ${d.type} | \`${d.expected}\` | ` +
          r.models.map((m) => `\`${d.byModel[m.model] ?? "?"}\``).join(" | ") +
          " |",
      )
    }
  }

  L.push("")
  L.push("## Is any model's confidence usable for gating?")
  L.push("")
  L.push("A confidence signal only helps if it separates right from wrong.")
  L.push("")
  L.push("| model | when right | when wrong | usable? |")
  L.push("|---|---|---|---|")
  for (const m of r.models) {
    const withC = m.cases.filter((c) => c.confidence !== null)
    const right = withC.filter((c) => c.correct).map((c) => c.confidence!)
    const wrong = withC.filter((c) => !c.correct).map((c) => c.confidence!)
    if (right.length === 0 || wrong.length === 0) {
      L.push(`| ${m.model} | — | — | no rows |`)
      continue
    }
    const mr = right.reduce((a, b) => a + b, 0) / right.length
    const mw = wrong.reduce((a, b) => a + b, 0) / wrong.length
    const usable = mr - mw > 0.05 ? "yes" : mr - mw < -0.05 ? "**inverted**" : "**no — carries no signal**"
    L.push(`| ${m.model} | ${mr.toFixed(3)} | ${mw.toFixed(3)} | ${usable} |`)
  }

  const failed = r.models.flatMap((m) => m.cases.filter((c) => c.error).map((c) => ({ m: m.model, c })))
  if (failed.length > 0) {
    L.push("")
    L.push("## Errors")
    L.push("")
    for (const f of failed) L.push(`- \`${f.m}/${f.c.id}\`: ${f.c.error}`)
  }
  return L.join("\n") + "\n"
}

/** Which backends are answering, so a run cannot silently test one model. */
export async function available(): Promise<HealthReport | null> {
  return probeHealth(ONESYSTEM)
}

export { levels as scoreLevels }
