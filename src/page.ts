/**
 * Render one self-contained HTML comparison page from one or more run reports.
 *
 * Usage: bun run src/page.ts out/ab-*.json > comparison.html
 *
 * Two kinds of content, kept apart on purpose:
 *
 *   - The measurements come from the reports, and this file only lays them out. It does not
 *     rank, does not average across corpora, and shows every number it was given.
 *   - The strengths, weaknesses and the System 1 verdict come from a hand-written findings
 *     file (findings.json), because they are judgements about what the numbers mean for a
 *     particular use, and burying that in a generator would make it look measured.
 *
 * Everything is inlined: no scripts fetched, no fonts fetched, no build step. The page has to
 * still work in a year from a directory someone no longer remembers.
 */

import { readFile } from "node:fs/promises"
import { basename } from "node:path"
import type { RunReport, ModelResult, PairAnalysis } from "./ab.ts"

const esc = (s: unknown): string =>
  String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")

const gb = (n: number | null | undefined): string =>
  n === null || n === undefined || !Number.isFinite(n) ? "—" : `${(n / 2 ** 30).toFixed(2)} G`
const pct = (n: number | null | undefined, digits = 1): string =>
  n === null || n === undefined ? "—" : `${(n * 100).toFixed(digits)}%`

interface Findings {
  /** Per model, in the order they should appear. */
  models: {
    model: string
    role: string
    strengths: string[]
    weaknesses: string[]
    /** One line: what you would actually use it for. */
    use: string
  }[]
  /** What a System 1 model is for, judged against what these three measured. */
  systemOne: { verdict: string; goodFor: string[]; badFor: string[]; wiring: string }
  /** Anything the numbers do not settle, stated as such. */
  limits: string[]
}

/** Load a corpus so disagreements can be read, not just identified by id. */
async function loadCorpus(path: string): Promise<Map<string, any>> {
  try {
    const doc = JSON.parse(await readFile(path, "utf8"))
    return new Map((doc.questions as any[]).map((q) => [q.id, q]))
  } catch {
    return new Map()
  }
}

function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0
  const s = [...xs].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2
}

/** A horizontal accuracy bar, so a 45% and an 86% cannot be mistaken for neighbours. */
function bar(fraction: number | null, tone: string): string {
  const w = fraction === null ? 0 : Math.max(1.5, fraction * 100)
  return `<span class="bar"><i class="${tone}" style="width:${w.toFixed(1)}%"></i></span>`
}

const TONE: Record<string, string> = { laya: "t-laya", julia: "t-julia", rizzo: "t-rizzo" }
const fallbackTone = "t-other"

async function main(): Promise<void> {
  /**
   * Split argv into positional report paths and `--flag value` pairs. Done as one pass
   * rather than by filtering out anything starting with `--`, which would sweep up the
   * flag's own value and try to read it as a report.
   */
  const FLAGS = ["--findings", "--metrics"] as const
  const positional: string[] = []
  const flagValue: Partial<Record<(typeof FLAGS)[number], string>> = {}
  for (let i = 0; i < process.argv.length - 2; i++) {
    const arg = process.argv[i + 2]!
    if ((FLAGS as readonly string[]).includes(arg)) {
      flagValue[arg as (typeof FLAGS)[number]] = process.argv[i + 3]
      i++
    } else {
      positional.push(arg)
    }
  }
  const paths = positional
  const findingsPath = flagValue["--findings"]
  /**
   * Things a run cannot measure about itself: how long a model took to load from cold, and
   * how much VRAM it holds. Both need a restarted daemon and a live card, so they are
   * measured separately and handed in, rather than inferred from a warm run.
   */
  const metrics: { coldMs?: Record<string, number>; vram?: Record<string, number | null> } = flagValue["--metrics"]
    ? JSON.parse(await readFile(flagValue["--metrics"]!, "utf8"))
    : {}
  if (paths.length === 0) {
    console.error(
      "usage: bun run src/page.ts out/ab-*.json [--findings findings.json] [--metrics metrics.json]",
    )
    process.exit(1)
  }

  const reports: { report: RunReport; corpusPath: string; corpus: Map<string, any> }[] = []
  for (const p of paths) {
    const report = JSON.parse(await readFile(p, "utf8")) as RunReport
    // The corpus lives next to the harness, named by the report's own name.
    const corpusPath = `corpus/${report.corpus}.json`
    reports.push({ report, corpusPath, corpus: await loadCorpus(corpusPath) })
  }
  const findings: Findings | null = findingsPath
    ? JSON.parse(await readFile(findingsPath, "utf8"))
    : null

  const modelNames = reports[0]!.report.models.map((m) => m.model)
  const toneOf = (m: string) => TONE[m] ?? fallbackTone

  // Aggregate across corpora, but keep each corpus's own row visible so a blended figure
  // never hides a corpus where a model did unusually well or badly.
  const totalCases = new Map<string, number>()
  const totalCorrect = new Map<string, number>()
  for (const { report } of reports) {
    for (const m of report.models) {
      totalCases.set(m.model, (totalCases.get(m.model) ?? 0) + m.total)
      totalCorrect.set(m.model, (totalCorrect.get(m.model) ?? 0) + m.correct)
    }
  }

  const H: string[] = []
  H.push(`<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>System 1 models — A/B/C comparison</title>
<style>
  :root {
    --bg:#0f1115; --panel:#161a21; --panel2:#1c212a; --line:#262c37;
    --fg:#e6e9ef; --muted:#98a2b3; --dim:#6b7480;
    --laya:#7aa2f7; --julia:#9ece6a; --rizzo:#e0af68; --other:#bb9af7;
    --good:#9ece6a; --bad:#f7768e; --warn:#e0af68;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1080px; margin:0 auto; padding:40px 24px 96px; }
  h1 { font-size:30px; margin:0 0 6px; letter-spacing:-.02em; }
  h2 { font-size:20px; margin:44px 0 14px; letter-spacing:-.01em; }
  h3 { font-size:16px; margin:26px 0 10px; color:var(--muted); font-weight:600; }
  p { margin:0 0 12px; }
  .sub { color:var(--muted); margin:0 0 4px; }
  .meta { color:var(--dim); font-size:13px; margin:0 0 28px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin:0 0 16px; }
  .grid { display:grid; gap:14px; }
  .g3 { grid-template-columns:repeat(3,1fr); }
  .g2 { grid-template-columns:repeat(2,1fr); }
  @media (max-width:820px){ .g3,.g2 { grid-template-columns:1fr; } }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  tr:last-child td { border-bottom:none; }
  code, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }
  .bar { display:inline-block; width:110px; height:7px; background:#252b36; border-radius:4px; overflow:hidden; vertical-align:middle; margin-right:8px; }
  .bar i { display:block; height:100%; border-radius:4px; }
  .t-laya{background:var(--laya)} .t-julia{background:var(--julia)} .t-rizzo{background:var(--rizzo)} .t-other{background:var(--other)}
  .tag { display:inline-block; padding:1px 8px; border-radius:20px; font-size:12px; font-weight:600; }
  .m-laya{color:var(--laya);background:rgba(122,162,247,.13)}
  .m-julia{color:var(--julia);background:rgba(158,206,106,.13)}
  .m-rizzo{color:var(--rizzo);background:rgba(224,175,104,.13)}
  .m-other{color:var(--other);background:rgba(187,154,247,.13)}
  .big { font-size:26px; font-weight:650; letter-spacing:-.02em; }
  .lbl { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  ul { margin:0 0 12px; padding-left:20px; } li { margin:0 0 6px; }
  .pro { border-left:2px solid var(--line); padding:2px 0 2px 16px; margin:0 0 14px; color:var(--muted); }
  .verdict { border-left:3px solid var(--rizzo); }
  .note { background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:12px 16px; color:var(--muted); font-size:14px; margin:0 0 16px; }
  .note strong { color:var(--fg); }
  details { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:0 16px; margin:0 0 8px; }
  details[open] { padding-bottom:12px; }
  summary { cursor:pointer; padding:12px 0; font-weight:550; }
  summary:hover { color:var(--rizzo); }
  .q { color:var(--muted); font-size:13.5px; margin:0 0 8px; }
  .opts { font-size:13px; color:var(--dim); margin:0 0 8px; }
  .why { font-size:13px; color:var(--dim); font-style:italic; }
  .yes { color:var(--good); } .no { color:var(--bad); }
  footer { color:var(--dim); font-size:13px; margin-top:48px; padding-top:20px; border-top:1px solid var(--line); }
</style></head><body><div class="wrap">`)

  H.push(`<h1>System 1 models — A/B/C comparison</h1>`)
  H.push(
    `<p class="sub">laya, julia and rizzo answering the same questions through onesystem.</p>`,
  )
  H.push(
    `<p class="meta">${reports
      .map(
        (r) =>
          `${esc(r.report.corpus)} <code>${esc(r.report.corpusSha)}</code> · ${r.report.models[0]?.total ?? 0} cases · ${esc(
            r.report.startedAt.slice(0, 16).replace("T", " "),
          )}`,
      )
      .join("<br>")}</p>`,
  )

  // ---- headline -----------------------------------------------------------
  H.push(`<h2>Headline</h2>`)
  H.push(`<div class="grid g3">`)
  for (const name of modelNames) {
    const correct = totalCorrect.get(name) ?? 0
    const total = totalCases.get(name) ?? 0
    const acc = total === 0 ? null : correct / total
    H.push(`<div class="card">
      <div class="lbl">${esc(name)}</div>
      <div class="big">${pct(acc, 0)}</div>
      <div class="sub">${correct}/${total} across ${reports.length} ${reports.length === 1 ? "corpus" : "corpora"}</div>
      ${bar(acc, toneOf(name))}
    </div>`)
  }
  H.push(`</div>`)

  // Per-corpus, so the blend above can be checked.
  H.push(`<h3>Per corpus</h3>`)
  H.push(`<div class="card"><table><tr><th>corpus</th>${modelNames
    .map((m) => `<th class="num">${esc(m)}</th>`)
    .join("")}<th class="num">cases</th></tr>`)
  for (const { report } of reports) {
    H.push(
      `<tr><td class="mono">${esc(report.corpus)}</td>` +
        report.models
          .map(
            (m) =>
              `<td class="num">${m.correct}/${m.total} ${bar(m.accuracy, toneOf(m.model))}</td>`,
          )
          .join("") +
        `<td class="num">${report.models[0]?.total ?? 0}</td></tr>`,
    )
  }
  H.push(`</table></div>`)

  // ---- cost ---------------------------------------------------------------
  H.push(`<h2>Cost</h2>`)
  H.push(
    `<div class="note">Cold is the first call after a daemon restart — model load, not <em>thinking</em>. ` +
      `Warm is every call after that. They differ by two orders of magnitude, so a single mean is misleading ` +
      `in both directions.</div>`,
  )
  H.push(`<div class="card"><table>
    <tr><th>model</th><th class="num">cold</th><th class="num">warm median</th><th class="num">warm p95</th>
    <th class="num">VRAM</th><th class="num">deterministic?</th><th class="num">tokens reported?</th></tr>`)
  const coldMs: Record<string, number | null> = metrics.coldMs ?? {}
  for (const name of modelNames) {
    const all = reports.flatMap((r) => r.report.models.filter((m) => m.model === name).flatMap((m) => m.cases.map((c) => c.ms)))
    const warm = all.slice(1)
    const p95 = warm.length ? warm.sort((a, b) => a - b)[Math.floor(warm.length * 0.95)]! : null
    const vram = metrics.vram?.[name] ?? null
    const stability = reports[0]!.report.stability?.find((s) => s.model === name)
    const tokens = reports[0]!.report.models.find((m) => m.model === name)?.cases.some((c) => c.inTokens != null) ?? false
    H.push(
      `<tr><td><span class="tag m-${toneOf(name).slice(2)}">${esc(name)}</span></td>` +
        `<td class="num">${coldMs[name] ? (coldMs[name]! / 1000).toFixed(1) + " s" : "—"}</td>` +
        `<td class="num">${median(warm).toFixed(0)} ms</td>` +
        `<td class="num">${p95 === null ? "—" : p95.toFixed(0) + " ms"}</td>` +
        `<td class="num">${gb(vram)}</td>` +
        `<td class="num">${stability ? (stability.identical === stability.total ? '<span class="yes">yes</span>' : `<span class="no">${stability.identical}/${stability.total}</span>`) : "not tested"}</td>` +
        `<td class="num">${tokens ? "yes" : "no"}</td></tr>`,
    )
  }
  H.push(`</table></div>`)

  // ---- per task -----------------------------------------------------------
  H.push(`<h2>Where each model is strong</h2>`)
  H.push(
    `<p class="sub">Accuracy per domain, pooled across corpora. A model that is excellent at one thing and ` +
      `mediocre at the rest is more useful than its average suggests, and the reverse is also true.</p>`,
  )
  const taskRows = new Map<string, { correct: Map<string, number>; total: Map<string, number> }>()
  for (const { report } of reports) {
    for (const m of report.models) {
      for (const [task, s] of Object.entries(m.byTask)) {
        const row = taskRows.get(task) ?? { correct: new Map(), total: new Map() }
        row.correct.set(m.model, (row.correct.get(m.model) ?? 0) + s.correct)
        row.total.set(m.model, (row.total.get(m.model) ?? 0) + s.total)
        taskRows.set(task, row)
      }
    }
  }
  H.push(`<div class="card"><table><tr><th>domain</th>${modelNames
    .map((m) => `<th class="num">${esc(m)}</th>`)
    .join("")}</tr>`)
  for (const [task, row] of [...taskRows.entries()].sort()) {
    H.push(
      `<tr><td>${esc(task)}</td>` +
        modelNames
          .map((m) => {
            const c = row.correct.get(m) ?? 0
            const t = row.total.get(m) ?? 0
            return `<td class="num">${t === 0 ? "—" : `${c}/${t} ${bar(c / t, toneOf(m))}`}</td>`
          })
          .join("") +
        `</tr>`,
    )
  }
  H.push(`</table></div>`)

  // ---- per type -----------------------------------------------------------
  H.push(`<h3>Per question type</h3>`)
  H.push(`<div class="card"><table><tr><th>type</th>${modelNames
    .map((m) => `<th class="num">${esc(m)}</th>`)
    .join("")}</tr>`)
  const typeRows = new Map<string, { correct: Map<string, number>; total: Map<string, number> }>()
  for (const { report } of reports) {
    for (const m of report.models) {
      for (const [t, s] of Object.entries(m.byType)) {
        const row = typeRows.get(t) ?? { correct: new Map(), total: new Map() }
        row.correct.set(m.model, (row.correct.get(m.model) ?? 0) + s.correct)
        row.total.set(m.model, (row.total.get(m.model) ?? 0) + s.total)
        typeRows.set(t, row)
      }
    }
  }
  for (const [t, row] of [...typeRows.entries()].sort()) {
    H.push(
      `<tr><td class="mono">${esc(t)}</td>` +
        modelNames
          .map((m) => {
            const c = row.correct.get(m) ?? 0
            const n = row.total.get(m) ?? 0
            return `<td class="num">${n === 0 ? "—" : `${c}/${n} ${bar(c / n, toneOf(m))}`}</td>`
          })
          .join("") +
        `</tr>`,
    )
  }
  H.push(`</table></div>`)

  // ---- calibration --------------------------------------------------------
  H.push(`<h2>Can you trust a model's confidence?</h2>`)
  H.push(
    `<p class="sub">Pooled over all ${reports.length} corpora. You gate on a confidence number by acting on the ` +
      `high band and escalating the low one, so what matters is whether the <em>top</em> of the range is pure ` +
      `and whether the model can express doubt at all. A small positive gap in the means is not enough: a model ` +
      `that reports 0.85 when right and 0.78 when wrong is technically better than chance and practically ` +
      `useless, because there is no band you can act on.</p>`,
  )
  H.push(`<div class="card"><table>
    <tr><th>model</th><th class="num">mean conf<br>right</th><th class="num">mean conf<br>wrong</th>
    <th class="num">lowest<br>ever</th><th class="num">high band<br>(≥0.70)</th><th class="num">precision<br>in band</th>
    <th class="num">wrong at<br>≥0.90</th><th>usable as a gate?</th></tr>`)
  const gateable: Record<string, boolean> = {}
  for (const name of modelNames) {
    // Pooled, because a confidence signal's usefulness is a property of the model rather
    // than of one corpus, and a 20-case slice is not enough rows to judge it on.
    const cases = reports.flatMap((r) => r.report.models.find((m) => m.model === name)?.cases ?? [])
    const withC = cases.filter((c) => c.confidence !== null)
    const right = withC.filter((c) => c.correct).map((c) => c.confidence!)
    const wrong = withC.filter((c) => !c.correct).map((c) => c.confidence!)
    if (right.length === 0 || wrong.length === 0) {
      H.push(`<tr><td>${esc(name)}</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="sub">not enough rows</td></tr>`)
      continue
    }
    const mr = mean(right)
    const mw = mean(wrong)
    const floor = Math.min(...withC.map((c) => c.confidence!))
    const band = withC.filter((c) => c.confidence! >= 0.7)
    const precision = band.length === 0 ? null : band.filter((c) => c.correct).length / band.length
    const overconfident = wrong.filter((c) => c > 0.9).length
    // Two things have to hold, and the second is an absolute bar rather than a comparison.
    // "Better than its own base rate" is not sufficient: a model at 46% overall whose top
    // band is 59% is better than nothing and still useless to act on. A gate exists to let
    // you skip the check on some answers, so the band has to be trustworthy in absolute
    // terms or you have just automated the error rate.
    const base = cases.filter((c) => c.correct).length / cases.length
    const purer = precision !== null && precision - base > 0.05
    const canDoubt = floor < 0.5
    const actionable = precision !== null && precision >= 0.85 && purer && canDoubt
    gateable[name] = actionable
    const verdict = actionable
      ? `<span class="yes">yes — ${band.filter((c) => c.correct).length}/${band.length} correct in the high band</span>`
      : !canDoubt
        ? '<span class="no">no — never reports low confidence, so there is no tail to escalate</span>'
        : precision === null
          ? '<span class="no">no — never reaches the high band at all</span>'
          : purer
            ? `<span class="no">barely — some signal, but ${pct(precision, 0)} in the high band is not something to act on</span>`
            : '<span class="no">no — the high band is no purer than the base rate</span>'
    H.push(
      `<tr><td><span class="tag m-${toneOf(name).slice(2)}">${esc(name)}</span></td>` +
        `<td class="num">${mr.toFixed(3)}</td><td class="num">${mw.toFixed(3)}</td>` +
        `<td class="num ${canDoubt ? "" : "no"}">${floor.toFixed(2)}</td>` +
        `<td class="num">${band.length}</td>` +
        `<td class="num">${precision === null ? "—" : pct(precision, 0)}</td>` +
        `<td class="num ${overconfident > 0 ? "no" : "yes"}">${overconfident}/${wrong.length}</td>` +
        `<td>${verdict}</td></tr>`,
    )
  }
  H.push(`</table></div>`)

  // ---- information overlap ------------------------------------------------
  H.push(`<h2>Do they know different things?</h2>`)
  H.push(
    `<p class="sub">Two models agreeing about as often as their own accuracies predict are carrying the ` +
      `same information, and running both buys nothing. <em>Lift</em> is observed agreement minus the ` +
      `independent rate: near zero means redundant, positive means there is something to combine.</p>`,
  )
  for (const { report } of reports) {
    H.push(`<h3 class="mono">${esc(report.corpus)}</h3>`)
    H.push(`<div class="card"><table>
      <tr><th>pair</th><th class="num">agree</th><th class="num">if independent</th><th class="num">lift</th>
      <th class="num">both wrong</th><th class="num">only one right</th></tr>`)
    for (const p of report.pairs) {
      H.push(
        `<tr><td>${esc(p.a)} <span class="sub">/</span> ${esc(p.b)}</td>` +
          `<td class="num">${p.agree}/${p.total} (${pct(p.observed, 0)})</td>` +
          `<td class="num">${pct(p.independent, 0)}</td>` +
          `<td class="num ${Math.abs(p.lift) < 0.05 ? "" : p.lift > 0 ? "yes" : "no"}">${p.lift >= 0 ? "+" : ""}${(p.lift * 100).toFixed(1)} pp</td>` +
          `<td class="num">${p.bothWrong}</td>` +
          `<td class="num">${p.onlyA} / ${p.onlyB}</td></tr>`,
      )
    }
    if (report.unanimous.total > 0) {
      H.push(
        `<tr><td colspan="6" class="sub" style="padding-top:12px">All three gave the same answer on ` +
          `<strong>${report.unanimous.count}/${report.unanimous.total}</strong> cases. The other ` +
          `${report.unanimous.total - report.unanimous.count} are listed below.</td></tr>`,
      )
    }
    H.push(`</table></div>`)
  }

  // ---- majority -----------------------------------------------------------
  H.push(`<h2>Does asking all three help?</h2>`)
  for (const { report } of reports) {
    if (!report.majority) continue
    const best = [...report.models].sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))[0]!
    const delta = (report.majority.accuracy ?? 0) - (best.accuracy ?? 0)
    H.push(`<div class="card">
      <p><strong>Three-model majority: ${pct(report.majority.accuracy)}</strong> vs best single model
      (<strong>${esc(best.model)} ${pct(best.accuracy)}</strong>) —
      <span class="${delta < -0.02 ? "no" : "yes"}">${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp</span>.</p>
      <p class="sub">${esc(report.majority.note)}. ${esc(report.corpus)} corpus.</p>
    </div>`)
  }
  H.push(
    `<div class="note"><strong>Read that before building a fan-out.</strong> A majority vote only helps when ` +
      `the models are of comparable quality. When one is clearly better than the other two, the modal answer ` +
      `is usually the wrong one, because two weak models outvote one strong one on exactly the cases the ` +
      `strong model finds difficult.</div>`,
  )

  // ---- oracle -------------------------------------------------------------
  H.push(`<h3>And how much is there to win at all?</h3>`)
  H.push(
    `<p class="sub">The ceiling for any strategy that picks between these models: correct whenever at ` +
      `<em>any</em> one of them was. If that is close to the best single model, no amount of clever routing ` +
      `helps — the models fail together.</p>`,
  )
  H.push(`<div class="card"><table>
    <tr><th>corpus</th><th class="num">oracle</th><th class="num">best single</th><th class="num">headroom</th><th>reading</th></tr>`)
  for (const { report } of reports) {
    const best = [...report.models].sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))[0]!
    const h = report.oracle.headroom
    H.push(
      `<tr><td class="mono">${esc(report.corpus)}</td>` +
        `<td class="num">${report.oracle.correct}/${report.oracle.total} (${pct(report.oracle.correct / report.oracle.total, 0)})</td>` +
        `<td class="num">${esc(best.model)} ${pct(best.accuracy, 0)}</td>` +
        `<td class="num">${h} case${h === 1 ? "" : "s"}</td>` +
        `<td class="sub">${h <= 4 ? "almost nothing left — the three agree on the hard cases and are wrong together" : "some room for a router"}</td></tr>`,
    )
  }
  H.push(`</table></div>`)

  // ---- findings -----------------------------------------------------------
  if (findings) {
    H.push(`<h2>Strengths and weaknesses</h2>`)
    for (const m of findings.models) {
      H.push(`<div class="card">
        <h3 style="margin-top:0"><span class="tag m-${toneOf(m.model).slice(2)}">${esc(m.model)}</span> ${esc(m.role)}</h3>
        <div class="grid g2">
          <div><div class="lbl">Strengths</div><ul>${m.strengths.map((s) => `<li>${s}</li>`).join("")}</ul></div>
          <div><div class="lbl">Weaknesses</div><ul>${m.weaknesses.map((s) => `<li>${s}</li>`).join("")}</ul></div>
        </div>
        <p class="pro" style="margin-top:10px"><strong>Use it for:</strong> ${esc(m.use)}</p>
      </div>`)
    }

    H.push(`<h2>As System 1 models</h2>`)
    H.push(`<div class="card verdict">
      <p>${findings.systemOne.verdict}</p>
      <div class="grid g2" style="margin-top:14px">
        <div><div class="lbl">Genuinely good for</div><ul>${findings.systemOne.goodFor.map((s) => `<li>${s}</li>`).join("")}</ul></div>
        <div><div class="lbl">Not for</div><ul>${findings.systemOne.badFor.map((s) => `<li>${s}</li>`).join("")}</ul></div>
      </div>
      <p class="pro" style="margin-top:12px"><strong>Wiring:</strong> ${esc(findings.systemOne.wiring)}</p>
    </div>`)

    H.push(`<h2>What this does not settle</h2>`)
    H.push(`<div class="card"><ul>${findings.limits.map((s) => `<li>${s}</li>`).join("")}</ul></div>`)
  }

  // ---- disagreements ------------------------------------------------------
  H.push(`<h2>Every disagreement</h2>`)
  H.push(
    `<p class="sub">Where the models differ, with the question and the reasoning behind the ground truth. ` +
      `These are the cases worth reading; the rest are all three saying the same thing.</p>`,
  )
  for (const { report, corpus } of reports) {
    H.push(`<h3 class="mono">${esc(report.corpus)} — ${report.disagreements.length} of ${report.models[0]?.total ?? 0}</h3>`)
    for (const d of report.disagreements) {
      const q = corpus.get(d.id)
      H.push(`<details><summary>
        <code>${esc(d.id)}</code> <span class="sub">${esc(d.task ?? "")} · ${esc(d.type)}</span>
      </summary>
      ${q ? `<p class="q">${esc(q.instructions)}</p>` : ""}
      ${
        q && q.criteria && !Array.isArray(q.criteria)
          ? `<p class="opts">${Object.entries(q.criteria)
              .map(([k, v]) => `<strong>${esc(k)}</strong> — ${esc(v)}`)
              .join("<br>")}</p>`
          : q
            ? `<p class="opts">${(q.criteria as string[]).map((c, i) => `<strong>${i}</strong> — ${esc(c)}`).join("<br>")}</p>`
            : ""
      }
      <table>
        <tr><th>expected</th>${report.models.map((m) => `<th>${esc(m.model)}</th>`).join("")}</tr>
        <tr><td class="mono">${esc(d.expected)}</td>${report.models
          .map((m) => {
            const a = d.byModel[m.model] ?? "?"
            const right = a === String((corpus.get(d.id)?.answer ?? ""))
            return `<td class="mono ${right ? "yes" : "no"}">${esc(a)}</td>`
          })
          .join("")}</tr>
      </table>
      ${q?.why ? `<p class="why" style="margin-top:10px">Ground truth: ${esc(q.why)}</p>` : ""}
      ${q?.note ? `<p class="sub" style="margin-top:4px">${esc(q.note)}</p>` : ""}
    </details>`)
    }
  }

  // ---- method -------------------------------------------------------------
  H.push(`<h2>Method</h2>`)
  H.push(`<div class="card">
    <p>Every model is sent the same questions, in the same order, with the same text. The only per-model
    difference is the <code>state</code> wrapper: laya and rizzo reject a bare string, julia rejects an
    object, so the text is identical and only the container differs.</p>
    <p>None of the three generate text — each returns a distribution over the options and the harness reads
    its argmax. There is no temperature and no seed. The determinism check asks every case twice and confirms
    the answers are identical, which is what makes a single run's differences meaningful rather than noise.</p>
    <p>Corpora are hashed, so a changed corpus is visible in the output rather than silently comparing
    different things. Per-corpus rows are always shown next to any blended figure.</p>
  </div>`)

  H.push(`<footer>
    Generated from ${paths.length} run${paths.length === 1 ? "" : "s"}:
    ${paths.map((p) => `<code>${esc(basename(p))}</code>`).join(", ")}.<br>
    Harness: <code>~/Work/ab</code>. Models served locally by onesystem on the RX 9070 XT.
  </footer>`)
  H.push(`</div></body></html>`)

  process.stdout.write(H.join("\n") + "\n")
}

await main()

export type { ModelResult, PairAnalysis }
