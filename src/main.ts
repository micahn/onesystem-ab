/**
 * Run the A/B/C.
 *
 * Usage:
 *   bun run src/main.ts                          # all three models, default corpus
 *   bun run src/main.ts --corpus corpus/x.json   # a different corpus
 *   bun run src/main.ts --models laya julia      # which ones
 *   bun run src/main.ts --repeat 3               # ask each case N times, to check determinism
 *
 * Refuses to run against one model when asked for more. A comparison that silently degraded
 * into a single run is the failure mode this script exists to avoid.
 */

import { available, expectedLabel, run, write, type RunReport } from "./ab.ts"
import { readFile } from "node:fs/promises"
import { join } from "node:path"

const args = process.argv.slice(2)

function flag(name: string, fallback?: string): string | undefined {
  const i = args.indexOf(`--${name}`)
  return i >= 0 ? args[i + 1] : fallback
}

const corpusPath = flag("corpus", join(import.meta.dir, "..", "corpus", "default.json"))!
const models = args.includes("--models")
  ? args.slice(args.indexOf("--models") + 1).filter((a) => !a.startsWith("--"))
  : ["laya", "julia", "rizzo"]
const repeat = Number(flag("repeat", "1"))

const health = await available()
if (!health) {
  console.error("no onesystem daemon is answering. Start one: `onesystem start`")
  process.exit(1)
}

const answering = health.backends.filter((b) => b.local).map((b) => b.name)
const missing = models.filter((m) => !answering.includes(m))
if (missing.length > 0) {
  console.error(
    `not enabled in the config: ${missing.join(", ")}\n` +
      `enabled: ${answering.join(", ") || "none"}\n` +
      `switch with: onesystem use <model>  (or enable them all in the config)`,
  )
  process.exit(1)
}

console.log(`corpus: ${corpusPath}  (${models.length} models${repeat > 1 ? `, x${repeat}` : ""})`)
console.log(`models: ${models.join(" vs ")}`)
console.log("")

const report = await run(corpusPath, models, { repeat })
const { json, md } = await write(report, join(import.meta.dir, "..", "out"))

const corpus = JSON.parse(await readFile(corpusPath, "utf8"))
for (const m of report.models) {
  const acc = m.accuracy === null ? "—" : `${(m.accuracy * 100).toFixed(1)}%`
  console.log(
    `${m.model.padEnd(7)} ${String(m.correct).padStart(3)}/${m.total}  ${acc.padStart(7)}  ` +
      `${String(m.meanMs).padStart(5)}ms mean  ${String(m.warmMeanMs).padStart(4)}ms warm  ${m.errors} err`,
  )
}

console.log("")
console.log("per task:")
const tasks = [...new Set(report.models.flatMap((m) => Object.keys(m.byTask)))].sort()
for (const t of tasks) {
  console.log(
    `  ${t.padEnd(11)} ` +
      report.models
        .map((m) => {
          const s = m.byTask[t]
          return s ? `${m.model} ${s.correct}/${s.total}` : null
        })
        .filter(Boolean)
        .join("   "),
  )
}

console.log("")
console.log("per type:")
const types = [...new Set(report.models.flatMap((m) => Object.keys(m.byType)))].sort()
for (const t of types) {
  console.log(
    `  ${t.padEnd(11)} ` +
      report.models
        .map((m) => {
          const s = m.byType[t]
          return s ? `${m.model} ${s.correct}/${s.total}` : null
        })
        .filter(Boolean)
        .join("   "),
  )
}

console.log("")
console.log("pairwise lift over independence (0 = same information):")
for (const p of report.pairs) {
  console.log(
    `  ${(p.a + " / " + p.b).padEnd(16)} agree ${pct(p.observed)}  vs independent ${pct(p.independent)}  ` +
      `lift ${p.lift >= 0 ? "+" : ""}${(p.lift * 100).toFixed(1)}pp   only-${p.a} ${p.onlyA}  only-${p.b} ${p.onlyB}`,
  )
}

if (report.unanimous.total > 0) {
  console.log("")
  console.log(`unanimous: ${report.unanimous.count}/${report.unanimous.total}`)
}
if (report.majority) {
  console.log(`majority:  ${(report.majority.accuracy! * 100).toFixed(1)}%  (${report.majority.note})`)
}
const best = Math.max(...report.models.map((m) => m.accuracy ?? 0))
console.log(
  `oracle:    ${report.oracle.correct}/${report.oracle.total} (${((report.oracle.correct / report.oracle.total) * 100).toFixed(1)}%)` +
    `   headroom over best single: ${report.oracle.headroom} cases`,
)
if (report.stability) {
  console.log("")
  console.log("determinism across repeats:")
  for (const s of report.stability) {
    console.log(`  ${s.model.padEnd(7)} ${s.identical}/${s.total} identical`)
  }
}

console.log("")
console.log(`disagreements: ${report.disagreements.length}`)
console.log(`written: ${md}`)
console.log(`         ${json}`)

function pct(n: number): string {
  return `${(n * 100).toFixed(0)}%`
}

export type { RunReport }
export { expectedLabel }
