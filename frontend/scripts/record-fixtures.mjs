// Records REAL responses from a running CineMatch API into src/test/fixtures/, so component tests
// run against genuine response shapes and values rather than hand-written mock arrays.
//   npm run record-fixtures      (API must be up; defaults to http://127.0.0.1:8000)
import { mkdirSync, writeFileSync } from 'node:fs'

const base = process.env.CINEMATCH_API ?? 'http://127.0.0.1:8000'
const out = new URL('../src/test/fixtures/', import.meta.url)
mkdirSync(out, { recursive: true })

const targets = {
  health: '/health',
  metrics: '/metrics',
  recommend: '/recommend/1?k=6',
  profile: '/profile/1',
  movie: '/movies/1',
  similar: '/similar/1?k=12',
  search: '/search?q=toy%20story&limit=24',
}
for (const [name, path] of Object.entries(targets)) {
  const res = await fetch(base + path)
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  writeFileSync(new URL(`${name}.json`, out), JSON.stringify(await res.json(), null, 2) + '\n')
  console.log('recorded', name, path)
}
