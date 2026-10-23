import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/* WCAG 2.x contrast, computed from the tokens in index.css (the single source of truth).
   AA needs 4.5:1 for normal text and 3:1 for large text and for UI component boundaries (1.4.11). */

const css = readFileSync(join(__dirname, '..', 'index.css'), 'utf8')

type RGBA = [number, number, number, number]

function token(name: string): RGBA {
  const m = css.match(new RegExp(`--color-${name}:\\s*([^;]+);`))
  if (!m) throw new Error(`token --color-${name} not found in index.css`)
  const v = m[1].trim()
  if (v.startsWith('#')) {
    const n = parseInt(v.slice(1), 16)
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 1]
  }
  const r = v.match(/rgba?\(([^)]+)\)/)
  if (!r) throw new Error(`cannot parse ${v}`)
  const [R, G, B, A = '1'] = r[1].split(',').map((s) => s.trim())
  return [Number(R), Number(G), Number(B), Number(A)]
}

/** Composite a possibly-translucent colour over an opaque background. */
function over(fg: RGBA, bg: RGBA): RGBA {
  const a = fg[3]
  return [0, 1, 2].map((i) => Math.round(fg[i] * a + bg[i] * (1 - a))).concat(1) as RGBA
}

function luminance([r, g, b]: RGBA): number {
  const lin = (c: number) => {
    const s = c / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}

function contrast(a: RGBA, b: RGBA): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

const bg = token('bg')
const solid = token('solid')
const glassOverBg = over(token('glass'), bg)

describe('text contrast (WCAG AA, 4.5:1)', () => {
  const cases: [string, RGBA, RGBA][] = [
    ['text on bg', token('text'), bg],
    ['text on solid', token('text'), solid],
    ['text on glass', token('text'), glassOverBg],
    ['muted on bg', token('muted'), bg],
    ['muted on solid', token('muted'), solid],
    ['muted on glass', token('muted'), glassOverBg],
    ['accent on bg', token('accent'), bg],
    ['accent on solid', token('accent'), solid],
    ['accent-2-text on bg', token('accent-2-text'), bg],
    ['accent-2-text on solid', token('accent-2-text'), solid],
    ['danger on solid', token('danger'), solid],
    ['danger on bg', token('danger'), bg],
    ['dark text on an accent-filled button', bg, token('accent')],
  ]
  it.each(cases)('%s', (_name, fg, back) => {
    expect(contrast(fg, back)).toBeGreaterThanOrEqual(4.5)
  })
})

describe('non-text UI contrast (WCAG 1.4.11, 3:1)', () => {
  it('control borders are visible against the page and against panels', () => {
    for (const back of [bg, solid, glassOverBg]) {
      expect(contrast(over(token('control-line'), back), back)).toBeGreaterThanOrEqual(3)
    }
  })
  it('violet is usable for non-text marks (rings, fills, graph nodes)', () => {
    expect(contrast(token('accent-2'), bg)).toBeGreaterThanOrEqual(3)
    expect(contrast(token('accent-2'), solid)).toBeGreaterThanOrEqual(3)
  })
})

describe('documented failures the design works around', () => {
  // These encode WHY the rules below exist; if the palette changes and they stop being true, the
  // rules can be relaxed deliberately instead of by accident.
  it('raw violet fails as small text, so it is never used for text', () => {
    expect(contrast(token('accent-2'), solid)).toBeLessThan(4.5)
  })
  it('light text on a cyan fill is unreadable, so cyan buttons use dark text', () => {
    expect(contrast(token('text'), token('accent'))).toBeLessThan(4.5)
  })
  it('the decorative 18% panel border is below 3:1, so it is never used on controls', () => {
    expect(contrast(over(token('line'), bg), bg)).toBeLessThan(3)
  })
})

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f)
    return statSync(p).isDirectory() ? sourceFiles(p) : /\.(tsx|ts|css)$/.test(f) ? [p] : []
  })
}

describe('source guards', () => {
  const files = sourceFiles(join(__dirname, '..')).filter((f) => !f.includes('test'))
  it('never sets text in raw violet (use text-accent-2-text)', () => {
    for (const f of files) {
      expect(readFileSync(f, 'utf8'), f).not.toMatch(/\btext-accent-2(?!-text)\b/)
    }
  })
  it('never puts light text on a cyan fill', () => {
    for (const f of files) {
      expect(readFileSync(f, 'utf8'), f).not.toMatch(/bg-accent(?![-\w/])[^"'`]*\btext-(text|muted)\b/)
    }
  })
})
