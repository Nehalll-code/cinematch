/** Thin typed fetch wrapper. Every request goes to our own origin under /api (Vite/the host proxies
 *  it to FastAPI), so there is no CORS and no third-party key in the browser. */

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

let getToken: () => string | null = () => null
let onUnauthorized: () => void = () => {}

export function configureApi(opts: { getToken: () => string | null; onUnauthorized: () => void }) {
  getToken = opts.getToken
  onUnauthorized = opts.onUnauthorized
}

/** FastAPI errors: {detail: string} for HTTPException, {detail: [{msg, loc}, ...]} for validation. */
function messageFrom(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail.length) {
    const first = detail[0] as { msg?: string; loc?: unknown[] }
    const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : undefined
    return field !== undefined && field !== 'body' ? `${String(field)}: ${first.msg}` : String(first.msg)
  }
  return fallback
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown; auth?: boolean } = {}): Promise<T> {
  const { json, auth = false, headers, ...rest } = init
  const h = new Headers(headers)
  if (json !== undefined) h.set('Content-Type', 'application/json')
  const token = auth ? getToken() : null
  if (token) h.set('Authorization', `Bearer ${token}`)

  let res: Response
  try {
    res = await fetch(`/api${path}`, { ...rest, headers: h, body: json !== undefined ? JSON.stringify(json) : rest.body })
  } catch {
    throw new ApiError(0, 'Cannot reach the CineMatch API. Is it running?')
  }
  if (res.status === 204) return undefined as T
  const body: unknown = await res.json().catch(() => null)
  if (!res.ok) {
    if (res.status === 401 && token) onUnauthorized() // an expired/invalid session, not a bad login attempt
    throw new ApiError(res.status, messageFrom(body, `Request failed (${res.status})`))
  }
  return body as T
}

/** Release year from a MovieLens title: "Heat (1995)" -> 1995. */
export function yearOf(title: string): number | null {
  const m = title.match(/\((\d{4})\)\s*$/)
  return m ? Number(m[1]) : null
}

/** Title without its trailing year, for display next to a separate year. */
export function bareTitle(title: string): string {
  return title.replace(/\s*\(\d{4}\)\s*$/, '')
}
