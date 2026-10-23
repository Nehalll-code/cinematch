import { useHealth, useMetrics } from '../lib/queries'

/** Live service readout. Every figure is read from /health or /metrics; nothing is a constant. */
export function SystemStatus() {
  const health = useHealth()
  const metrics = useMetrics()

  if (health.isPending) return <p role="status" className="font-mono text-xs text-muted">connecting to API…</p>
  if (health.isError) {
    return <p role="alert" className="font-mono text-xs text-danger">API unreachable · {health.error.message}</p>
  }
  const h = health.data
  const m = metrics.data
  const lookups = m && 'cache' in m ? m.cache.hits + m.cache.misses : 0
  return (
    <p className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-xs text-muted" aria-label="System status">
      <span className={h.status === 'ok' ? 'text-accent' : 'text-danger'}>● {h.status === 'ok' ? 'model ready' : 'model not loaded'}</span>
      <span>{h.n_items.toLocaleString()} films</span>
      <span>{h.n_users.toLocaleString()} users</span>
      <span>index {h.index_verified ? 'verified' : h.db_configured ? 'unverified' : 'n/a (no database)'}</span>
      {m && 'p50_ms' in m ? (
        <span title="Server-side time of the /recommend lookup (a cache hit is ~0)">
          p50 {m.p50_ms.toFixed(2)} ms · p95 {m.p95_ms.toFixed(2)} ms · cache {lookups ? `${Math.round((100 * m.cache.hits) / lookups)}% hit` : '—'}
        </span>
      ) : (
        <span>no recommend requests yet</span>
      )}
    </p>
  )
}
