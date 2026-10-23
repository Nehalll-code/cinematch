import type { ReactNode } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'
import { ApiError } from '../lib/api'

/** Every query renders exactly one of: loading, error (with retry), empty, or data. Loading uses
 *  static skeleton blocks: the brief rules out ambient animation, and a pulsing placeholder is one. */
export function QueryState<T>({
  query, label, isEmpty, empty, skeleton, children,
}: {
  query: UseQueryResult<T>
  label: string // what is loading, e.g. "recommendations"
  isEmpty?: (data: T) => boolean
  empty?: ReactNode
  skeleton?: ReactNode
  children: (data: T) => ReactNode
}) {
  if (query.isPending) {
    return (
      <div role="status" aria-live="polite" aria-busy="true">
        <span className="sr-only">Loading {label}…</span>
        {skeleton ?? <div className="h-24 rounded-xl bg-glass border border-line" aria-hidden="true" />}
      </div>
    )
  }
  if (query.isError) {
    const err = query.error
    const offline = err instanceof ApiError && err.status === 0
    return (
      <div role="alert" className="solid rounded-xl border border-control-line p-4">
        <p className="font-medium text-danger">{offline ? 'The CineMatch API is unreachable' : `Could not load ${label}`}</p>
        <p className="mt-1 text-sm text-muted">{err.message}</p>
        <button
          type="button"
          onClick={() => void query.refetch()}
          className="interactive mt-3 rounded-lg border border-control-line bg-solid px-3 py-1.5 text-sm text-text"
        >
          Try again
        </button>
      </div>
    )
  }
  if (isEmpty?.(query.data)) return <>{empty}</>
  return <>{children(query.data)}</>
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="solid rounded-xl border border-line p-6 text-center">
      <p className="font-medium text-text">{title}</p>
      {children && <div className="mt-2 text-sm text-muted">{children}</div>}
    </div>
  )
}

export function Banner({ tone = 'info', children }: { tone?: 'info' | 'warn'; children: ReactNode }) {
  return (
    <div
      role="note"
      className={`solid rounded-xl border px-4 py-3 text-sm ${tone === 'warn' ? 'border-accent-2' : 'border-line'}`}
    >
      {children}
    </div>
  )
}
