import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { READ_ONLY_HINT } from '../components/FilmCard'
import { Panel } from '../components/Panel'
import { Poster } from '../components/Poster'
import { EmptyState, QueryState } from '../components/QueryState'
import { bareTitle } from '../lib/api'
import { useSearch, useWatchlist, useWatchlistToggle } from '../lib/queries'
import { useRequiredViewer } from '../lib/viewer'

/** Debounce so typing does not fire a request per keystroke. */
function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

function Search() {
  const viewer = useRequiredViewer()
  const [text, setText] = useState('')
  const q = useDebounced(text.trim(), 250)
  const search = useSearch(q)
  const watchlist = useWatchlist(viewer)
  const toggle = useWatchlistToggle()
  const onList = (id: number) => watchlist.data?.some((w) => w.movie_id === id) ?? false

  return (
    <Panel>
      <div className="solid flex flex-col gap-4 rounded-xl p-5">
        <h2 className="text-lg font-medium">Search films</h2>
        <label className="flex flex-col gap-1 text-sm">
          Title
          <input
            type="search"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. Heat, Casablanca, 1995"
            className="rounded-lg border border-control-line bg-solid px-3 py-2 text-text placeholder:text-muted"
          />
        </label>
        {q.length < 2 ? (
          <p className="text-sm text-muted">Type at least two characters.</p>
        ) : (
          <QueryState
            query={search}
            label="search results"
            isEmpty={(d) => d.length === 0}
            empty={<EmptyState title={`No films match “${q}”`}>Titles include the year, so try part of a name or a year.</EmptyState>}
          >
            {(hits) => (
              <ul className="flex flex-col gap-2" aria-label="Search results">
                {hits.map((h) => (
                  <li key={h.movie_id} className="flex items-center gap-3 rounded-lg border border-line p-2">
                    <div className="w-12 shrink-0 overflow-hidden rounded"><Poster movieId={h.movie_id} title={h.title} width={185} /></div>
                    <div className="min-w-0 flex-1">
                      <Link to={`/film/${h.movie_id}`} className="glow-focus block truncate text-sm font-medium hover:text-accent">{bareTitle(h.title)}</Link>
                      <p className="truncate font-mono text-xs text-muted">{h.year ?? '—'} · {h.genres.join(', ') || 'no genres'}</p>
                    </div>
                    <button
                      type="button"
                      aria-pressed={onList(h.movie_id)}
                      aria-label={`${onList(h.movie_id) ? 'Remove' : 'Add'} ${h.title} ${onList(h.movie_id) ? 'from' : 'to'} watchlist`}
                      disabled={viewer.kind !== 'account' || toggle.isPending}
                      title={viewer.kind !== 'account' ? READ_ONLY_HINT : undefined}
                      onClick={() => toggle.mutate({ movieId: h.movie_id, on: !onList(h.movie_id) })}
                      className="interactive shrink-0 rounded-lg border border-control-line bg-solid px-2 py-1 text-xs text-text disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {onList(h.movie_id) ? '✓ Listed' : '+ Watchlist'}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </QueryState>
        )}
        {toggle.error && <p role="alert" className="text-sm text-danger">{toggle.error.message}</p>}
      </div>
    </Panel>
  )
}

function Watchlist() {
  const viewer = useRequiredViewer()
  const watchlist = useWatchlist(viewer)
  const toggle = useWatchlistToggle()

  return (
    <Panel>
      <div className="solid flex flex-col gap-4 rounded-xl p-5">
        <h2 className="text-lg font-medium">Your watchlist</h2>
        {viewer.kind !== 'account' ? (
          <EmptyState title="Watchlists belong to accounts">
            You are exploring a MovieLens user, which is read-only. Leave and create an account to keep a watchlist.
          </EmptyState>
        ) : (
          <QueryState
            query={watchlist}
            label="your watchlist"
            isEmpty={(d) => d.length === 0}
            empty={<EmptyState title="Your watchlist is empty">Search for a film and press “+ Watchlist”, or add one from your feed.</EmptyState>}
          >
            {(items) => (
              <ul className="flex flex-col gap-2" aria-label="Watchlist">
                {items.map((w) => (
                  <li key={w.movie_id} className="flex items-center gap-3 rounded-lg border border-line p-2">
                    <div className="w-12 shrink-0 overflow-hidden rounded"><Poster movieId={w.movie_id} title={w.title} width={185} /></div>
                    <div className="min-w-0 flex-1">
                      <Link to={`/film/${w.movie_id}`} className="glow-focus block truncate text-sm font-medium hover:text-accent">{bareTitle(w.title)}</Link>
                      <p className="font-mono text-xs text-muted">added {new Date(w.added_at).toLocaleDateString()}</p>
                    </div>
                    <button
                      type="button"
                      disabled={toggle.isPending}
                      aria-label={`Remove ${w.title} from watchlist`}
                      onClick={() => toggle.mutate({ movieId: w.movie_id, on: false })}
                      className="interactive shrink-0 rounded-lg border border-control-line bg-solid px-2 py-1 text-xs text-text disabled:opacity-60"
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </QueryState>
        )}
      </div>
    </Panel>
  )
}

export function Browse() {
  return (
    <section aria-labelledby="browse-heading" className="flex flex-col gap-4">
      <h1 id="browse-heading" className="text-2xl font-semibold tracking-tight">Search &amp; watchlist</h1>
      <div className="grid gap-6 lg:grid-cols-2">
        <Search />
        <Watchlist />
      </div>
    </section>
  )
}
