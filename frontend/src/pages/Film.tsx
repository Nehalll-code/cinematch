import { lazy, Suspense } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Panel } from '../components/Panel'
import { Poster } from '../components/Poster'
import { EmptyState, QueryState } from '../components/QueryState'
import { READ_ONLY_HINT } from '../components/FilmCard'
import { bareTitle } from '../lib/api'
import { useMovie, useSimilar, useWatchlist, useWatchlistToggle } from '../lib/queries'
import { useRequiredViewer } from '../lib/viewer'

// d3-force + canvas code is only needed on this screen, so it is split out of the main bundle
const SimilarityGraph = lazy(() => import('../components/SimilarityGraph').then((m) => ({ default: m.SimilarityGraph })))

export function Film() {
  const { id } = useParams()
  const movieId = Number(id)
  const viewer = useRequiredViewer()
  const navigate = useNavigate()
  const movie = useMovie(movieId)
  const similar = useSimilar(movieId)
  const watchlist = useWatchlist(viewer)
  const toggle = useWatchlistToggle()
  const onList = watchlist.data?.some((w) => w.movie_id === movieId) ?? false

  if (!Number.isInteger(movieId) || movieId <= 0) {
    return <EmptyState title="That is not a film id">Go back to the <Link to="/browse" className="text-accent underline">search page</Link>.</EmptyState>
  }

  return (
    <section aria-labelledby="film-heading" className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <div className="flex flex-col gap-4">
        <QueryState query={movie} label="film details">
          {(m) => (
            <Panel>
              <div className="flex flex-col">
                <Poster movieId={m.movie_id} title={m.title} width={500} className="rounded-t-xl" />
                <div className="solid flex flex-col gap-3 p-4">
                  <h1 id="film-heading" className="text-2xl font-semibold leading-tight">{bareTitle(m.title)}</h1>
                  <p className="font-mono text-sm text-muted">{m.year ?? 'year unknown'} · MovieLens #{m.movie_id}</p>
                  <ul className="flex flex-wrap gap-1" aria-label="Genres">
                    {m.genres.map((g) => <li key={g} className="rounded border border-line px-2 py-0.5 text-xs text-muted">{g}</li>)}
                    {m.genres.length === 0 && <li className="text-xs text-muted">no genres listed</li>}
                  </ul>
                  <button
                    type="button"
                    aria-pressed={onList}
                    disabled={viewer.kind !== 'account' || toggle.isPending}
                    title={viewer.kind !== 'account' ? READ_ONLY_HINT : undefined}
                    onClick={() => toggle.mutate({ movieId: m.movie_id, on: !onList })}
                    className="interactive self-start rounded-lg border border-control-line bg-solid px-3 py-1.5 text-sm text-text disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {onList ? '✓ On your watchlist' : '+ Add to watchlist'}
                  </button>
                  {toggle.error && <p role="alert" className="text-sm text-danger">{toggle.error.message}</p>}
                </div>
              </div>
            </Panel>
          )}
        </QueryState>
      </div>

      <div className="flex min-w-0 flex-col gap-4">
        <div>
          <h2 className="text-lg font-medium">Similar films</h2>
          <p className="text-sm text-muted">
            Content similarity: cosine distance between TF-IDF vectors of genres, title words and release year.
            Edge width is the similarity itself. Click a film to recentre the graph on it.
          </p>
        </div>
        <QueryState
          query={similar}
          label="similar films"
          isEmpty={(d) => d.length === 0}
          empty={<EmptyState title="No similar films found" />}
        >
          {(neighbours) => (
            <>
              <Panel>
                <div className="solid rounded-xl p-2">
                  <Suspense fallback={<div role="status" className="grid h-[440px] place-items-center text-sm text-muted">Loading graph…</div>}>
                    <SimilarityGraph
                      key={movieId}
                      center={{ id: movieId, title: movie.data?.title ?? `Film ${movieId}` }}
                      neighbours={neighbours}
                      onSelect={(next) => navigate(`/film/${next}`)}
                    />
                  </Suspense>
                </div>
              </Panel>
              <ol className="grid gap-1 sm:grid-cols-2" aria-label="Similar films, most similar first">
                {neighbours.map((n) => (
                  <li key={n.movie_id}>
                    <Link
                      to={`/film/${n.movie_id}`}
                      className="interactive flex items-baseline justify-between gap-3 rounded-lg border border-line bg-solid px-3 py-2 text-sm"
                    >
                      <span className="truncate">{n.title}</span>
                      <span className="shrink-0 font-mono text-xs text-accent" title="cosine similarity">{n.score.toFixed(3)}</span>
                    </Link>
                  </li>
                ))}
              </ol>
            </>
          )}
        </QueryState>
      </div>
    </section>
  )
}
