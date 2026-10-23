import { Link } from 'react-router-dom'
import { bareTitle, yearOf } from '../lib/api'
import { useInteract, useWatchlist, useWatchlistToggle } from '../lib/queries'
import type { Movie, Viewer } from '../lib/types'
import { Panel } from './Panel'
import { Poster } from './Poster'
import { StarRating } from './StarRating'

export const READ_ONLY_HINT = 'Read-only: you are exploring a MovieLens user. Create an account to rate films.'

/** One recommended film with its real data (poster, title, year, genres, score) and the
 *  rate / dismiss / watchlist controls. Controls write via POST /interactions and PUT|DELETE
 *  /watchlist, then the relevant queries are invalidated by the mutation hooks. */
export function FilmCard({ movie, viewer, scoreLabel, onAction }: {
  movie: Movie
  viewer: Viewer
  scoreLabel: string
  onAction?: (message: string) => void
}) {
  const readOnly = viewer.kind !== 'account'
  const interact = useInteract()
  const toggle = useWatchlistToggle()
  const watchlist = useWatchlist(viewer)
  const onList = watchlist.data?.some((w) => w.movie_id === movie.movie_id) ?? false
  const year = yearOf(movie.title)
  const name = bareTitle(movie.title)
  const error = interact.error?.message ?? toggle.error?.message

  return (
    <Panel as="article" className="w-52 shrink-0 snap-start" aria-label={movie.title}>
      <div className="flex h-full flex-col">
        <Link to={`/film/${movie.movie_id}`} className="glow-focus block rounded-t-xl" aria-label={`Open ${movie.title}`}>
          <Poster movieId={movie.movie_id} title={movie.title} width={342} />
        </Link>
        <div className="solid flex flex-1 flex-col gap-2 p-3">
          <div>
            <h3 className="line-clamp-2 text-sm font-medium leading-snug text-text">{name}</h3>
            <p className="mt-0.5 font-mono text-xs text-muted">{year ?? '—'}</p>
          </div>
          <ul className="flex flex-wrap gap-1" aria-label="Genres">
            {movie.genres.slice(0, 3).map((g) => (
              <li key={g} className="rounded border border-line px-1.5 py-0.5 text-[11px] text-muted">{g}</li>
            ))}
          </ul>
          <p className="font-mono text-xs text-muted" title={scoreLabel}>
            score <span className="text-accent">{movie.score.toFixed(4)}</span>
          </p>

          <div className="mt-auto flex flex-col gap-2 border-t border-line pt-2" title={readOnly ? READ_ONLY_HINT : undefined}>
            <StarRating
              label={`Rate ${movie.title}`}
              disabled={readOnly || interact.isPending}
              onRate={(n) =>
                interact.mutate(
                  { movie_id: movie.movie_id, type: 'rating', rating: n },
                  { onSuccess: () => onAction?.(`Rated ${name} ${n}★. Your feed was refreshed.`) },
                )
              }
            />
            <div className="flex gap-2">
              <button
                type="button"
                disabled={readOnly || toggle.isPending}
                aria-pressed={onList}
                onClick={() =>
                  toggle.mutate(
                    { movieId: movie.movie_id, on: !onList },
                    { onSuccess: () => onAction?.(onList ? `Removed ${name} from your watchlist.` : `Added ${name} to your watchlist.`) },
                  )
                }
                className="interactive flex-1 rounded-lg border border-control-line bg-solid px-2 py-1 text-xs text-text disabled:cursor-not-allowed disabled:opacity-60"
              >
                {onList ? '✓ On watchlist' : '+ Watchlist'}
              </button>
              <button
                type="button"
                disabled={readOnly || interact.isPending}
                aria-label={`Dismiss ${movie.title}`}
                onClick={() =>
                  interact.mutate(
                    { movie_id: movie.movie_id, type: 'dismiss' },
                    { onSuccess: () => onAction?.(`Dismissed ${name}. It won't be recommended again.`) },
                  )
                }
                className="interactive rounded-lg border border-control-line bg-solid px-2 py-1 text-xs text-text disabled:cursor-not-allowed disabled:opacity-60"
              >
                Dismiss
              </button>
            </div>
            {error && <p role="alert" className="text-xs text-danger">{error}</p>}
          </div>
        </div>
      </div>
    </Panel>
  )
}
