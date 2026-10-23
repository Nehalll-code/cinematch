import { useState } from 'react'
import { bareTitle, yearOf } from '../lib/api'

/** Poster from our own /api/poster proxy (the TMDB key never reaches the browser). If there is no
 *  poster (no tmdbId, no TMDB key configured, TMDB has none, or the request fails) the card says so
 *  plainly and shows the real title, rather than a fake image. */
export function Poster({ movieId, title, width = 342, className = '' }: {
  movieId: number
  title: string
  width?: 185 | 342 | 500
  className?: string
}) {
  const [failed, setFailed] = useState(false)
  if (failed) {
    return (
      <div
        role="img"
        aria-label={`No poster available for ${title}`}
        className={`flex aspect-[2/3] flex-col justify-between bg-solid p-3 ${className}`}
      >
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted">no poster</span>
        <span className="text-sm font-medium leading-snug text-text">
          {bareTitle(title)}
          {yearOf(title) && <span className="block font-mono text-xs text-muted">{yearOf(title)}</span>}
        </span>
      </div>
    )
  }
  return (
    <img
      src={`/api/poster/${movieId}?w=${width}`}
      alt={`Poster for ${title}`}
      loading="lazy"
      onError={() => setFailed(true)}
      className={`aspect-[2/3] w-full bg-solid object-cover ${className}`}
    />
  )
}
