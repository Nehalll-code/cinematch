import { useState } from 'react'
import { Carousel } from '../components/Carousel'
import { FilmCard } from '../components/FilmCard'
import { Panel } from '../components/Panel'
import { Banner, EmptyState, QueryState } from '../components/QueryState'
import { FEED_SIZE, useRecommendations } from '../lib/queries'
import { useRequiredViewer } from '../lib/viewer'

const PERSONAL =
  'Rank-normalised score in [0, 1]: 1.0 means the model ranks this film first, out of every film you have not seen. It is a ranking score, not a predicted star rating.'
const POPULAR =
  'Popularity rank score in [0, 1]. Shown instead of a personal score because there is not enough history to personalise yet.'

function FeedSkeleton() {
  return (
    <div className="flex gap-4 overflow-hidden" aria-hidden="true">
      {Array.from({ length: 5 }, (_, i) => (
        <Panel key={i} as="div" className="w-52 shrink-0"><div className="aspect-[2/3] w-full" /><div className="h-40 solid" /></Panel>
      ))}
    </div>
  )
}

export function Feed() {
  const viewer = useRequiredViewer()
  const recs = useRecommendations(viewer)
  const [message, setMessage] = useState('')

  return (
    <section aria-labelledby="feed-heading" className="flex flex-col gap-4">
      <div>
        <h1 id="feed-heading" className="text-2xl font-semibold tracking-tight">Your feed</h1>
        <p className="text-sm text-muted">
          Top {FEED_SIZE} films you have not seen, ranked for you. Rating or dismissing a film removes it and refreshes the ranking.
        </p>
      </div>

      {viewer.kind === 'explore' && (
        <Banner>
          Exploring MovieLens user #{viewer.userId}. This view is read-only: rating, dismissing and
          watchlists belong to accounts. Use <strong>Leave</strong> (top right) to go back and sign in.
        </Banner>
      )}

      <p role="status" aria-live="polite" className="min-h-5 text-sm text-accent">{message}</p>

      <QueryState
        query={recs}
        label="recommendations"
        skeleton={<FeedSkeleton />}
        isEmpty={(d) => d.items.length === 0}
        empty={<EmptyState title="No recommendations right now">You have seen or dismissed everything the model can suggest.</EmptyState>}
      >
        {(data) => (
          <>
            {data.cold_start && (
              <Banner tone="warn">
                {viewer.kind === 'account'
                  ? 'You are new here, so this is the popular-films fallback, not a personal ranking. Rate a few films 4★ or higher and your feed will change.'
                  : 'This MovieLens user has too little history for a personal ranking, so the popular-films fallback is shown.'}
              </Banner>
            )}
            <Carousel label="Recommended films">
              {data.items.map((m) => (
                <FilmCard
                  key={m.movie_id}
                  movie={m}
                  viewer={viewer}
                  scoreLabel={data.cold_start ? POPULAR : PERSONAL}
                  onAction={setMessage}
                />
              ))}
            </Carousel>
            <p className="font-mono text-xs text-muted">
              {data.items.length} films · computed in {data.latency_ms} ms{data.cached ? ' (served from cache)' : ''}
            </p>
          </>
        )}
      </QueryState>
    </section>
  )
}
