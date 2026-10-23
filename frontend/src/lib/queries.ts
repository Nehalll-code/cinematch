import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './api'
import type {
  Health,
  InteractionIn,
  InteractionOut,
  Metrics,
  Movie,
  MovieDetail,
  Recommendations,
  SearchHit,
  TasteProfile,
  Viewer,
  WatchlistEntry,
} from './types'

export const FEED_SIZE = 20
export const SIMILAR_SIZE = 12

/* Query keys carry the viewer so one viewer's data can never be served to another. */
const who = (v: Viewer) => (v.kind === 'account' ? 'me' : `ml-${v.userId}`)

export const useHealth = () =>
  useQuery({ queryKey: ['health'], queryFn: () => api<Health>('/health'), refetchInterval: 30_000 })

export const useMetrics = () =>
  useQuery({ queryKey: ['metrics'], queryFn: () => api<Metrics>('/metrics'), refetchInterval: 30_000 })

export function useRecommendations(v: Viewer) {
  return useQuery({
    queryKey: ['recs', who(v)],
    queryFn: () =>
      v.kind === 'account'
        ? api<Recommendations>(`/me/recommendations?k=${FEED_SIZE}`, { auth: true })
        : api<Recommendations>(`/recommend/${v.userId}?k=${FEED_SIZE}`),
  })
}

export function useTasteProfile(v: Viewer) {
  return useQuery({
    queryKey: ['profile', who(v)],
    queryFn: () =>
      v.kind === 'account' ? api<TasteProfile>('/me/profile', { auth: true }) : api<TasteProfile>(`/profile/${v.userId}`),
  })
}

export const useMovie = (id: number) =>
  useQuery({ queryKey: ['movie', id], queryFn: () => api<MovieDetail>(`/movies/${id}`) })

export const useSimilar = (id: number) =>
  useQuery({ queryKey: ['similar', id], queryFn: () => api<Movie[]>(`/similar/${id}?k=${SIMILAR_SIZE}`) })

export const useSearch = (q: string) =>
  useQuery({
    queryKey: ['search', q],
    queryFn: () => api<SearchHit[]>(`/search?q=${encodeURIComponent(q)}&limit=24`),
    enabled: q.trim().length >= 2,
  })

export function useWatchlist(v: Viewer) {
  return useQuery({
    queryKey: ['watchlist'],
    queryFn: () => api<WatchlistEntry[]>('/watchlist', { auth: true }),
    enabled: v.kind === 'account', // MovieLens users have no watchlist
  })
}

/** Rate / dismiss. Both change what the feed should contain (the film is now seen) and what the
 *  taste profile is built from, so both are invalidated rather than patched locally. */
export function useInteract() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: InteractionIn) => api<InteractionOut>('/interactions', { method: 'POST', json: body, auth: true }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['recs'] })
      void qc.invalidateQueries({ queryKey: ['profile'] })
    },
  })
}

export function useWatchlistToggle() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ movieId, on }: { movieId: number; on: boolean }) =>
      api<void>(`/watchlist/${movieId}`, { method: on ? 'PUT' : 'DELETE', auth: true }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['watchlist'] }),
  })
}
