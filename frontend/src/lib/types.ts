/* Response shapes. Each mirrors a Pydantic model or a literal dict in the FastAPI service
   (src/cinematch/*.py); they are written from the code, not guessed. */

// schemas.py: Movie
export interface Movie {
  movie_id: number
  title: string
  genres: string[]
  score: number // /recommend, /me/recommendations: rank-normalised score in [0,1]; /similar: cosine similarity
}

// schemas.py: Recommendations
export interface Recommendations {
  user_id: number
  cold_start: boolean // true => popularity fallback, not a personalised ranking
  latency_ms: number
  cached: boolean
  items: Movie[]
}

// catalog.py: MovieDetail / SearchHit
export interface MovieDetail {
  movie_id: number
  title: string
  year: number | null
  genres: string[]
  poster_available: boolean
}
export interface SearchHit {
  movie_id: number
  title: string
  year: number | null
  genres: string[]
}

// catalog.py: TasteProfile
export interface TasteProfile {
  user_id: number
  n_positive: number
  genre_mass: number
  axes: { genre: string; value: number }[]
}

// accounts.py
export interface Account {
  id: number
  email: string
}
export interface Token {
  access_token: string
  token_type: string
  expires_in: number
}

// user_data.py
export type InteractionType = 'rating' | 'watch' | 'dismiss'
export interface InteractionIn {
  movie_id: number
  type: InteractionType
  rating?: number
}
export interface InteractionOut {
  movie_id: number
  type: InteractionType
  rating: number | null
  created_at: string
}
export interface WatchlistEntry {
  movie_id: number
  title: string
  added_at: string
}

// api.py: /health and /metrics (literal dicts)
export interface Health {
  status: 'ok' | 'degraded'
  model_loaded: boolean
  db_configured: boolean
  index_verified: boolean
  n_users: number
  n_items: number
}
export type Metrics =
  | { requests: 0 }
  | {
      requests: number
      p50_ms: number
      p95_ms: number
      p99_ms: number
      cache: { hits: number; misses: number; maxsize: number; currsize: number }
    }

/** Who is looking at the app: a registered account (full write path) or a read-only MovieLens user. */
export type Viewer =
  | { kind: 'account'; token: string; email: string }
  | { kind: 'explore'; userId: number }
