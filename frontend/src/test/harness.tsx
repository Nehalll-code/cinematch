import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { vi } from 'vitest'
import { Shell } from '../components/Shell'
import { ViewerProvider } from '../lib/viewer'
import type { Viewer } from '../lib/types'
import { Browse } from '../pages/Browse'
import { Feed } from '../pages/Feed'
import { Film } from '../pages/Film'
import { Landing } from '../pages/Landing'
import { Taste } from '../pages/Taste'
import { useViewer } from '../lib/viewer'
import health from './fixtures/health.json'
import metrics from './fixtures/metrics.json'
import movie from './fixtures/movie.json'
import profile from './fixtures/profile.json'
import recommend from './fixtures/recommend.json'
import search from './fixtures/search.json'
import similar from './fixtures/similar.json'

/** Real responses recorded from the running API (scripts/record-fixtures.mjs). */
export const fixtures = { health, metrics, movie, profile, recommend, search, similar }

type Handler = (req: { method: string; path: string; body: unknown }) => unknown | Response
export type Calls = { method: string; path: string; body: unknown }[]

/** Replace global fetch with a router. Unlisted routes fail loudly so a test cannot pass by
 *  silently depending on a request it did not declare. */
export function mockApi(routes: Record<string, Handler>): Calls {
  const calls: Calls = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input).replace(/^\/api/, '')
      const method = init?.method ?? 'GET'
      const body = init?.body ? JSON.parse(String(init.body)) : undefined
      calls.push({ method, path, body })
      const key = Object.keys(routes).find((k) => `${method} ${path}`.startsWith(k))
      if (!key) throw new Error(`unmocked request: ${method} ${path}`)
      const out = routes[key]({ method, path, body })
      if (out instanceof Response) return out
      if (out === undefined) return new Response(null, { status: 204 })
      return new Response(JSON.stringify(out), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }),
  )
  return calls
}

export const baseRoutes: Record<string, Handler> = {
  'GET /health': () => health,
  'GET /metrics': () => metrics,
}

function Gate({ children }: { children: React.ReactNode }) {
  const { viewer } = useViewer()
  return viewer ? <>{children}</> : <Landing />
}

export function renderApp(path: string, viewer: Viewer | null) {
  sessionStorage.clear()
  if (viewer) sessionStorage.setItem('cinematch.viewer', JSON.stringify(viewer))
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 0 } } })
  return render(
    <QueryClientProvider client={client}>
      <ViewerProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route element={<Shell />}>
              <Route index element={<Gate><Feed /></Gate>} />
              <Route path="taste" element={<Gate><Taste /></Gate>} />
              <Route path="film/:id" element={<Gate><Film /></Gate>} />
              <Route path="browse" element={<Gate><Browse /></Gate>} />
            </Route>
          </Routes>
        </MemoryRouter>
      </ViewerProvider>
    </QueryClientProvider>,
  )
}

export const explorer: Viewer = { kind: 'explore', userId: 1 }
export const account: Viewer = { kind: 'account', token: 'test-token', email: 'ada@example.com' }
