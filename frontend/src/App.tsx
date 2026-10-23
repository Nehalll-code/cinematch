import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Shell } from './components/Shell'
import { ApiError } from './lib/api'
import { ViewerProvider, useViewer } from './lib/viewer'
import { Browse } from './pages/Browse'
import { Feed } from './pages/Feed'
import { Landing } from './pages/Landing'

// Taste pulls in Recharts and Film pulls in the graph; neither is needed to render the feed
const Taste = lazy(() => import('./pages/Taste').then((m) => ({ default: m.Taste })))
const Film = lazy(() => import('./pages/Film').then((m) => ({ default: m.Film })))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 15_000,
      // retry transient failures (network, 5xx) once; never retry a 4xx, which will not change
      retry: (count, err) => count < 1 && !(err instanceof ApiError && err.status >= 400 && err.status < 500),
      refetchOnWindowFocus: false,
    },
  },
})

/** Everything except the landing screen needs a viewer (an account, or a read-only MovieLens user). */
function RequireViewer({ children }: { children: React.ReactNode }) {
  const { viewer } = useViewer()
  return viewer ? <>{children}</> : <Landing />
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ViewerProvider>
        <BrowserRouter>
          <Suspense fallback={<p role="status" className="text-sm text-muted">Loading…</p>}>
          <Routes>
            <Route element={<Shell />}>
              <Route index element={<RequireViewer><Feed /></RequireViewer>} />
              <Route path="taste" element={<RequireViewer><Taste /></RequireViewer>} />
              <Route path="film/:id" element={<RequireViewer><Film /></RequireViewer>} />
              <Route path="browse" element={<RequireViewer><Browse /></RequireViewer>} />
              <Route path="*" element={<RequireViewer><Feed /></RequireViewer>} />
            </Route>
          </Routes>
          </Suspense>
        </BrowserRouter>
      </ViewerProvider>
    </QueryClientProvider>
  )
}
