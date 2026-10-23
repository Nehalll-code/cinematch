import { cleanup, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { account, baseRoutes, explorer, fixtures, mockApi, renderApp } from './harness'

// jsdom has no canvas, so the force graph is replaced by a stub; everything it would draw is
// still listed as links beside it, and those are what is asserted.
vi.mock('../components/SimilarityGraph', () => ({
  SimilarityGraph: ({ neighbours }: { neighbours: unknown[] }) => <div data-testid="graph">{neighbours.length} nodes</div>,
}))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const rec = fixtures.recommend

describe('Feed', () => {
  it('shows a skeleton while loading, then every film from the API with title, year, genres and score', async () => {
    mockApi({ ...baseRoutes, 'GET /recommend/1': () => rec })
    renderApp('/', explorer)
    expect(document.querySelector('[aria-busy="true"]')).toBeInTheDocument() // the loading state
    expect(await screen.findAllByRole('article')).toHaveLength(rec.items.length)
    for (const m of rec.items) {
      const card = screen.getByRole('article', { name: m.title })
      expect(within(card).getByText(m.score.toFixed(4))).toBeInTheDocument()
      for (const g of m.genres.slice(0, 3)) expect(within(card).getByText(g)).toBeInTheDocument()
    }
    const first = screen.getByRole('article', { name: rec.items[0].title })
    expect(within(first).getByText(rec.items[0].title.match(/\((\d{4})\)/)![1])).toBeInTheDocument() // year parsed from the title
  })

  it('is read-only when exploring a MovieLens user: controls are disabled and explained', async () => {
    mockApi({ ...baseRoutes, 'GET /recommend/1': () => rec })
    renderApp('/', explorer)
    const card = await screen.findByRole('article', { name: rec.items[0].title })
    expect(within(card).getByRole('button', { name: /Dismiss/ })).toBeDisabled()
    expect(within(card).getByRole('radio', { name: '5 stars' })).toBeDisabled()
    expect(screen.getByText(/read-only/i, { selector: '[role=note]' })).toBeInTheDocument()
  })

  it('shows an error with a working retry when the API fails', async () => {
    let fail = true
    mockApi({
      ...baseRoutes,
      'GET /recommend/1': () => (fail ? new Response(JSON.stringify({ detail: 'model artifact not loaded' }), { status: 503 }) : rec),
    })
    renderApp('/', explorer)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('model artifact not loaded')
    fail = false
    await userEvent.click(within(alert).getByRole('button', { name: 'Try again' }))
    expect(await screen.findAllByRole('article')).toHaveLength(rec.items.length)
  })

  it('says plainly when the API is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('failed to fetch') }))
    renderApp('/', explorer)
    expect((await screen.findAllByText(/unreachable|Cannot reach/i)).length).toBeGreaterThan(0)
  })

  it('explains the cold-start fallback instead of presenting popularity as personal', async () => {
    mockApi({ ...baseRoutes, 'GET /me/recommendations': () => ({ ...rec, cold_start: true }), 'GET /watchlist': () => [] })
    renderApp('/', account)
    expect(await screen.findByText(/popular-films fallback, not a personal ranking/)).toBeInTheDocument()
  })

  it('rating posts to /interactions as the account and re-fetches the feed', async () => {
    const calls = mockApi({
      ...baseRoutes,
      'GET /me/recommendations': () => rec,
      'GET /watchlist': () => [],
      'POST /interactions': ({ body }) => ({ ...(body as object), rating: 5, created_at: '2026-01-01T00:00:00Z' }),
    })
    renderApp('/', account)
    const first = rec.items[0]
    const card = await screen.findByRole('article', { name: first.title })
    const before = calls.filter((c) => c.path.startsWith('/me/recommendations')).length
    await userEvent.click(within(card).getByRole('radio', { name: '5 stars' }))
    await waitFor(() =>
      expect(calls).toContainEqual({ method: 'POST', path: '/interactions', body: { movie_id: first.movie_id, type: 'rating', rating: 5 } }),
    )
    await waitFor(() => expect(calls.filter((c) => c.path.startsWith('/me/recommendations')).length).toBeGreaterThan(before))
    expect(await screen.findByText(/Rated .* 5★/)).toBeInTheDocument()
  })

  it('dismiss posts a dismiss interaction', async () => {
    const calls = mockApi({
      ...baseRoutes, 'GET /me/recommendations': () => rec, 'GET /watchlist': () => [],
      'POST /interactions': ({ body }) => ({ ...(body as object), rating: null, created_at: '2026-01-01T00:00:00Z' }),
    })
    renderApp('/', account)
    const card = await screen.findByRole('article', { name: rec.items[1].title })
    await userEvent.click(within(card).getByRole('button', { name: /Dismiss/ }))
    await waitFor(() =>
      expect(calls).toContainEqual({ method: 'POST', path: '/interactions', body: { movie_id: rec.items[1].movie_id, type: 'dismiss' } }),
    )
  })

  it('adding to the watchlist PUTs and flips the button state from the server list', async () => {
    let list: { movie_id: number; title: string; added_at: string }[] = []
    const calls = mockApi({
      ...baseRoutes, 'GET /me/recommendations': () => rec,
      'GET /watchlist': () => list,
      'PUT /watchlist/': ({ path }) => {
        const id = Number(path.split('/').pop())
        list = [{ movie_id: id, title: 'x', added_at: '2026-01-01T00:00:00Z' }]
        return undefined
      },
    })
    renderApp('/', account)
    const card = await screen.findByRole('article', { name: rec.items[0].title })
    await userEvent.click(within(card).getByRole('button', { name: '+ Watchlist' }))
    await waitFor(() => expect(calls.some((c) => c.method === 'PUT' && c.path === `/watchlist/${rec.items[0].movie_id}`)).toBe(true))
    expect(await within(card).findByRole('button', { name: /On watchlist/ })).toHaveAttribute('aria-pressed', 'true')
  })
})

describe('Session', () => {
  it('shows sign-in and explore options when there is no viewer', async () => {
    mockApi({ ...baseRoutes })
    renderApp('/', null)
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /Explore as a MovieLens user/ })).toBeInTheDocument()
  })

  it('surfaces the API message when sign-in fails', async () => {
    mockApi({ ...baseRoutes, 'POST /auth/login': () => new Response(JSON.stringify({ detail: 'invalid credentials' }), { status: 401 }) })
    renderApp('/', null)
    await userEvent.type(await screen.findByLabelText('Email'), 'ada@example.com')
    await userEvent.type(screen.getByLabelText('Password'), 'wrong password')
    await userEvent.click(screen.getAllByRole('button', { name: 'Sign in' })[0])
    expect(await screen.findByRole('alert')).toHaveTextContent('invalid credentials')
  })

  it('refuses an unknown MovieLens user with a clear message', async () => {
    mockApi({ ...baseRoutes, 'GET /profile/99999': () => new Response(JSON.stringify({ detail: 'unknown user 99999' }), { status: 404 }) })
    renderApp('/', null)
    await userEvent.type(await screen.findByLabelText(/User id/), '99999')
    await userEvent.click(screen.getByRole('button', { name: 'Explore' }))
    expect(await screen.findByText('No MovieLens user #99999.')).toBeInTheDocument()
  })

  it('an expired token (401 on a protected call) signs the user out with an explanation', async () => {
    mockApi({
      ...baseRoutes,
      'GET /me/recommendations': () => new Response(JSON.stringify({ detail: 'invalid or expired token' }), { status: 401 }),
      'GET /watchlist': () => new Response(JSON.stringify({ detail: 'invalid or expired token' }), { status: 401 }),
    })
    renderApp('/', account)
    expect(await screen.findByText(/session expired/i)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
  })
})

describe('Taste profile', () => {
  it('renders the count, genre share and strongest genres straight from the endpoint', async () => {
    mockApi({ ...baseRoutes, 'GET /profile/1': () => fixtures.profile })
    renderApp('/taste', explorer)
    const p = fixtures.profile
    expect(await screen.findByText(`Built from ${p.n_positive} liked films`)).toBeInTheDocument()
    expect(screen.getByText(new RegExp(`Genre terms are ${Math.round(100 * p.genre_mass)}%`))).toBeInTheDocument()
    const top = [...p.axes].sort((a, b) => b.value - a.value)[0]
    const list = screen.getByRole('list', { name: 'Strongest genres' })
    expect(within(list).getAllByRole('listitem')[0]).toHaveTextContent(top.genre)
    expect(within(list).getAllByRole('listitem')[0]).toHaveTextContent(top.value.toFixed(3))
  })

  it('has a real empty state for a user with no liked films', async () => {
    mockApi({ ...baseRoutes, 'GET /me/profile': () => ({ user_id: 7, n_positive: 0, genre_mass: 0, axes: fixtures.profile.axes.map((a) => ({ ...a, value: 0 })) }) })
    renderApp('/taste', account)
    expect(await screen.findByText('No taste profile yet')).toBeInTheDocument()
  })
})

describe('Film detail', () => {
  it('shows metadata and lists every similar film with its similarity', async () => {
    mockApi({ ...baseRoutes, 'GET /movies/1': () => fixtures.movie, 'GET /similar/1': () => fixtures.similar, 'GET /watchlist': () => [] })
    renderApp('/film/1', account)
    expect(await screen.findByRole('heading', { level: 1, name: 'Toy Story' })).toBeInTheDocument()
    const list = await screen.findByRole('list', { name: /Similar films/ })
    expect(within(list).getAllByRole('listitem')).toHaveLength(fixtures.similar.length)
    expect(within(list).getAllByRole('listitem')[0]).toHaveTextContent(fixtures.similar[0].score.toFixed(3))
    expect(await screen.findByTestId('graph')).toHaveTextContent(`${fixtures.similar.length} nodes`)
  })

  it('shows an error state for a film that does not exist', async () => {
    mockApi({ ...baseRoutes, 'GET /movies/9': () => new Response(JSON.stringify({ detail: 'unknown movie 9' }), { status: 404 }), 'GET /similar/9': () => new Response(JSON.stringify({ detail: 'unknown movie 9' }), { status: 404 }), 'GET /watchlist': () => [] })
    renderApp('/film/9', account)
    expect((await screen.findAllByRole('alert'))[0]).toHaveTextContent('unknown movie 9')
  })
})

describe('Search & watchlist', () => {
  it('searches after a pause and lists real results', async () => {
    mockApi({ ...baseRoutes, 'GET /search': () => fixtures.search, 'GET /watchlist': () => [] })
    renderApp('/browse', account)
    await userEvent.type(await screen.findByLabelText('Title'), 'toy story')
    const results = await screen.findByRole('list', { name: 'Search results' })
    expect(within(results).getAllByRole('listitem')).toHaveLength(fixtures.search.length)
  })

  it('has an empty state for no matches and for an empty watchlist', async () => {
    mockApi({ ...baseRoutes, 'GET /search': () => [], 'GET /watchlist': () => [] })
    renderApp('/browse', account)
    expect(await screen.findByText('Your watchlist is empty')).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('Title'), 'zzzzzz')
    expect(await screen.findByText(/No films match/)).toBeInTheDocument()
  })

  it('explains that watchlists belong to accounts for an explored user', async () => {
    mockApi({ ...baseRoutes })
    renderApp('/browse', explorer)
    expect(await screen.findByText('Watchlists belong to accounts')).toBeInTheDocument()
  })
})
