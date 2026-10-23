import { useState, type FormEvent } from 'react'
import { Banner } from '../components/QueryState'
import { Panel } from '../components/Panel'
import { api, ApiError } from '../lib/api'
import { useHealth } from '../lib/queries'
import type { TasteProfile } from '../lib/types'
import { useViewer } from '../lib/viewer'

const field = 'w-full rounded-lg border border-control-line bg-solid px-3 py-2 text-text placeholder:text-muted'
const primary = 'interactive rounded-lg bg-accent px-4 py-2 font-medium text-bg disabled:cursor-not-allowed disabled:opacity-60'
const secondary = 'interactive rounded-lg border border-control-line bg-solid px-4 py-2 text-text disabled:opacity-60'

function AccountForm() {
  const { signIn, register } = useViewer()
  const [mode, setMode] = useState<'signin' | 'register'>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await (mode === 'signin' ? signIn : register)(email, password)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel>
      <form onSubmit={submit} className="solid flex flex-col gap-4 rounded-xl p-6" aria-labelledby="account-heading">
        <h2 id="account-heading" className="text-lg font-medium">{mode === 'signin' ? 'Sign in' : 'Create an account'}</h2>
        <p className="text-sm text-muted">
          A new account starts with no history, so you will see popular films until you rate a few.
          Ratings of 4★ or higher are what personalise your feed.
        </p>
        <label className="flex flex-col gap-1 text-sm">
          Email
          <input className={field} type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Password {mode === 'register' && <span className="text-muted">(8–128 characters)</span>}
          <input
            className={field} type="password" required minLength={mode === 'register' ? 8 : undefined} maxLength={128}
            autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
            value={password} onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {error && <p role="alert" className="text-sm text-danger">{error}</p>}
        <div className="flex flex-wrap gap-3">
          <button type="submit" disabled={busy} className={primary}>{busy ? 'Working…' : mode === 'signin' ? 'Sign in' : 'Create account'}</button>
          <button type="button" className={secondary} onClick={() => { setMode(mode === 'signin' ? 'register' : 'signin'); setError(null) }}>
            {mode === 'signin' ? 'Need an account?' : 'Have an account?'}
          </button>
        </div>
      </form>
    </Panel>
  )
}

function ExploreForm() {
  const { explore } = useViewer()
  const health = useHealth()
  const [id, setId] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function go(userId: number) {
    setBusy(true)
    setError(null)
    try {
      // confirm the user exists before switching modes (404 -> a clear message, not a broken feed)
      await api<TasteProfile>(`/profile/${userId}`)
      explore(userId)
    } catch (err) {
      setError(err instanceof ApiError && err.status === 404 ? `No MovieLens user #${userId}.` : (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const n = health.data?.n_users
  return (
    <Panel>
      <div className="solid flex flex-col gap-4 rounded-xl p-6">
        <h2 className="text-lg font-medium">Explore as a MovieLens user</h2>
        <p className="text-sm text-muted">
          Browse the feed and taste profile of one of the real users the model was trained on. Read-only:
          rating and watchlists need an account.
        </p>
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault()
            const v = Number(id)
            if (Number.isInteger(v) && v > 0) void go(v)
            else setError('Enter a positive whole number.')
          }}
        >
          <label className="flex flex-col gap-1 text-sm">
            User id{n ? ` (1–${n.toLocaleString()})` : ''}
            <input className={`${field} w-36`} inputMode="numeric" value={id} onChange={(e) => setId(e.target.value)} />
          </label>
          <button type="submit" disabled={busy || !id} className={primary}>Explore</button>
          <button
            type="button" className={secondary} disabled={busy || !n}
            onClick={() => void go(1 + Math.floor(Math.random() * (n ?? 1)))}
          >
            Random user
          </button>
        </form>
        {error && <p role="alert" className="text-sm text-danger">{error}</p>}
      </div>
    </Panel>
  )
}

export function Landing() {
  const { notice } = useViewer()
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Recommendations you can audit.</h1>
        <p className="mt-2 max-w-2xl text-muted">
          A hybrid recommender evaluated on a leakage-safe temporal split. Every score, bar, and node
          on these screens is read live from the API.
        </p>
      </div>
      {notice && <Banner tone="warn">{notice}</Banner>}
      <div className="grid gap-6 md:grid-cols-2">
        <AccountForm />
        <ExploreForm />
      </div>
    </div>
  )
}
