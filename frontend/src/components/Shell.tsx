import { NavLink, Outlet } from 'react-router-dom'
import { useViewer } from '../lib/viewer'
import { SystemStatus } from './SystemStatus'

const link = ({ isActive }: { isActive: boolean }) =>
  `interactive rounded-lg border px-3 py-1.5 text-sm ${
    isActive ? 'border-accent text-accent' : 'border-transparent text-text hover:text-accent'
  }`

export function Shell() {
  const { viewer, signOut } = useViewer()
  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6">
      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="font-mono text-lg tracking-[0.25em] text-accent" aria-label="CineMatch">CINEMATCH</p>
          {viewer && (
            <nav aria-label="Main" className="flex flex-wrap gap-1">
              <NavLink to="/" end className={link}>Feed</NavLink>
              <NavLink to="/taste" className={link}>Taste profile</NavLink>
              <NavLink to="/browse" className={link}>Search &amp; watchlist</NavLink>
            </nav>
          )}
          {viewer && (
            <div className="flex items-center gap-3 text-sm">
              <span className="font-mono text-xs text-muted">
                {viewer.kind === 'account' ? viewer.email : `MovieLens user #${viewer.userId} · read-only`}
              </span>
              <button
                type="button"
                onClick={signOut}
                className="interactive rounded-lg border border-control-line bg-solid px-3 py-1.5 text-text"
              >
                {viewer.kind === 'account' ? 'Sign out' : 'Leave'}
              </button>
            </div>
          )}
        </div>
        <SystemStatus />
      </header>
      <main className="flex-1"><Outlet /></main>
      <footer className="border-t border-line pt-4 text-xs text-muted">
        <p>
          This product uses the TMDB API but is not endorsed or certified by TMDB. Film data and
          ratings: GroupLens MovieLens (ml-1m).
        </p>
      </footer>
    </div>
  )
}
