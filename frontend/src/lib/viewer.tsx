import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api, configureApi } from './api'
import type { Account, Token, Viewer } from './types'

/* Session handling.
   The JWT lives in sessionStorage: it disappears when the tab closes and is never sent anywhere but
   our own /api. Be clear about what that is and is not: any script running in the page can read it,
   so an XSS bug would leak it. The production answer is an httpOnly, SameSite cookie, which needs CSRF
   protection and server changes; refresh tokens and rotation are out of scope (see README). */

const KEY = 'cinematch.viewer'

function load(): Viewer | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as Viewer) : null
  } catch {
    return null
  }
}

function store(v: Viewer | null) {
  try {
    if (v) sessionStorage.setItem(KEY, JSON.stringify(v))
    else sessionStorage.removeItem(KEY)
  } catch {
    /* storage can be unavailable (private mode); the session then simply lasts until reload */
  }
}

interface Ctx {
  viewer: Viewer | null
  notice: string | null
  signIn: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  explore: (userId: number) => void
  signOut: () => void
  clearNotice: () => void
}

const ViewerContext = createContext<Ctx | null>(null)

export function ViewerProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const [viewer, setViewer] = useState<Viewer | null>(load)
  const [notice, setNotice] = useState<string | null>(null)

  // The API client must know the token and the 401 handler BEFORE any child fires its first request.
  // A plain useEffect is too late: children's effects run before their parent's, so after a reload
  // the first feed request would go out with no token and a 401 would be ignored. A parent's LAYOUT
  // effect runs before any child's passive effect, which is when TanStack Query starts fetching.
  const viewerRef = useRef<Viewer | null>(viewer)
  const expiredRef = useRef<() => void>(() => {})
  useLayoutEffect(() => {
    configureApi({
      getToken: () => (viewerRef.current?.kind === 'account' ? viewerRef.current.token : null),
      onUnauthorized: () => expiredRef.current(),
    })
  }, [])

  const set = useCallback(
    (v: Viewer | null) => {
      viewerRef.current = v
      store(v)
      setViewer(v)
      qc.clear() // never show one viewer's cached feed to the next
    },
    [qc],
  )
  useLayoutEffect(() => {
    expiredRef.current = () => {
      set(null)
      setNotice('Your session expired. Please sign in again.')
    }
  }, [set])

  const signIn = useCallback(
    async (email: string, password: string) => {
      const t = await api<Token>('/auth/login', { method: 'POST', json: { email, password } })
      // validate the token and fetch the canonical email in one call
      const me = await api<Account>('/me', { headers: { Authorization: `Bearer ${t.access_token}` } })
      setNotice(null)
      set({ kind: 'account', token: t.access_token, email: me.email })
    },
    [set],
  )

  const value = useMemo<Ctx>(
    () => ({
      viewer,
      notice,
      signIn,
      clearNotice: () => setNotice(null),
      async register(email, password) {
        await api<Account>('/auth/register', { method: 'POST', json: { email, password } })
        await signIn(email, password)
      },
      explore(userId) {
        setNotice(null)
        set({ kind: 'explore', userId })
      },
      signOut() {
        set(null)
      },
    }),
    [viewer, notice, set, signIn],
  )

  return <ViewerContext.Provider value={value}>{children}</ViewerContext.Provider>
}

export function useViewer(): Ctx {
  const ctx = useContext(ViewerContext)
  if (!ctx) throw new Error('useViewer must be used inside <ViewerProvider>')
  return ctx
}

/** For pages behind the route guard, where a viewer always exists. */
export function useRequiredViewer(): Viewer {
  const { viewer } = useViewer()
  if (!viewer) throw new Error('this page requires a viewer; it must render inside <RequireViewer>')
  return viewer
}
