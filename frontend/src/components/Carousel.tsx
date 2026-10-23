import { useRef, type ReactNode } from 'react'

/** Horizontally scrolling, scroll-snapping row with previous/next buttons. Native scrolling keeps
 *  touch, trackpad, and keyboard behaviour for free; the buttons exist for mouse users and as a
 *  discoverable affordance. */
export function Carousel({ label, children }: { label: string; children: ReactNode }) {
  const ref = useRef<HTMLUListElement>(null)
  const scroll = (dir: -1 | 1) => {
    const el = ref.current
    if (!el) return
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    el.scrollBy({ left: dir * el.clientWidth * 0.8, behavior: reduce ? 'auto' : 'smooth' })
  }
  const btn =
    'interactive absolute top-1/2 z-10 -translate-y-1/2 rounded-full border border-control-line bg-solid px-3 py-2 text-text'
  return (
    <div className="relative">
      <button type="button" aria-label={`Scroll ${label} left`} onClick={() => scroll(-1)} className={`${btn} left-1`}>‹</button>
      <ul ref={ref} aria-label={label} className="scroll-x flex snap-x snap-mandatory gap-4 overflow-x-auto px-1 pb-3" role="list">
        {Array.isArray(children)
          ? children.map((c, i) => <li key={i} className="shrink-0 list-none">{c}</li>)
          : <li className="list-none">{children}</li>}
      </ul>
      <button type="button" aria-label={`Scroll ${label} right`} onClick={() => scroll(1)} className={`${btn} right-1`}>›</button>
    </div>
  )
}
