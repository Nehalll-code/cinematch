import { useState } from 'react'

/** Five whole stars (1-5). The API accepts half-stars, but a five-button radio group is far easier
 *  to operate by keyboard and screen reader than a half-star slider; the resolution is a UI choice,
 *  not an API limit. */
export function StarRating({ label, disabled, value, onRate }: {
  label: string
  disabled?: boolean
  value?: number
  onRate: (stars: number) => void
}) {
  const [hover, setHover] = useState(0)
  const shown = hover || value || 0
  return (
    <div role="radiogroup" aria-label={label} className="flex" onMouseLeave={() => setHover(0)}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          role="radio"
          aria-checked={value === n}
          aria-label={`${n} star${n > 1 ? 's' : ''}`}
          disabled={disabled}
          onMouseEnter={() => setHover(n)}
          onFocus={() => setHover(n)}
          onBlur={() => setHover(0)}
          onClick={() => onRate(n)}
          className={`interactive rounded p-0.5 text-lg leading-none disabled:cursor-not-allowed disabled:opacity-60 ${
            n <= shown ? 'text-accent' : 'text-muted'
          }`}
        >
          <span aria-hidden="true">{n <= shown ? '★' : '☆'}</span>
        </button>
      ))}
    </div>
  )
}
