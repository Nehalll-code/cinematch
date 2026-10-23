import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from 'recharts'
import { Link } from 'react-router-dom'
import { Panel } from '../components/Panel'
import { Banner, EmptyState, QueryState } from '../components/QueryState'
import { useTasteProfile } from '../lib/queries'
import type { TasteProfile } from '../lib/types'
import { useRequiredViewer } from '../lib/viewer'

function TopGenres({ axes }: { axes: TasteProfile['axes'] }) {
  const top = [...axes].filter((a) => a.value > 0).sort((a, b) => b.value - a.value).slice(0, 6)
  const max = top[0]?.value ?? 1
  return (
    <ol className="flex flex-col gap-2" aria-label="Strongest genres">
      {top.map((a) => (
        <li key={a.genre} className="text-sm">
          <div className="flex justify-between">
            <span>{a.genre}</span>
            <span className="font-mono text-xs text-muted">{a.value.toFixed(3)}</span>
          </div>
          {/* bar length = this genre's profile weight relative to the strongest one */}
          <div className="mt-1 h-1.5 w-full rounded bg-bg" aria-hidden="true">
            <div className="h-1.5 rounded bg-accent" style={{ width: `${(100 * a.value) / max}%` }} />
          </div>
        </li>
      ))}
    </ol>
  )
}

export function Taste() {
  const viewer = useRequiredViewer()
  const profile = useTasteProfile(viewer)

  return (
    <section aria-labelledby="taste-heading" className="flex flex-col gap-4">
      <div>
        <h1 id="taste-heading" className="text-2xl font-semibold tracking-tight">Taste profile</h1>
        <p className="max-w-3xl text-sm text-muted">
          The content model&apos;s actual profile of {viewer.kind === 'account' ? 'you' : `MovieLens user #${viewer.userId}`}:
          the normalised mean TF-IDF vector of the films liked (rated 4★ or higher), read from the API, not computed in your browser.
        </p>
      </div>

      <QueryState
        query={profile}
        label="the taste profile"
        isEmpty={(d) => d.n_positive === 0}
        empty={
          <EmptyState title="No taste profile yet">
            A profile is built from films rated 4★ or higher, and there are none yet.{' '}
            <Link to="/" className="text-accent underline">Rate some films in your feed</Link> and come back.
          </EmptyState>
        }
      >
        {(data) => (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <Panel>
              <div className="solid rounded-xl p-4">
                <div role="img" aria-label={`Radar chart of genre weights: ${data.axes.filter((a) => a.value > 0).map((a) => `${a.genre} ${a.value.toFixed(2)}`).join(', ')}`} className="h-[420px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={data.axes} outerRadius="68%">
                      <PolarGrid stroke="var(--color-control-line)" />
                      <PolarAngleAxis dataKey="genre" tick={{ fill: 'var(--color-text)', fontSize: 11 }} />
                      <PolarRadiusAxis tick={{ fill: 'var(--color-muted)', fontSize: 10 }} stroke="var(--color-control-line)" />
                      <Radar name="Profile weight" dataKey="value" stroke="var(--color-accent)" fill="var(--color-accent)" fillOpacity={0.25} />
                      <Tooltip
                        formatter={(v) => [Number(v).toFixed(4), 'weight']}
                        contentStyle={{ background: 'var(--color-solid)', border: '1px solid var(--color-control-line)', color: 'var(--color-text)' }}
                        itemStyle={{ color: 'var(--color-text)' }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </Panel>

            <div className="flex flex-col gap-4">
              <Panel>
                <div className="solid flex flex-col gap-3 rounded-xl p-4">
                  <h2 className="font-medium">Built from {data.n_positive.toLocaleString()} liked films</h2>
                  <TopGenres axes={data.axes} />
                </div>
              </Panel>
              <Banner>
                Genre terms are {(100 * data.genre_mass).toFixed(0)}% of this profile vector; the remainder is title words and
                release years, which the radar leaves out. Weights are TF-IDF, so each liked film in a rarer genre
                adds more to the profile than each liked film in a common one. That is how this model reads
                taste, shown as it is.
              </Banner>
            </div>
          </div>
        )}
      </QueryState>
    </section>
  )
}
