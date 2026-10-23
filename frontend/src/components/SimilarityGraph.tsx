import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d'
import { bareTitle, yearOf } from '../lib/api'
import type { Movie } from '../lib/types'

/* Force-directed graph of a film and its most similar films.

   Everything drawn is data from /similar: one node per returned film, and one edge from the centre
   film to each neighbour whose width is that neighbour's real cosine similarity. There are NO edges
   between neighbours because the endpoint does not provide them, and inventing them would be
   fabricating data. Clicking a neighbour recentres the graph on it (the page re-fetches /similar for
   the new centre). A canvas is invisible to screen readers, so the same films are also listed as
   links beside it. */

interface GNode { id: number; label: string; sim: number; center: boolean }
interface GLink { source: number; target: number; sim: number }

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

export function SimilarityGraph({ center, neighbours, onSelect }: {
  center: { id: number; title: string }
  neighbours: Movie[]
  onSelect: (movieId: number) => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const graph = useRef<ForceGraphMethods<GNode, GLink> | undefined>(undefined)
  const [size, setSize] = useState({ w: 600, h: 440 })
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.max(280, Math.floor(e.contentRect.width)), h: 440 }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const data = useMemo(() => {
    const nodes: GNode[] = [
      { id: center.id, label: center.title, sim: 1, center: true },
      ...neighbours.map((m) => ({ id: m.movie_id, label: m.title, sim: m.score, center: false })),
    ]
    const links: GLink[] = neighbours.map((m) => ({ source: center.id, target: m.movie_id, sim: m.score }))
    return { nodes, links }
  }, [center, neighbours])

  const maxSim = Math.max(...neighbours.map((m) => m.score), 1e-6)

  // more similar films sit closer to the centre: the link force's distance is set from the data
  useEffect(() => {
    graph.current?.d3Force('link')?.distance((l: GLink) => 150 - 90 * (l.sim / maxSim))
    graph.current?.d3ReheatSimulation()
  }, [data, maxSim])
  const colors = { accent: css('--color-accent'), violet: css('--color-accent-2'), text: css('--color-text'), line: css('--color-control-line') }

  return (
    <div ref={box} className="w-full" style={{ height: size.h }} role="presentation">
      <ForceGraph2D
        ref={graph}
        width={size.w}
        height={size.h}
        graphData={data}
        backgroundColor="rgba(0,0,0,0)"
        cooldownTicks={80}
        linkWidth={(l: unknown) => 0.6 + 3.4 * ((l as GLink).sim / maxSim)}
        linkColor={() => colors.line}
        nodeRelSize={1}
        onNodeHover={(n) => setHover(n ? (n as GNode).id : null)}
        onNodeClick={(n) => { if (!(n as GNode).center) onSelect((n as GNode).id) }}
        nodeCanvasObject={(n, ctx, scale) => {
          const node = n as GNode & { x: number; y: number }
          const r = node.center ? 11 : 5 + 7 * (node.sim / maxSim)
          ctx.beginPath()
          ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
          ctx.fillStyle = node.center ? colors.accent : colors.violet
          ctx.fill()
          if (node.id === hover) {
            ctx.lineWidth = 2 / scale
            ctx.strokeStyle = colors.accent
            ctx.stroke()
          }
          if (node.center || node.id === hover) {
            const text = `${bareTitle(node.label)}${yearOf(node.label) ? ` (${yearOf(node.label)})` : ''}`
            ctx.font = `${12 / scale}px ui-sans-serif, system-ui, sans-serif`
            ctx.textAlign = 'center'
            ctx.fillStyle = colors.text
            ctx.fillText(text, node.x, node.y + r + 12 / scale)
          }
        }}
        nodePointerAreaPaint={(n, color, ctx) => {
          const node = n as GNode & { x: number; y: number }
          ctx.fillStyle = color
          ctx.beginPath()
          ctx.arc(node.x, node.y, (node.center ? 11 : 5 + 7 * (node.sim / maxSim)) + 3, 0, 2 * Math.PI)
          ctx.fill()
        }}
      />
    </div>
  )
}
