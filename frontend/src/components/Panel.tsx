import type { ReactNode } from 'react'

/** Glass container with the one bevelled corner (see .panel in index.css). Put body text inside a
 *  `.solid` child: translucency under paragraphs fails contrast. */
export function Panel({ children, className = '', as: Tag = 'section', ...rest }: {
  children: ReactNode
  className?: string
  as?: 'section' | 'article' | 'div' | 'aside'
} & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag className={`panel ${className}`} {...rest}>
      <div className="panel-inner">{children}</div>
    </Tag>
  )
}
