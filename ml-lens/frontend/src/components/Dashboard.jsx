import Card from './Card'
import StatsCard from './StatsCard'

export default function Dashboard() {
  const stats = [
    { label: 'Total Evaluations', value: '24', subtext: 'This month' },
    { label: 'Pass Rate', value: '92%', subtext: 'Baseline models' },
    { label: 'Avg Score', value: '8.4/10', subtext: 'All evaluations' }
  ]

  const recentEvals = [
    { id: 1, name: 'GPT-4 vs Claude', status: 'completed', score: 8.7 },
    { id: 2, name: 'Summarization Task', status: 'in-progress', score: null },
    { id: 3, name: 'Code Generation', status: 'completed', score: 8.2 }
  ]

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Dashboard</h2>
        <button className="btn-primary">+ New Evaluation</button>
      </div>

      <div className="stats-grid">
        {stats.map((stat, idx) => (
          <StatsCard key={idx} {...stat} />
        ))}
      </div>

      <div className="dashboard-section">
        <h3 className="section-title">Recent Evaluations</h3>
        <div className="evals-list">
          {recentEvals.map(eval => (
            <Card key={eval.id} title={eval.name}>
              <div className="eval-row">
                <span className={`badge badge-${eval.status}`}>
                  {eval.status}
                </span>
                {eval.score && (
                  <span className="eval-score">{eval.score}</span>
                )}
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
