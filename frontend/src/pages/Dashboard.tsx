import { mockRecentScreenings, mockStats } from '../data/mockData';
import { Link } from 'react-router-dom';
import { ShieldCheck, AlertTriangle, Clock } from 'lucide-react';

export default function Dashboard() {
  return (
    <div className="space-y-16">
      <header>
        <h1 className="sentinel-display text-5xl mb-4 text-[var(--sentinel-text)]">Intelligence Overview</h1>
        <p className="text-[var(--sentinel-text-muted)] text-lg">AI-Based Document Screening and Forgery Detection</p>
      </header>

      {/* Horizontal Stats Band */}
      <section className="flex flex-wrap gap-12 py-8 border-y border-[var(--sentinel-rule)]">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-[var(--sentinel-text-muted)]">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-sm font-medium uppercase tracking-wider">Total Screened</span>
          </div>
          <span className="sentinel-display text-4xl">{mockStats.totalScreened.toLocaleString()}</span>
        </div>
        
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-[var(--sentinel-positive)]">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-sm font-medium uppercase tracking-wider text-[var(--sentinel-text-muted)]">Low Risk</span>
          </div>
          <span className="sentinel-display text-4xl text-[var(--sentinel-positive)]">{mockStats.lowRisk.toLocaleString()}</span>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-[var(--sentinel-critical)]">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-sm font-medium uppercase tracking-wider text-[var(--sentinel-text-muted)]">Flagged</span>
          </div>
          <span className="sentinel-display text-4xl text-[var(--sentinel-critical)]">{mockStats.flagged.toLocaleString()}</span>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-[var(--sentinel-text-muted)]">
            <Clock className="w-4 h-4" />
            <span className="text-sm font-medium uppercase tracking-wider">Avg Processing</span>
          </div>
          <span className="sentinel-display text-4xl">{mockStats.avgTime}</span>
        </div>
      </section>

      {/* Recent Screenings List directly on canvas */}
      <section>
        <div className="flex items-center justify-between mb-8">
          <h2 className="sentinel-display text-3xl">Recent Activity</h2>
          <Link to="/screening/new" className="text-sm text-[var(--sentinel-accent)] hover:text-[var(--sentinel-accent-hover)] transition-colors">
            Start New Screening &rarr;
          </Link>
        </div>

        <div className="w-full">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[var(--sentinel-rule)] text-[var(--sentinel-text-muted)] text-sm">
                <th className="pb-4 font-medium uppercase tracking-wider">ID</th>
                <th className="pb-4 font-medium uppercase tracking-wider">Document</th>
                <th className="pb-4 font-medium uppercase tracking-wider">Subject</th>
                <th className="pb-4 font-medium uppercase tracking-wider">Risk Level</th>
                <th className="pb-4 font-medium uppercase tracking-wider text-right">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--sentinel-rule)]">
              {mockRecentScreenings.map((screening) => (
                <tr key={screening.id} className="group hover:bg-[var(--sentinel-surface)] transition-colors">
                  <td className="py-4 font-mono text-sm text-[var(--sentinel-text-muted)]">{screening.id}</td>
                  <td className="py-4">{screening.document_type}</td>
                  <td className="py-4">{screening.person_name}</td>
                  <td className="py-4">
                    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-semibold
                      ${screening.risk_level === 'LOW' ? 'bg-[var(--sentinel-positive-tint)] text-[var(--sentinel-positive)]' : ''}
                      ${screening.risk_level === 'REVIEW' ? 'bg-[var(--sentinel-caution-tint)] text-[var(--sentinel-caution)]' : ''}
                      ${screening.risk_level === 'HIGH' ? 'bg-[var(--sentinel-critical-tint)] text-[var(--sentinel-critical)]' : ''}
                    `}>
                      {screening.risk_level}
                    </span>
                  </td>
                  <td className="py-4 text-sm text-right text-[var(--sentinel-text-muted)]">
                    {new Date(screening.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
