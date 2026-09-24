import { Outlet, Link, useLocation } from 'react-router-dom';
import { Shield } from 'lucide-react';

export default function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top Navigation */}
      <header className="border-b border-[var(--sentinel-rule)] bg-[var(--sentinel-surface)]">
        <div className="max-w-[var(--sentinel-content-max)] mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-[var(--sentinel-accent)]" />
            <span className="font-semibold text-[var(--sentinel-text)] tracking-wide">SENTINEL</span>
          </div>
          <nav className="flex items-center gap-8">
            <Link 
              to="/dashboard" 
              className={`text-sm font-medium transition-colors hover:text-[var(--sentinel-text)] ${location.pathname === '/dashboard' ? 'text-[var(--sentinel-accent)]' : 'text-[var(--sentinel-text-muted)]'}`}
            >
              Dashboard
            </Link>
            <Link 
              to="/screening/new" 
              className="sentinel-primary-action inline-flex items-center justify-center text-decoration-none"
            >
              New Screening
            </Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto bg-[var(--sentinel-canvas)] bg-[image:var(--sentinel-atmosphere)] bg-no-repeat bg-fixed">
        <div className="max-w-[var(--sentinel-content-max)] mx-auto px-6 py-12">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
