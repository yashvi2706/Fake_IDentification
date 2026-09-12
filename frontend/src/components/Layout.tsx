import React from 'react'
import { Outlet, Link, useNavigate } from 'react-router-dom'
import { Shield, Upload as UploadIcon, LayoutDashboard, LogOut } from 'lucide-react'

export default function Layout() {
  const navigate = useNavigate()

  const handleLogout = () => {
    navigate('/login')
  }

  return (
    <div className="min-h-screen bg-secondary flex">
      {/* Sidebar */}
      <aside className="w-64 bg-card border-r border-border flex flex-col">
        <div className="p-6 flex items-center gap-3 border-b border-border">
          <Shield className="w-8 h-8 text-primary" />
          <h1 className="font-bold text-lg leading-tight">Identity Screening<br/>System</h1>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          <Link to="/upload" className="flex items-center gap-3 px-4 py-3 text-muted-foreground hover:bg-secondary hover:text-foreground rounded-md transition-colors">
            <UploadIcon className="w-5 h-5" />
            <span className="font-medium">Document Upload</span>
          </Link>
          <Link to="/dashboard" className="flex items-center gap-3 px-4 py-3 text-muted-foreground hover:bg-secondary hover:text-foreground rounded-md transition-colors">
            <LayoutDashboard className="w-5 h-5" />
            <span className="font-medium">Results Dashboard</span>
          </Link>
        </nav>

        <div className="p-4 border-t border-border">
          <button onClick={handleLogout} className="flex items-center gap-3 w-full px-4 py-3 text-muted-foreground hover:bg-destructive/10 hover:text-destructive rounded-md transition-colors">
            <LogOut className="w-5 h-5" />
            <span className="font-medium">Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
