import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import ScreeningFlow from './pages/ScreeningFlow';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="screening/new" element={<ScreeningFlow />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
