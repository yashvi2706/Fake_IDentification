import { Component, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackMessage?: string;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error('[ErrorBoundary] Caught render error:', error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    this.props.onReset?.();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center py-20 gap-6 text-center">
          <AlertTriangle className="w-12 h-12 text-[var(--sentinel-caution)]" />
          <div>
            <h2 className="sentinel-display text-2xl mb-2">
              {this.props.fallbackMessage ?? 'Unable to display analysis results.'}
            </h2>
            <p className="text-sm text-[var(--sentinel-text-muted)] mb-6">
              {this.state.error?.message ?? 'An unexpected render error occurred.'}
            </p>
            <button onClick={this.handleReset} className="sentinel-secondary-action">
              Try Another Document
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
