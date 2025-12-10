"""
UER Compilation Performance Metrics

Tracks compilation performance, success rates, and optimization opportunities.
"""

from typing import Dict, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)


class CompilationMetrics:
    """
    Tracks UER compilation performance and reliability metrics.

    Provides insights into optimization opportunities and system health.
    """

    def __init__(self):
        """Initialize metrics tracking."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.single_compilations = 0
        self.batch_compilations = 0
        self.total_embeddings_processed = 0
        self.successful_compilations = 0
        self.failed_compilations = 0
        self.total_compilation_time = 0.0
        self.batch_sizes = []
        self.alignment_times = []
        self.normalization_times = []
        self.error_counts = {}

    def record_compilation(self, duration: float, success: bool) -> None:
        """Record a single compilation event."""
        self.single_compilations += 1
        self.total_embeddings_processed += 1

        if success:
            self.successful_compilations += 1
        else:
            self.failed_compilations += 1

        self.total_compilation_time += duration

    def record_batch_compilation(self, duration: float, batch_size: int, success: bool) -> None:
        """Record a batch compilation event."""
        self.batch_compilations += 1
        self.total_embeddings_processed += batch_size
        self.batch_sizes.append(batch_size)

        if success:
            self.successful_compilations += batch_size
        else:
            self.failed_compilations += batch_size

        self.total_compilation_time += duration

    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Aggregate metrics into a dictionary."""
        total_ops = self.single_compilations + self.batch_compilations

        metrics = {
            'total_operations': total_ops,
            'single_compilations': self.single_compilations,
            'batch_compilations': self.batch_compilations,
            'total_embeddings': self.total_embeddings_processed,
            'successful_compilations': self.successful_compilations,
            'failed_compilations': self.failed_compilations,
            'success_rate': self.successful_compilations / max(1, self.total_embeddings_processed),
            'total_time_seconds': self.total_compilation_time,
            'avg_time_per_embedding': self.total_compilation_time / max(1, self.total_embeddings_processed),
            'operations_per_second': total_ops / max(0.001, self.total_compilation_time),
            'error_breakdown': dict(self.error_counts)
        }

        if self.batch_sizes:
            metrics.update({
                'avg_batch_size': sum(self.batch_sizes) / len(self.batch_sizes),
                'max_batch_size': max(self.batch_sizes),
                'min_batch_size': min(self.batch_sizes)
            })

        return metrics

    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary."""
        metrics = self.to_dict()
        summary = ".2f"        summary += f"  Success Rate: {metrics['success_rate']:.1%}\n"
        summary += f"  Throughput: {metrics['operations_per_second']:.1f} ops/sec\n"
        summary += f"  Avg Processing: {metrics['avg_time_per_embedding']*1000:.2f}ms per embedding\n"

        if metrics['error_breakdown']:
            summary += f"  Top Errors: {sorted(metrics['error_breakdown'].items(), key=lambda x: x[1], reverse=True)[:3]}\n"

        return summary

    def log_performance_report(self) -> None:
        """Log a detailed performance report."""
        logger.info("UER Compilation Performance Report:")
        logger.info(self.get_performance_summary())
