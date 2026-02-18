"""
Geometry-Complexity Correlation Analysis

Comprehensive statistical analysis of correlations between
code geometry features and Krylov complexity dynamics.

Features analyzed:
- Geometric: code distance, encoding rate, stabilizer weight, depth
- Dynamic: growth exponent, saturation time, saturation value, Lanczos coefficients
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class CorrelationResult:
    """Result of a correlation computation."""
    feature1: str
    feature2: str
    correlation: float
    p_value: float
    method: str
    confidence_interval: Optional[Tuple[float, float]] = None
    n_samples: int = 0


class GeometryComplexityAnalyzer:
    """
    Comprehensive correlation analysis between code geometry and complexity.

    Supports multiple correlation measures:
    - Pearson (linear)
    - Spearman (rank)
    - Kendall (concordance)
    - Mutual Information (non-linear)
    """

    def __init__(
        self,
        geometric_features: Optional[Dict[str, np.ndarray]] = None,
        dynamic_features: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize analyzer with feature data.

        Args:
            geometric_features: Dict mapping feature names to value arrays
            dynamic_features: Dict mapping feature names to value arrays
        """
        self.geometric_features = geometric_features or {}
        self.dynamic_features = dynamic_features or {}
        self.correlation_matrix = None
        self.p_value_matrix = None

    def set_geometric_features(self, features: Dict[str, np.ndarray]):
        """Set geometric features from code data."""
        self.geometric_features = features

    def set_dynamic_features(self, features: Dict[str, np.ndarray]):
        """Set dynamic features from complexity data."""
        self.dynamic_features = features

    def extract_geometric_features_from_codes(self, codes: List) -> Dict[str, np.ndarray]:
        """
        Extract geometric features from a list of codes.

        Features extracted:
        - distance: Code distance d
        - rate: Encoding rate k/n
        - n_physical: Number of physical qubits
        - n_logical: Number of logical qubits
        - avg_stabilizer_weight: Average weight of stabilizers
        - depth: HaPPY depth (if applicable)
        """
        features = {
            'distance': [],
            'rate': [],
            'n_physical': [],
            'n_logical': [],
            'avg_stabilizer_weight': [],
            'depth': []
        }

        for code in codes:
            # Get stabilizer code
            if hasattr(code, 'stabilizer_code'):
                sc = code.stabilizer_code
                depth = code.depth if hasattr(code, 'depth') else 0
            else:
                sc = code
                depth = 0

            # Estimate effective depth for non-HaPPY codes from stabilizer structure
            if depth == 0 and sc.n_physical > 1:
                # Use log2(n_physical) as proxy for effective circuit depth
                # This approximates the geometric "depth" for codes without explicit layers
                depth = max(1, int(np.log2(sc.n_physical)))

            # Extract features
            features['distance'].append(sc.distance if sc.distance else 0)
            features['rate'].append(sc.n_logical / max(1, sc.n_physical))
            features['n_physical'].append(sc.n_physical)
            features['n_logical'].append(sc.n_logical)

            # Average stabilizer weight
            weights = [sum(1 for p in s if p != 'I') for s in sc.stabilizers]
            features['avg_stabilizer_weight'].append(np.mean(weights) if weights else 0)

            features['depth'].append(depth)

        self.geometric_features = {k: np.array(v) for k, v in features.items()}
        return self.geometric_features

    def extract_dynamic_features_from_results(
        self,
        krylov_results: List
    ) -> Dict[str, np.ndarray]:
        """
        Extract dynamic features from Krylov complexity results.

        Features extracted:
        - growth_exponent: Early-time power law exponent alpha
        - saturation_value: Long-time saturation C_sat
        - saturation_time: Time to reach 90% saturation
        - krylov_dim: Krylov subspace dimension
        - avg_lanczos_alpha: Average diagonal Lanczos coefficient
        - avg_lanczos_beta: Average off-diagonal Lanczos coefficient
        - max_complexity: Maximum complexity reached
        """
        features = {
            'growth_exponent': [],
            'saturation_value': [],
            'saturation_time': [],
            'krylov_dim': [],
            'avg_lanczos_alpha': [],
            'avg_lanczos_beta': [],
            'max_complexity': []
        }

        for result in krylov_results:
            features['growth_exponent'].append(
                result.growth_exponent if result.growth_exponent else 0
            )
            features['saturation_value'].append(
                result.saturation_value if result.saturation_value else 0
            )
            features['krylov_dim'].append(result.krylov_dimension)

            # Saturation time (time to reach 90% of max)
            max_c = np.max(result.complexity)
            if max_c > 0:
                sat_idx = np.argmax(result.complexity >= 0.9 * max_c)
                sat_time = result.times[sat_idx] if sat_idx > 0 else result.times[-1]
            else:
                sat_time = result.times[-1]
            features['saturation_time'].append(sat_time)

            # Lanczos coefficients
            alpha, beta = result.lanczos_coeffs
            features['avg_lanczos_alpha'].append(np.mean(alpha) if len(alpha) > 0 else 0)
            features['avg_lanczos_beta'].append(np.mean(beta) if len(beta) > 0 else 0)

            features['max_complexity'].append(max_c)

        self.dynamic_features = {k: np.array(v) for k, v in features.items()}
        return self.dynamic_features

    def compute_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'pearson'
    ) -> Tuple[float, float]:
        """
        Compute correlation between two arrays.

        Args:
            x, y: Data arrays
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            (correlation, p_value)
        """
        # Remove NaN/Inf values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 3:
            return 0.0, 1.0

        # Check for constant arrays (no variance) - correlation undefined
        if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
            return 0.0, 1.0

        if method == 'pearson':
            r, p = stats.pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            r, p = stats.spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            r, p = stats.kendalltau(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown method: {method}")

        return float(r), float(p)

    def compute_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_neighbors: int = 5
    ) -> float:
        """
        Compute mutual information between two arrays.

        Uses k-nearest neighbors estimation.
        """
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask].reshape(-1, 1)
        y_clean = y[mask]

        if len(x_clean) < 10:
            return 0.0

        try:
            mi = mutual_info_regression(x_clean, y_clean, n_neighbors=n_neighbors)
            return float(mi[0])
        except:
            return 0.0

    def compute_all_correlations(
        self,
        methods: List[str] = ['pearson', 'spearman']
    ) -> Dict[Tuple[str, str], List[CorrelationResult]]:
        """
        Compute all pairwise correlations between geometric and dynamic features.

        Returns:
            Dict mapping (feature1, feature2) to list of CorrelationResults
        """
        results = {}

        for g_name, g_values in self.geometric_features.items():
            for d_name, d_values in self.dynamic_features.items():
                key = (g_name, d_name)
                results[key] = []

                for method in methods:
                    r, p = self.compute_correlation(g_values, d_values, method)

                    result = CorrelationResult(
                        feature1=g_name,
                        feature2=d_name,
                        correlation=r,
                        p_value=p,
                        method=method,
                        n_samples=len(g_values)
                    )
                    results[key].append(result)

        return results

    def compute_correlation_matrix(
        self,
        method: str = 'pearson'
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute full correlation matrix.

        Returns:
            (matrix, geometric_names, dynamic_names)
        """
        g_names = list(self.geometric_features.keys())
        d_names = list(self.dynamic_features.keys())

        matrix = np.zeros((len(g_names), len(d_names)))
        p_matrix = np.zeros((len(g_names), len(d_names)))

        for i, g_name in enumerate(g_names):
            for j, d_name in enumerate(d_names):
                r, p = self.compute_correlation(
                    self.geometric_features[g_name],
                    self.dynamic_features[d_name],
                    method
                )
                matrix[i, j] = r
                p_matrix[i, j] = p

        self.correlation_matrix = matrix
        self.p_value_matrix = p_matrix

        return matrix, g_names, d_names

    def bootstrap_confidence_interval(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'pearson',
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for correlation.

        Returns:
            (lower, upper) bounds
        """
        n = len(x)
        correlations = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            r, _ = self.compute_correlation(x[idx], y[idx], method)
            if np.isfinite(r):
                correlations.append(r)

        if len(correlations) < 10:
            return (-1.0, 1.0)

        alpha = (1 - confidence) / 2
        lower = np.percentile(correlations, alpha * 100)
        upper = np.percentile(correlations, (1 - alpha) * 100)

        return (lower, upper)

    def find_significant_correlations(
        self,
        alpha: float = 0.05,
        method: str = 'pearson',
        correction: str = 'bonferroni'
    ) -> List[CorrelationResult]:
        """
        Find statistically significant correlations.

        Args:
            alpha: Significance level
            method: Correlation method
            correction: Multiple testing correction ('bonferroni' or 'fdr')

        Returns:
            List of significant CorrelationResults
        """
        all_correlations = self.compute_all_correlations([method])

        # Collect all p-values
        results_list = []
        for key, results in all_correlations.items():
            for result in results:
                results_list.append(result)

        n_tests = len(results_list)

        if correction == 'bonferroni':
            adjusted_alpha = alpha / n_tests
        elif correction == 'fdr':
            # Benjamini-Hochberg
            p_values = [r.p_value for r in results_list]
            sorted_idx = np.argsort(p_values)
            adjusted_alpha = alpha  # Use FDR threshold

        significant = []
        for result in results_list:
            if result.p_value < adjusted_alpha:
                significant.append(result)

        # Sort by absolute correlation
        significant.sort(key=lambda r: abs(r.correlation), reverse=True)

        return significant

    def summary_report(self) -> str:
        """Generate a summary report of correlation analysis."""
        report = []
        report.append("=" * 60)
        report.append("GEOMETRY-COMPLEXITY CORRELATION ANALYSIS")
        report.append("=" * 60)

        # Compute correlations
        matrix, g_names, d_names = self.compute_correlation_matrix('pearson')

        report.append("\n## Pearson Correlations\n")
        report.append(f"Geometric features: {g_names}")
        report.append(f"Dynamic features: {d_names}")

        # Find strongest correlations
        report.append("\n## Strongest Correlations\n")
        significant = self.find_significant_correlations(alpha=0.05)

        for result in significant[:10]:
            ci = self.bootstrap_confidence_interval(
                self.geometric_features[result.feature1],
                self.dynamic_features[result.feature2],
                'pearson'
            )
            report.append(
                f"  {result.feature1} <-> {result.feature2}: "
                f"r={result.correlation:.3f} (p={result.p_value:.2e}), "
                f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]"
            )

        # Average correlation per geometric feature
        report.append("\n## Average Correlation by Geometric Feature\n")
        for i, g_name in enumerate(g_names):
            avg_r = np.mean(np.abs(matrix[i, :]))
            report.append(f"  {g_name}: avg|r| = {avg_r:.3f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


class PhaseTransitionAnalyzer:
    """
    Detect phase transitions in complexity behavior.

    Scans parameter space and looks for:
    - Discontinuities in derivatives
    - Diverging susceptibilities
    - Critical slowing down
    """

    def __init__(self):
        self.scan_results = {}

    def scan_parameter(
        self,
        param_values: np.ndarray,
        complexity_curves: List[np.ndarray],
        times: np.ndarray
    ) -> Dict[str, Any]:
        """
        Scan a parameter range for phase transition signatures.

        Args:
            param_values: Array of parameter values
            complexity_curves: List of C_K(t) arrays
            times: Time array

        Returns:
            Dict with analysis results
        """
        n_params = len(param_values)

        # Extract summary statistics at each parameter value
        growth_exponents = []
        saturation_values = []
        saturation_times = []

        for curve in complexity_curves:
            # Growth exponent (fit early times)
            mask = (times > 0.1) & (times < 2.0) & (curve > 1e-6)
            if np.sum(mask) > 5:
                log_t = np.log(times[mask])
                log_c = np.log(curve[mask])
                coeff = np.polyfit(log_t, log_c, 1)
                growth_exponents.append(coeff[0])
            else:
                growth_exponents.append(0.0)

            # Saturation value
            saturation_values.append(np.mean(curve[-len(curve)//5:]))

            # Saturation time (90% of max)
            max_c = np.max(curve)
            if max_c > 0:
                sat_idx = np.argmax(curve >= 0.9 * max_c)
                saturation_times.append(times[sat_idx])
            else:
                saturation_times.append(times[-1])

        growth_exponents = np.array(growth_exponents)
        saturation_values = np.array(saturation_values)
        saturation_times = np.array(saturation_times)

        # Compute derivatives (susceptibility)
        d_growth = np.gradient(growth_exponents, param_values)
        d_sat = np.gradient(saturation_values, param_values)

        # Find maxima in derivatives (potential phase transitions)
        growth_susceptibility = np.abs(d_growth)
        sat_susceptibility = np.abs(d_sat)

        # Detect peaks
        from scipy.signal import find_peaks
        growth_peaks, _ = find_peaks(growth_susceptibility, prominence=0.1)
        sat_peaks, _ = find_peaks(sat_susceptibility, prominence=0.1)

        results = {
            'param_values': param_values,
            'growth_exponents': growth_exponents,
            'saturation_values': saturation_values,
            'saturation_times': saturation_times,
            'growth_susceptibility': growth_susceptibility,
            'sat_susceptibility': sat_susceptibility,
            'growth_peak_params': param_values[growth_peaks] if len(growth_peaks) > 0 else [],
            'sat_peak_params': param_values[sat_peaks] if len(sat_peaks) > 0 else [],
        }

        return results

    def identify_critical_point(
        self,
        scan_results: Dict[str, Any]
    ) -> Optional[float]:
        """
        Identify the critical parameter value.

        Returns the parameter value where susceptibility is maximum.
        """
        param_values = scan_results['param_values']
        susceptibility = scan_results['growth_susceptibility'] + scan_results['sat_susceptibility']

        if np.max(susceptibility) > 0.2:  # Threshold for significance
            critical_idx = np.argmax(susceptibility)
            return param_values[critical_idx]

        return None


if __name__ == '__main__':
    print("Testing GeometryComplexityAnalyzer...")

    # Create dummy data
    np.random.seed(42)
    n_samples = 100

    geometric = {
        'distance': np.random.randint(2, 10, n_samples).astype(float),
        'rate': np.random.uniform(0.1, 0.5, n_samples),
        'depth': np.random.randint(1, 5, n_samples).astype(float)
    }

    # Create correlated dynamic features
    dynamic = {
        'growth_exponent': 0.5 * geometric['depth'] + np.random.randn(n_samples) * 0.2,
        'saturation_value': 2.0 * geometric['distance'] + np.random.randn(n_samples) * 1.0,
        'krylov_dim': 5 * geometric['depth'] + np.random.randint(0, 10, n_samples)
    }

    # Analyze
    analyzer = GeometryComplexityAnalyzer(geometric, dynamic)

    # Compute correlations
    matrix, g_names, d_names = analyzer.compute_correlation_matrix('pearson')
    print(f"Correlation matrix shape: {matrix.shape}")
    print(f"Geometric features: {g_names}")
    print(f"Dynamic features: {d_names}")

    # Find significant correlations
    significant = analyzer.find_significant_correlations(alpha=0.01)
    print(f"\nSignificant correlations (p < 0.01): {len(significant)}")
    for r in significant[:5]:
        print(f"  {r.feature1} <-> {r.feature2}: r={r.correlation:.3f}, p={r.p_value:.2e}")

    # Print summary
    print("\n" + analyzer.summary_report())
