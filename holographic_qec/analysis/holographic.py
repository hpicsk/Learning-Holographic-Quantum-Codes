"""
AdS/CFT Holographic Dictionary Tests

Implements empirical tests of holographic correspondence predictions:
1. Code distance d ~ Black hole entropy S_BH
2. Complexity growth rate ~ Temperature T
3. Saturation complexity ~ Page curve (min of subsystem sizes)
4. Entanglement structure ~ Bulk geometry
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class HolographicTestResult:
    """Result of a holographic dictionary test."""
    test_name: str
    prediction: str
    observed_correlation: float
    p_value: float
    passed: bool
    fit_params: Optional[Dict[str, float]] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    details: str = ""


class HolographicDictionary:
    """
    Test empirical correspondences from AdS/CFT.

    Based on holographic quantum error correction:
    - Bulk operators are encoded in boundary
    - Code distance relates to geometric features
    - Complexity growth has holographic interpretation
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: Alpha for statistical tests
        """
        self.alpha = significance_level
        self.test_results = []

    def test_entropy_distance_relation(
        self,
        depths: np.ndarray,
        distances: np.ndarray,
        n_physical: np.ndarray
    ) -> HolographicTestResult:
        """
        Test: S_BH ∝ d (code distance ~ entropy)

        For HaPPY codes, distance grows with depth ~ log of boundary size.
        In holographic terms: black hole entropy S_BH ∝ Area ∝ d.

        Prediction: d ~ exp(depth) or d ~ n_physical^α for some α > 0
        """
        # Test 1: Distance vs depth (exponential relation)
        mask = (depths > 0) & (distances > 0)
        d_clean = depths[mask]
        dist_clean = distances[mask]

        if len(d_clean) < 5:
            return HolographicTestResult(
                test_name="entropy_distance",
                prediction="d ~ exp(depth)",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Insufficient data"
            )

        # Check for constant arrays (no variance) - correlation undefined
        if np.std(d_clean) < 1e-10 or np.std(dist_clean) < 1e-10:
            return HolographicTestResult(
                test_name="entropy_distance",
                prediction="d ~ exp(depth)",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Constant values - cannot compute correlation"
            )

        # Fit exponential: d = a * exp(b * depth)
        try:
            def exp_func(x, a, b):
                return a * np.exp(b * x)

            popt, pcov = curve_fit(
                exp_func, d_clean, dist_clean,
                p0=[1, 0.5], maxfev=1000
            )

            # Compute R^2
            y_pred = exp_func(d_clean, *popt)
            ss_res = np.sum((dist_clean - y_pred) ** 2)
            ss_tot = np.sum((dist_clean - np.mean(dist_clean)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)

            # Linear correlation for comparison
            r_linear, p_linear = stats.pearsonr(d_clean, dist_clean)

            passed = (r2 > 0.5) or (r_linear > 0.6 and p_linear < self.alpha)

            return HolographicTestResult(
                test_name="entropy_distance",
                prediction="d ~ exp(depth) [S_BH ~ Area]",
                observed_correlation=r_linear,
                p_value=p_linear,
                passed=passed,
                fit_params={'a': popt[0], 'b': popt[1], 'r2': r2},
                details=f"Exponential fit R²={r2:.3f}, Linear r={r_linear:.3f}"
            )

        except Exception as e:
            # Fall back to linear correlation
            r, p = stats.pearsonr(d_clean, dist_clean)
            return HolographicTestResult(
                test_name="entropy_distance",
                prediction="d ~ depth",
                observed_correlation=r,
                p_value=p,
                passed=(r > 0.5 and p < self.alpha),
                details=f"Fit failed, linear r={r:.3f}, p={p:.3e}"
            )

    def test_complexity_temperature(
        self,
        growth_exponents: np.ndarray,
        hamiltonian_temperatures: np.ndarray
    ) -> HolographicTestResult:
        """
        Test: dC_K/dt ~ T (complexity growth ~ temperature)

        Higher temperature → faster scrambling → faster complexity growth.

        In the Ising model, temperature relates to transverse field g.
        """
        mask = np.isfinite(growth_exponents) & np.isfinite(hamiltonian_temperatures)
        alpha = growth_exponents[mask]
        T = hamiltonian_temperatures[mask]

        if len(alpha) < 5:
            return HolographicTestResult(
                test_name="complexity_temperature",
                prediction="dC/dt ~ T",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Insufficient data"
            )

        # Check if T values have any variance (all identical values break regression)
        if np.std(T) < 1e-10:
            return HolographicTestResult(
                test_name="complexity_temperature",
                prediction="dC_K/dt ~ T (Lloyd bound)",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Constant temperature values - cannot compute correlation"
            )

        # Linear correlation
        r, p = stats.pearsonr(T, alpha)

        # The prediction is that higher T leads to higher growth rate
        passed = (r > 0.3 and p < self.alpha)

        # Try linear fit
        slope, intercept, _, _, std_err = stats.linregress(T, alpha)

        return HolographicTestResult(
            test_name="complexity_temperature",
            prediction="dC_K/dt ~ T (Lloyd bound)",
            observed_correlation=r,
            p_value=p,
            passed=passed,
            fit_params={'slope': slope, 'intercept': intercept, 'std_err': std_err},
            details=f"Slope={slope:.3f}±{std_err:.3f}, r={r:.3f}"
        )

    def test_page_curve(
        self,
        saturation_values: np.ndarray,
        n_physical: np.ndarray,
        n_logical: np.ndarray
    ) -> HolographicTestResult:
        """
        Test: C_sat ~ min(|A|, |B|) for bipartition A|B

        Complexity saturates at Page value, proportional to the smaller
        subsystem size.

        For codes: min(k, n-k) where k = logical, n-k = redundancy
        """
        # Page value approximation: min(k, n-k)
        page_value = np.minimum(n_logical, n_physical - n_logical)

        mask = np.isfinite(saturation_values) & (page_value > 0)
        C_sat = saturation_values[mask]
        page = page_value[mask]

        if len(C_sat) < 5:
            return HolographicTestResult(
                test_name="page_curve",
                prediction="C_sat ~ min(k, n-k)",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Insufficient data"
            )

        # Check for constant arrays (no variance) - correlation undefined
        if np.std(page) < 1e-10 or np.std(C_sat) < 1e-10:
            return HolographicTestResult(
                test_name="page_curve",
                prediction="C_sat ~ min(k, n-k)",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Constant values - cannot compute correlation"
            )

        # Correlation with Page value
        r, p = stats.pearsonr(page, C_sat)

        # The saturation should scale with Page value
        passed = (r > 0.4 and p < self.alpha)

        return HolographicTestResult(
            test_name="page_curve",
            prediction="C_sat ~ min(k, n-k) [Page curve]",
            observed_correlation=r,
            p_value=p,
            passed=passed,
            details=f"Correlation with Page value: r={r:.3f}, p={p:.3e}"
        )

    def test_entanglement_geometry(
        self,
        krylov_dims: np.ndarray,
        code_depths: np.ndarray,
        distances: np.ndarray
    ) -> HolographicTestResult:
        """
        Test: Krylov dimension ~ geodesic length in bulk

        The Krylov subspace dimension should correlate with
        geometric measures of the bulk (code depth, distance).

        Prediction: Deeper codes (more bulk) → larger Krylov spaces
        """
        # Correlate Krylov dimension with code depth
        mask = (krylov_dims > 0) & (code_depths > 0)
        K = krylov_dims[mask]
        D = code_depths[mask]

        if len(K) < 5:
            return HolographicTestResult(
                test_name="entanglement_geometry",
                prediction="dim(K) ~ depth",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Insufficient data"
            )

        # Check for constant arrays (no variance) - correlation undefined
        if np.std(K) < 1e-10 or np.std(D) < 1e-10:
            return HolographicTestResult(
                test_name="entanglement_geometry",
                prediction="dim(K) ~ depth",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Constant values - cannot compute correlation"
            )

        r_depth, p_depth = stats.pearsonr(D, K)

        # Also correlate with distance
        dist_clean = distances[mask]
        # Handle constant distance values
        if np.std(dist_clean) < 1e-10:
            r_dist, p_dist = 0.0, 1.0
        else:
            r_dist, p_dist = stats.pearsonr(dist_clean, K)

        passed = (r_depth > 0.5 and p_depth < self.alpha) or \
                 (r_dist > 0.5 and p_dist < self.alpha)

        return HolographicTestResult(
            test_name="entanglement_geometry",
            prediction="dim(K) ~ bulk geometry",
            observed_correlation=max(r_depth, r_dist),
            p_value=min(p_depth, p_dist),
            passed=passed,
            details=f"r(depth)={r_depth:.3f}, r(distance)={r_dist:.3f}"
        )

    def run_all_tests(
        self,
        geometric_data: Dict[str, np.ndarray],
        dynamic_data: Dict[str, np.ndarray],
        hamiltonian_data: Optional[Dict[str, np.ndarray]] = None
    ) -> List[HolographicTestResult]:
        """
        Run all holographic dictionary tests.

        Args:
            geometric_data: Dict with 'depth', 'distance', 'n_physical', 'n_logical'
            dynamic_data: Dict with 'growth_exponent', 'saturation_value', 'krylov_dim'
            hamiltonian_data: Optional dict with 'temperature' or 'field_strength'

        Returns:
            List of HolographicTestResult
        """
        results = []

        # Test 1: Entropy-Distance
        result1 = self.test_entropy_distance_relation(
            geometric_data.get('depth', np.array([])),
            geometric_data.get('distance', np.array([])),
            geometric_data.get('n_physical', np.array([]))
        )
        results.append(result1)

        # Test 2: Complexity-Temperature
        if hamiltonian_data and 'temperature' in hamiltonian_data:
            result2 = self.test_complexity_temperature(
                dynamic_data.get('growth_exponent', np.array([])),
                hamiltonian_data['temperature']
            )
        else:
            # Use field strength as proxy for temperature
            field = hamiltonian_data.get('field_strength', np.ones(len(dynamic_data.get('growth_exponent', [])))) \
                    if hamiltonian_data else np.ones(len(dynamic_data.get('growth_exponent', [])))
            result2 = self.test_complexity_temperature(
                dynamic_data.get('growth_exponent', np.array([])),
                field
            )
        results.append(result2)

        # Test 3: Page Curve
        result3 = self.test_page_curve(
            dynamic_data.get('saturation_value', np.array([])),
            geometric_data.get('n_physical', np.array([])),
            geometric_data.get('n_logical', np.array([]))
        )
        results.append(result3)

        # Test 4: Entanglement-Geometry
        result4 = self.test_entanglement_geometry(
            dynamic_data.get('krylov_dim', np.array([])),
            geometric_data.get('depth', np.array([])),
            geometric_data.get('distance', np.array([]))
        )
        results.append(result4)

        self.test_results = results
        return results

    def summary_report(self) -> str:
        """Generate summary report of holographic tests."""
        report = []
        report.append("=" * 60)
        report.append("HOLOGRAPHIC DICTIONARY TEST RESULTS")
        report.append("=" * 60)

        n_passed = sum(1 for r in self.test_results if r.passed)
        n_total = len(self.test_results)

        report.append(f"\nTests passed: {n_passed}/{n_total}")
        report.append("-" * 40)

        for result in self.test_results:
            status = "PASS" if result.passed else "FAIL"
            report.append(f"\n[{status}] {result.test_name}")
            report.append(f"  Prediction: {result.prediction}")
            report.append(f"  Correlation: r = {result.observed_correlation:.3f}")
            report.append(f"  P-value: {result.p_value:.2e}")
            report.append(f"  Details: {result.details}")

            if result.fit_params:
                params_str = ", ".join(f"{k}={v:.3f}" for k, v in result.fit_params.items())
                report.append(f"  Fit: {params_str}")

        report.append("\n" + "=" * 60)

        # Interpretation
        report.append("\nINTERPRETATION:")
        if n_passed >= 3:
            report.append("Strong evidence for holographic correspondence in code geometry.")
        elif n_passed >= 2:
            report.append("Moderate evidence for holographic features in complexity dynamics.")
        elif n_passed >= 1:
            report.append("Weak evidence - some holographic predictions confirmed.")
        else:
            report.append("No significant holographic signatures detected.")

        return "\n".join(report)


class RTFormula:
    """
    Implementation of Ryu-Takayanagi formula tests.

    S(A) = Area(γ_A) / 4G_N

    Tests whether entanglement entropy scales with minimal surface area
    in the bulk geometry.
    """

    def __init__(self):
        pass

    def compute_entanglement_entropy(
        self,
        psi: np.ndarray,
        subsystem_qubits: List[int],
        total_qubits: int
    ) -> float:
        """
        Compute entanglement entropy of a subsystem.

        S(A) = -Tr(ρ_A log ρ_A)

        Args:
            psi: Full system state vector
            subsystem_qubits: Indices of qubits in subsystem A
            total_qubits: Total number of qubits

        Returns:
            Entanglement entropy
        """
        # Reshape to tensor
        dim_A = 2 ** len(subsystem_qubits)
        dim_B = 2 ** (total_qubits - len(subsystem_qubits))

        # This requires proper partial trace - simplified here
        # Full implementation would need to account for qubit ordering

        rho = np.outer(psi, np.conj(psi))

        # Compute partial trace (simplified)
        # Assuming subsystem_qubits are first qubits
        rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_A = np.trace(rho_reshaped, axis1=1, axis2=3)

        # Compute entropy
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove zeros

        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))

        return entropy

    def test_area_law(
        self,
        entropies: np.ndarray,
        boundary_sizes: np.ndarray
    ) -> HolographicTestResult:
        """
        Test area law: S(A) ~ |∂A|

        Args:
            entropies: Entanglement entropies
            boundary_sizes: Sizes of subsystem boundaries

        Returns:
            Test result
        """
        mask = np.isfinite(entropies) & (boundary_sizes > 0)
        S = entropies[mask]
        B = boundary_sizes[mask]

        if len(S) < 5:
            return HolographicTestResult(
                test_name="area_law",
                prediction="S(A) ~ |∂A|",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Insufficient data"
            )

        # Check for constant arrays (no variance) - regression undefined
        if np.std(B) < 1e-10 or np.std(S) < 1e-10:
            return HolographicTestResult(
                test_name="area_law",
                prediction="S(A) ~ |∂A|",
                observed_correlation=0.0,
                p_value=1.0,
                passed=False,
                details="Constant values - cannot compute correlation"
            )

        r, p = stats.pearsonr(B, S)
        slope, intercept, _, _, std_err = stats.linregress(B, S)

        passed = (r > 0.5 and p < 0.05)

        return HolographicTestResult(
            test_name="area_law",
            prediction="S(A) ~ |∂A| (Ryu-Takayanagi)",
            observed_correlation=r,
            p_value=p,
            passed=passed,
            fit_params={'slope': slope, 'intercept': intercept},
            details=f"Linear fit: S = {slope:.3f}|∂A| + {intercept:.3f}"
        )


if __name__ == '__main__':
    print("Testing HolographicDictionary...")

    # Create synthetic data with holographic correlations
    np.random.seed(42)
    n_samples = 50

    # Geometric data
    depths = np.random.randint(1, 6, n_samples).astype(float)
    distances = 2 * depths + np.random.randn(n_samples) * 0.5  # Correlated with depth
    n_physical = 10 * depths + np.random.randint(0, 5, n_samples)
    n_logical = np.ones(n_samples)  # Typically 1 for HaPPY

    geometric = {
        'depth': depths,
        'distance': distances,
        'n_physical': n_physical,
        'n_logical': n_logical
    }

    # Dynamic data (correlated with geometry)
    growth_exp = 1.0 + 0.3 * depths + np.random.randn(n_samples) * 0.1
    sat_value = 5 * n_logical + np.random.randn(n_samples) * 0.5
    krylov_dim = 3 * depths + np.random.randint(5, 15, n_samples)

    dynamic = {
        'growth_exponent': growth_exp,
        'saturation_value': sat_value,
        'krylov_dim': krylov_dim.astype(float)
    }

    # Hamiltonian data
    hamiltonian = {
        'temperature': np.random.uniform(0.5, 2.0, n_samples),
        'field_strength': np.random.uniform(0.0, 2.0, n_samples)
    }

    # Run tests
    dictionary = HolographicDictionary()
    results = dictionary.run_all_tests(geometric, dynamic, hamiltonian)

    print(dictionary.summary_report())
