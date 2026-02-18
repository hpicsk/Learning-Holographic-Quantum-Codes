"""
Visualization Utilities

Publication-quality plots for:
- Correlation matrices
- Krylov complexity curves
- Poincare disk embeddings
- Phase diagrams
- Holographic test results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from typing import Dict, List, Tuple, Optional, Any
import os


# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})


def plot_correlation_matrix(
    correlation_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Geometry-Complexity Correlations",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r',
    annotate: bool = True,
    p_values: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Plot correlation matrix as heatmap.

    Args:
        correlation_matrix: (n_row, n_col) correlation values
        row_labels: Labels for rows (geometric features)
        col_labels: Labels for columns (dynamic features)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
        annotate: Whether to annotate cells with values
        p_values: Optional p-value matrix for significance stars

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(correlation_matrix, cmap=cmap, aspect='auto',
                   vmin=-1, vmax=1)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation (r)', rotation=270, labelpad=20)

    # Ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    # Annotate cells
    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = correlation_matrix[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                text = f'{value:.2f}'

                # Add significance stars
                if p_values is not None:
                    if p_values[i, j] < 0.001:
                        text += '***'
                    elif p_values[i, j] < 0.01:
                        text += '**'
                    elif p_values[i, j] < 0.05:
                        text += '*'

                ax.text(j, i, text, ha='center', va='center',
                       color=color, fontsize=10)

    ax.set_title(title)
    ax.set_xlabel('Dynamic Features')
    ax.set_ylabel('Geometric Features')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()

    return fig


def plot_complexity_curves(
    times: np.ndarray,
    complexity_curves: List[np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = "Krylov Complexity Dynamics",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = False,
    show_theory: bool = False
) -> plt.Figure:
    """
    Plot multiple Krylov complexity curves.

    Args:
        times: Time array
        complexity_curves: List of C_K(t) arrays
        labels: Legend labels for each curve
        title: Plot title
        save_path: Path to save figure
        log_scale: Use log-log scale
        show_theory: Overlay theoretical predictions

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(complexity_curves)))

    for i, curve in enumerate(complexity_curves):
        label = labels[i] if labels else f'Code {i+1}'
        ax.plot(times, curve, color=colors[i], label=label, linewidth=2)

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time t (log scale)')
        ax.set_ylabel(r'$C_K(t)$ (log scale)')
    else:
        ax.set_xlabel('Time t')
        ax.set_ylabel(r'Krylov Complexity $C_K(t)$')

    # Show theoretical prediction
    if show_theory and not log_scale:
        # Early time: C_K ~ t^2
        t_early = times[times < times[-1] * 0.2]
        c_early = 0.5 * t_early ** 2
        ax.plot(t_early, c_early, 'k--', alpha=0.5, label=r'$\sim t^2$ (theory)')

    ax.set_title(title)
    ax.legend(loc='lower right', ncol=2 if len(complexity_curves) > 4 else 1)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()

    return fig


def plot_poincare_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    title: str = "Poincare Disk Embeddings",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = 'viridis',
    show_geodesics: bool = False
) -> plt.Figure:
    """
    Visualize embeddings in the Poincare disk.

    Args:
        embeddings: (n, 2) array of 2D Poincare disk coordinates
        labels: Optional (n,) array of labels for coloring
        label_names: Optional dict mapping label values to names
        title: Plot title
        save_path: Path to save figure
        show_geodesics: Draw geodesic circles

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw unit circle boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.fill(np.cos(theta), np.sin(theta), color='lightgray', alpha=0.1)

    # Draw geodesic reference circles
    if show_geodesics:
        for r in [0.3, 0.5, 0.7, 0.9]:
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                   'k--', alpha=0.2, linewidth=0.5)

    # Plot points
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))

        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names.get(lab, f'Class {lab}') if label_names else f'{lab}'
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=[colors[i]], s=80, alpha=0.7, label=name,
                      edgecolors='white', linewidth=0.5)

        ax.legend(loc='upper right')
    else:
        ax.scatter(embeddings[:, 0], embeddings[:, 1],
                  c='steelblue', s=80, alpha=0.7,
                  edgecolors='white', linewidth=0.5)

    # Annotations
    ax.annotate('Boundary\n(Physical)', xy=(0.85, -0.1), fontsize=10, ha='center')
    ax.annotate('Bulk\n(Logical)', xy=(0, 0), fontsize=10, ha='center')

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    # Remove ticks for cleaner look
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()

    return fig


def plot_phase_diagram(
    param1_values: np.ndarray,
    param2_values: np.ndarray,
    order_parameter: np.ndarray,
    param1_name: str = r'$\Delta$',
    param2_name: str = r'$h$',
    title: str = "Phase Diagram",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'plasma'
) -> plt.Figure:
    """
    Plot 2D phase diagram.

    Args:
        param1_values: Values for first parameter (x-axis)
        param2_values: Values for second parameter (y-axis)
        order_parameter: 2D array of order parameter values
        param1_name: Label for x-axis
        param2_name: Label for y-axis
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create meshgrid if needed
    if order_parameter.ndim == 1:
        n1 = len(param1_values)
        n2 = len(param2_values)
        order_parameter = order_parameter.reshape(n2, n1)

    X, Y = np.meshgrid(param1_values, param2_values)

    # Plot heatmap
    im = ax.pcolormesh(X, Y, order_parameter, cmap=cmap, shading='auto')

    # Contour lines
    contours = ax.contour(X, Y, order_parameter, levels=5,
                          colors='white', alpha=0.5, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Order Parameter', rotation=270, labelpad=20)

    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()

    return fig


def plot_holographic_tests(
    test_results: List,
    title: str = "Holographic Dictionary Tests",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize results of holographic dictionary tests.

    Args:
        test_results: List of HolographicTestResult objects
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Bar chart of correlations
    ax1 = axes[0]
    names = [r.test_name.replace('_', ' ').title() for r in test_results]
    correlations = [r.observed_correlation for r in test_results]
    colors = ['green' if r.passed else 'red' for r in test_results]

    bars = ax1.barh(names, correlations, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax1.set_xlabel('Correlation (r)')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_title('Test Correlations')

    # Add pass/fail labels
    for bar, result in zip(bars, test_results):
        label = 'PASS' if result.passed else 'FAIL'
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=10,
                color='green' if result.passed else 'red')

    # Right: Summary statistics
    ax2 = axes[1]
    ax2.axis('off')

    n_passed = sum(1 for r in test_results if r.passed)
    n_total = len(test_results)

    summary_text = f"HOLOGRAPHIC CORRESPONDENCE TESTS\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += f"Tests Passed: {n_passed}/{n_total}\n\n"

    for result in test_results:
        status = "[PASS]" if result.passed else "[FAIL]"
        summary_text += f"{status} {result.test_name}\n"
        summary_text += f"       Prediction: {result.prediction[:40]}...\n"
        summary_text += f"       r = {result.observed_correlation:.3f}, p = {result.p_value:.2e}\n\n"

    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot training history (losses, metrics over epochs).

    Args:
        history: Dict mapping metric names to lists of values
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    n_metrics = len(history)
    fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, list(history.items())[:3]):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, 'b-', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(name.replace('_', ' ').title())
        ax.set_title(name.replace('_', ' ').title())

        # Add min/max annotations
        min_val = min(values)
        max_val = max(values)
        min_epoch = values.index(min_val) + 1

        if 'loss' in name.lower():
            ax.axhline(y=min_val, color='r', linestyle='--', alpha=0.5)
            ax.text(len(values) * 0.7, min_val * 1.1,
                   f'Best: {min_val:.4f} (epoch {min_epoch})',
                   fontsize=9, color='red')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()

    return fig


def create_figure_grid(
    figures: List[plt.Figure],
    n_cols: int = 2,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Combine multiple figures into a grid layout.

    Note: This is a simplified version - for publication,
    use subfigures or manual layout.
    """
    n_figs = len(figures)
    n_rows = (n_figs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, ax in enumerate(axes):
        if i < n_figs:
            # Copy figure content (simplified)
            ax.set_title(f'Panel {i+1}')
        else:
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()

    return fig


def ensure_output_dir(path: str):
    """Ensure output directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


if __name__ == '__main__':
    print("Testing visualization utilities...")

    # Create output directory
    os.makedirs('results/test_plots', exist_ok=True)

    # Test correlation matrix
    print("Generating correlation matrix plot...")
    corr_matrix = np.random.randn(4, 5) * 0.5
    corr_matrix = np.clip(corr_matrix, -1, 1)
    row_labels = ['distance', 'rate', 'depth', 'weight']
    col_labels = ['growth_exp', 'sat_value', 'sat_time', 'krylov_dim', 'max_C']

    plot_correlation_matrix(
        corr_matrix, row_labels, col_labels,
        save_path='results/test_plots/correlation_matrix.png'
    )

    # Test complexity curves
    print("Generating complexity curves plot...")
    times = np.linspace(0, 10, 100)
    curves = [np.tanh(0.5 * i * times) * i * 2 for i in range(1, 5)]
    labels = [f'Depth {i}' for i in range(1, 5)]

    plot_complexity_curves(
        times, curves, labels,
        save_path='results/test_plots/complexity_curves.png'
    )

    # Test Poincare embeddings
    print("Generating Poincare embeddings plot...")
    n_points = 50
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    radii = np.random.uniform(0.1, 0.9, n_points)
    embeddings = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    labels = np.random.randint(0, 3, n_points)

    plot_poincare_embeddings(
        embeddings, labels,
        label_names={0: 'HaPPY', 1: 'LDPC', 2: 'Random'},
        save_path='results/test_plots/poincare_embeddings.png',
        show_geodesics=True
    )

    print("Test plots saved to results/test_plots/")
