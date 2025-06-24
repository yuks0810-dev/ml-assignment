#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Curse of Dimensionality Analysis
========================================

Enhanced analysis focusing on meaningful curse of dimensionality metrics instead of sparsity.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import math
import warnings
warnings.filterwarnings('ignore')

# Font and plot settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')
sns.set_palette("husl")

# Reproducibility seed
np.random.seed(42)

def generate_synthetic_dataset(n_samples, n_features, random_state=42):
    """Generate synthetic classification dataset"""
    n_informative = min(10, n_features)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=5,
        class_sep=1.0,
        random_state=random_state
    )
    
    return X, y

def analyze_dimensionality_effects(dimensions, n_samples=1000):
    """Analyze curse of dimensionality effects across different dimensions"""
    analysis_results = {}

    for dim in dimensions:
        print(f"Analyzing dimension {dim}...")

        # Generate dataset
        X, y = generate_synthetic_dataset(n_samples, dim)

        # Distance distribution analysis (subset sampling for computational efficiency)
        sample_size = min(200, n_samples)
        indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[indices]

        # Calculate pairwise distances
        distances = pdist(X_sample, metric='euclidean')

        # Distance concentration metrics
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        cv_distance = std_distance / mean_distance if mean_distance > 0 else 0

        # Nearest neighbor analysis
        nn_model = NearestNeighbors(n_neighbors=2)
        nn_model.fit(X_sample)
        nn_distances, nn_indices = nn_model.kneighbors(X_sample)
        nn_distances = nn_distances[:, 1]  # Exclude self (distance 0)
        
        # Volume and density effects
        # Approximate volume of unit hypersphere in d dimensions
        if dim % 2 == 0:
            unit_sphere_volume = math.pi ** (dim/2) / math.factorial(dim//2)
        else:
            unit_sphere_volume = (2 ** ((dim+1)/2) * math.pi ** ((dim-1)/2) * 
                                math.factorial((dim-1)//2)) / math.factorial(dim)
        
        # Effective density (samples per unit volume)
        data_range = np.max(X_sample) - np.min(X_sample)
        effective_volume = data_range ** dim
        effective_density = len(X_sample) / effective_volume if effective_volume > 0 else 0
        
        # Distance ratio analysis (max/min distance ratio)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        distance_ratio = max_distance / min_distance if min_distance > 0 else float('inf')
        
        # Hubness phenomenon (concentration of nearest neighbors)
        nn_counts = np.bincount(nn_indices.flatten(), minlength=len(X_sample))
        nn_skewness = np.std(nn_counts) / np.mean(nn_counts) if np.mean(nn_counts) > 0 else 0

        analysis_results[dim] = {
            'dimension': dim,
            'distances': distances,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'cv_distance': cv_distance,
            'distance_concentration': 1 - cv_distance,
            'nn_distances': nn_distances,
            'mean_nn_distance': np.mean(nn_distances),
            'std_nn_distance': np.std(nn_distances),
            'unit_sphere_volume': unit_sphere_volume,
            'effective_density': effective_density,
            'distance_ratio': distance_ratio,
            'nn_skewness': nn_skewness,
            'min_distance': min_distance,
            'max_distance': max_distance
        }

    return analysis_results

def visualize_dimensionality_effects(analysis_results):
    """Generate comprehensive visualization for curse of dimensionality effects"""
    dimensions = list(analysis_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Effective Data Density by Dimension (replaces useless sparsity plot)
    ax1 = axes[0, 0]
    densities = [analysis_results[dim]['effective_density'] for dim in dimensions]

    bars = ax1.bar(range(len(dimensions)), densities, 
                   color=plt.cm.plasma(np.linspace(0, 1, len(dimensions))))
    ax1.set_xticks(range(len(dimensions)))
    ax1.set_xticklabels([f'{dim}D' for dim in dimensions])
    ax1.set_ylabel('Effective Data Density')
    ax1.set_title('Data Sparsity: Effective Density by Dimension\n(Shows exponential decrease with dimension)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    for bar, value in zip(bars, densities):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                     f'{value:.1e}', ha='center', va='bottom', fontsize=9, rotation=45)

    # 2. Nearest Neighbor Distance Distribution (more informative than general distances)
    ax2 = axes[0, 1]

    # Compare low vs high dimension
    low_dim = min(dimensions)
    high_dim = max(dimensions)
    
    nn_distances_low = analysis_results[low_dim]['nn_distances']
    nn_distances_high = analysis_results[high_dim]['nn_distances']

    ax2.hist(nn_distances_low, bins=25, alpha=0.6, label=f'{low_dim}D', density=True, color='blue')
    ax2.hist(nn_distances_high, bins=25, alpha=0.6, label=f'{high_dim}D', density=True, color='red')
    ax2.set_xlabel('Nearest Neighbor Distance')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Nearest Neighbor Distance Distribution\n({low_dim}D vs {high_dim}D)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distance Concentration and Hubness Effects
    ax3 = axes[1, 0]

    distance_ratios = [analysis_results[dim]['distance_ratio'] for dim in dimensions]
    hubness_scores = [analysis_results[dim]['nn_skewness'] for dim in dimensions]

    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(dimensions, distance_ratios, 'o-', label='Distance Ratio (Max/Min)', 
                     linewidth=2, markersize=8, color='red')
    line2 = ax3_twin.plot(dimensions, hubness_scores, 's-', label='Hubness Score', 
                          linewidth=2, markersize=8, color='green')
    
    ax3.set_xlabel('Number of Dimensions')
    ax3.set_ylabel('Distance Ratio (Max/Min)', color='red')
    ax3_twin.set_ylabel('Hubness Score', color='green')
    ax3.set_title('Distance Concentration and Hubness Effects\n(Key curse of dimensionality phenomena)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    # 4. Volume Growth vs Data Density (demonstrates the core problem)
    ax4 = axes[1, 1]

    sphere_volumes = [analysis_results[dim]['unit_sphere_volume'] for dim in dimensions]
    effective_densities = [analysis_results[dim]['effective_density'] for dim in dimensions]

    ax4_twin = ax4.twinx()
    
    line1 = ax4.semilogy(dimensions, sphere_volumes, 'o-', label='Unit Hypersphere Volume', 
                         linewidth=2, markersize=8, color='purple')
    line2 = ax4_twin.semilogy(dimensions, effective_densities, 's-', label='Effective Data Density', 
                              linewidth=2, markersize=8, color='orange')
    
    ax4.set_xlabel('Number of Dimensions')
    ax4.set_ylabel('Unit Hypersphere Volume', color='purple')
    ax4_twin.set_ylabel('Effective Data Density', color='orange')
    ax4.set_title('Volume Growth vs Data Density\n(Core mechanism of curse of dimensionality)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right')

    plt.tight_layout()
    plt.savefig('dimensionality_effects_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Output summary statistics
    print("\nSummary of Curse of Dimensionality Effects:")
    print("Dim | Density    | Dist Ratio | NN Dist   | Hubness  | Concentration")
    print("-" * 70)
    for dim in dimensions:
        stats = analysis_results[dim]
        print(f"{dim:3d} | {stats['effective_density']:10.2e} | {stats['distance_ratio']:10.2f} | "
              f"{stats['mean_nn_distance']:9.4f} | {stats['nn_skewness']:8.4f} | {stats['distance_concentration']:6.4f}")

    return analysis_results

def main():
    """Main execution function - demonstrates improved analysis"""
    print("Improved Curse of Dimensionality Analysis")
    print("=" * 50)
    
    # Test dimensions
    dimensions = [10, 50, 100, 200, 500]
    
    print("Executing improved dimensionality effects analysis...")
    results = analyze_dimensionality_effects(dimensions)
    visualize_dimensionality_effects(results)
    
    print("\nKey Improvements:")
    print("1. Replaced meaningless sparsity plot with effective data density")
    print("2. Added nearest neighbor distance analysis")
    print("3. Introduced hubness phenomenon measurement")
    print("4. Visualized volume growth vs density relationship")
    print("5. All plots now demonstrate actual curse of dimensionality effects")

if __name__ == "__main__":
    main()