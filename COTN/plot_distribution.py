import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

# Set style for better visualization
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Baseline values configuration
BASELINE_VALUES = {
    'mae': 0.6263,  # Example value, please modify
    'mse': 0.6631,  # Example value, please modify
    'rmse': 0.6236,  # Example value, please modify
    'mape': 0.0891,  # Example value, please modify
    'mspe': 0.0736   # Example value, please modify
}

def extract_lee_number(folder_name):
    """Extract lee number from folder name"""
    match = re.search(r'lee(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None

def load_metrics_data(results_dir, lee_prefix=None):
    metrics_data = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'mape': [],
        'mspe': []
    }
    
    folders = []
    # Traverse all folders in results directory
    for folder in os.listdir(results_dir):
        if folder.startswith('informer_ETTh1_lee'):
            lee_num = extract_lee_number(folder)
            if lee_num is not None:
                if lee_prefix is not None:
                    first_digit = str(lee_num)[0]
                    if first_digit != str(lee_prefix):
                        continue
                folders.append((folder, lee_num))
    
    # Sort by lee number
    folders.sort(key=lambda x: x[1])
    
    # Load sorted data
    for folder, _ in folders:
        try:
            metrics_file = os.path.join(results_dir, folder, 'metrics.npy')
            if os.path.exists(metrics_file):
                metrics = np.load(metrics_file)
                metrics_data['mae'].append(metrics[0])
                metrics_data['mse'].append(metrics[1])
                metrics_data['rmse'].append(metrics[2])
                metrics_data['mape'].append(metrics[3])
                metrics_data['mspe'].append(metrics[4])
        except Exception as e:
            print(f"Error loading {folder}: {e}")
    
    return metrics_data

def plot_single_distribution(data, metric_name, remove_outliers=False, lee_prefix=None):
    """Plot distribution for a single metric"""
    # Create new figure
    plt.figure()
    
    # Process data
    if remove_outliers:
        mean = np.mean(data)
        std = np.std(data)
        plot_data = data[np.abs(data - mean) <= 3 * std]
        outliers_info = f"(Filtered: {len(plot_data)}/{len(data)} samples)"
    else:
        plot_data = data
        outliers_info = "(All samples)"
    
    # Calculate statistics
    mean = np.mean(plot_data)
    std = np.std(plot_data)
    median = np.median(plot_data)
    ci_lower = np.percentile(plot_data, 2.5)
    ci_upper = np.percentile(plot_data, 97.5)
    
    # Plot distribution
    sns.kdeplot(data=plot_data, fill=True, alpha=0.5)
    
    # Add vertical lines for mean and median
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
    plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.4f}')
    
    # Add baseline vertical line if available
    if metric_name in BASELINE_VALUES:
        baseline = BASELINE_VALUES[metric_name]
        plt.axvline(baseline, color='blue', linestyle=':', linewidth=2, 
                   label=f'Baseline: {baseline:.4f}')
        
        # Calculate improvement percentage
        improvement = ((baseline - mean) / baseline) * 100
        improvement_text = f'Improvement: {improvement:.2f}%'
    else:
        improvement_text = 'No baseline available'
    
    # Add labels and title
    prefix_info = f"Lee{lee_prefix} " if lee_prefix is not None else "All Lee "
    plt.title(f'Distribution of {metric_name.upper()} - {prefix_info}Series {outliers_info}', 
              fontsize=12, pad=20)
    plt.xlabel(f'{metric_name.upper()} Value')
    plt.ylabel('Density')
    
    # Add statistics text box
    stats_text = (f'Standard Dev: {std:.4f}\n'
                 f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n'
                 f'{improvement_text}')
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize legend
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    prefix_str = f'_lee{lee_prefix}' if lee_prefix is not None else '_all'
    filename = f'{metric_name}_distribution{prefix_str}_{"filtered" if remove_outliers else "full"}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def print_statistics(metrics_data, lee_prefix=None):
    prefix_info = f"Lee{lee_prefix}" if lee_prefix is not None else "All Lee"
    print(f"\nStatistics Summary ({prefix_info} Series):")
    for metric in metrics_data:
        data = np.array(metrics_data[metric])
        mean = np.mean(data)
        print(f"\n{metric.upper()}:")
        print(f"Sample Count: {len(data)}")
        print(f"Mean: {mean:.4f}")
        print(f"Median: {np.median(data):.4f}")
        print(f"Std Dev: {np.std(data):.4f}")
        print(f"Min: {np.min(data):.4f}")
        print(f"Max: {np.max(data):.4f}")
        
        # Print improvement over baseline if available
        if metric in BASELINE_VALUES:
            baseline = BASELINE_VALUES[metric]
            improvement = ((baseline - mean) / baseline) * 100
            print(f"Improvement over baseline: {improvement:.2f}%")

def main():
    results_dir = './results'
    lee_prefixes = [1, 2, 3, None]
    
    # Allow manual input of baseline values
    print("Current baseline values:")
    for metric, value in BASELINE_VALUES.items():
        print(f"{metric.upper()}: {value}")
    
    print("\nWould you like to modify baseline values? (y/n)")
    if input().lower() == 'y':
        for metric in BASELINE_VALUES.keys():
            try:
                new_value = float(input(f"Enter new baseline value for {metric.upper()} (current: {BASELINE_VALUES[metric]}): "))
                BASELINE_VALUES[metric] = new_value
            except ValueError:
                print(f"Invalid input, keeping current value for {metric}")
    
    for lee_prefix in lee_prefixes:
        prefix_info = f"Lee{lee_prefix}" if lee_prefix is not None else "All Lee"
        print(f"\nProcessing {prefix_info} series data...")
        
        # Load data
        metrics_data = load_metrics_data(results_dir, lee_prefix)
        
        if not any(metrics_data.values()):
            print(f"No data found for {prefix_info} series, skipping...")
            continue
        
        # Generate plots for each metric
        for metric in metrics_data.keys():
            data = np.array(metrics_data[metric])
            
            print(f"Plotting {metric.upper()} distribution (full data)...")
            plot_single_distribution(data, metric, remove_outliers=False, lee_prefix=lee_prefix)
            
            print(f"Plotting {metric.upper()} distribution (filtered data)...")
            plot_single_distribution(data, metric, remove_outliers=True, lee_prefix=lee_prefix)
        
        # Print statistics
        print_statistics(metrics_data, lee_prefix)
    
    print("\nAnalysis complete! All plots have been saved.")

if __name__ == "__main__":
    main() 