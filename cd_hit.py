# <------ CELL 1 ------>
import os
import subprocess
import sys
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up paths
input_file = r"DNA_Sequencing\assets\SarsCov2SpikemRNA.fasta"
results_dir = r"DNA_Sequencing\results\cd_hit_results"

# Ensure results directory exists
Path(results_dir).mkdir(parents=True, exist_ok=True)

print("Setup completed!")
print(f"Input file: {input_file}")
print(f"Results directory: {results_dir}")

# Check if input file exists
if os.path.exists(input_file):
    print(f"✓ Input file found")
else:
    print(f"✗ Input file not found at {input_file}")
    print("Please check the file path and make sure the file exists.")


# Install required packages
packages = ['biopython', 'pandas', 'matplotlib', 'seaborn', 'numpy']

for package in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"✓ {package} installed/updated")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")

# Try to install vsearch
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vsearch'])
    print("✓ vsearch installed successfully")
except subprocess.CalledProcessError:
    print("✗ vsearch pip package not available")
    print("Will use vsearch binary directly if available")

# Check if vsearch binary is available
try:
    result = subprocess.run(["vsearch", "--version"], capture_output=True, text=True, timeout=10)
    print("✓ VSEARCH binary is available")
    print(f"Version: {result.stderr.strip()}")
except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
    print("✗ VSEARCH binary not found")
    print("Please install VSEARCH from: https://github.com/torognes/vsearch")
    print("Or download pre-compiled binary")



# <------ CELL 3 ------>
def analyze_fasta_file(file_path):
    """Analyze the input FASTA file to understand the data"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    
    sequences = list(SeqIO.parse(file_path, "fasta"))
    
    print(f"Total sequences: {len(sequences)}")
    
    # Get sequence lengths
    seq_lengths = [len(seq.seq) for seq in sequences]
    
    print(f"Sequence length statistics:")
    print(f"  Min length: {min(seq_lengths)}")
    print(f"  Max length: {max(seq_lengths)}")
    print(f"  Average length: {sum(seq_lengths)/len(seq_lengths):.2f}")
    
    # Show first few sequence headers
    print(f"\nFirst 5 sequence headers:")
    for i, seq in enumerate(sequences[:5]):
        print(f"  {i+1}: {seq.id}")
    
    # Check sequence type (DNA vs RNA)
    sample_seq = str(sequences[0].seq).upper()
    has_u = 'U' in sample_seq
    has_t = 'T' in sample_seq
    
    if has_u and not has_t:
        seq_type = "RNA"
    elif has_t and not has_u:
        seq_type = "DNA"
    else:
        seq_type = "Mixed/Unknown"
    
    print(f"\nSequence type detected: {seq_type}")
    
    return sequences, seq_lengths

# Analyze input file
sequences, seq_lengths = analyze_fasta_file(input_file)


# <------ CELL 4 ------>
def run_vsearch_clustering(input_file, output_prefix, identity_threshold=0.9):
    """
    Run VSEARCH clustering on sequences
    
    Parameters:
    - input_file: path to input FASTA file
    - output_prefix: prefix for output files
    - identity_threshold: sequence identity threshold (0.0-1.0)
    """
    
    centroids_file = f"{output_prefix}_centroids.fasta"
    clusters_file = f"{output_prefix}_clusters.uc"
    
    # VSEARCH clustering command
    cmd = [
        "vsearch",
        "--cluster_fast", input_file,
        "--id", str(identity_threshold),
        "--centroids", centroids_file,
        "--uc", clusters_file,
        "--sizein",
        "--sizeout",
        "--threads", "4"
    ]
    
    try:
        print(f"Running VSEARCH clustering at {identity_threshold*100}% identity...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ VSEARCH clustering completed successfully")
        print(f"✓ Centroids saved to: {centroids_file}")
        print(f"✓ Cluster info saved to: {clusters_file}")
        return True, centroids_file, clusters_file
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running VSEARCH: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False, None, None
    except FileNotFoundError:
        print("✗ VSEARCH not found in PATH")
        print("Please install VSEARCH from: https://github.com/torognes/vsearch")
        return False, None, None

def parse_vsearch_clusters(clusters_file):
    """Parse VSEARCH .uc file to extract cluster information"""
    clusters = defaultdict(list)
    
    try:
        with open(clusters_file, 'r') as f:
            for line in f:
                if line.startswith('S') or line.startswith('H'):
                    parts = line.strip().split('\t')
                    cluster_id = int(parts[1])
                    seq_id = parts[8]
                    record_type = parts[0]
                    
                    clusters[cluster_id].append({
                        'seq_id': seq_id,
                        'is_centroid': record_type == 'S'
                    })
    except FileNotFoundError:
        print(f"Cluster file not found: {clusters_file}")
        return None
    
    return clusters



# <------ CELL 5 ------>
# Run clustering at different identity thresholds
identity_thresholds = [0.95, 0.90, 0.85, 0.80]
clustering_results = {}

for threshold in identity_thresholds:
    print(f"\n{'='*50}")
    print(f"CLUSTERING AT {threshold*100}% IDENTITY")
    print(f"{'='*50}")
    
    output_prefix = os.path.join(results_dir, f"vsearch_{int(threshold*100)}")
    
    success, centroids_file, clusters_file = run_vsearch_clustering(
        input_file, output_prefix, threshold
    )
    
    if success:
        # Parse clustering results
        clusters = parse_vsearch_clusters(clusters_file)
        
        if clusters:
            num_clusters = len(clusters)
            cluster_sizes = [len(cluster) for cluster in clusters.values()]
            
            clustering_results[threshold] = {
                'num_clusters': num_clusters,
                'cluster_sizes': cluster_sizes,
                'centroids_file': centroids_file,
                'clusters_file': clusters_file
            }
            
            print(f"✓ Found {num_clusters} clusters")
            print(f"✓ Cluster size range: {min(cluster_sizes)}-{max(cluster_sizes)}")
            print(f"✓ Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.2f}")
        else:
            print("✗ Failed to parse cluster results")
    else:
        print("✗ Clustering failed")

print(f"\n{'='*50}")
print("CLUSTERING SUMMARY")
print(f"{'='*50}")
for threshold, results in clustering_results.items():
    print(f"{threshold*100}% identity: {results['num_clusters']} clusters")



# <------ CELL 6 ------>
def analyze_clustering_results(clustering_results, total_sequences):
    """Analyze and summarize clustering results"""
    analysis_data = []
    
    for threshold, results in clustering_results.items():
        cluster_sizes = results['cluster_sizes']
        
        analysis = {
            'threshold': threshold,
            'num_clusters': results['num_clusters'],
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            'singletons': sum(1 for size in cluster_sizes if size == 1),
            'reduction_ratio': (1 - results['num_clusters'] / total_sequences) * 100,
            'largest_clusters': sorted(cluster_sizes, reverse=True)[:5]
        }
        
        analysis_data.append(analysis)
        
        print(f"\nThreshold {threshold*100}%:")
        print(f"  Clusters: {analysis['num_clusters']}")
        print(f"  Size range: {analysis['min_cluster_size']}-{analysis['max_cluster_size']}")
        print(f"  Average size: {analysis['avg_cluster_size']:.2f}")
        print(f"  Singletons: {analysis['singletons']}")
        print(f"  Reduction: {analysis['reduction_ratio']:.1f}%")
        print(f"  Top 5 cluster sizes: {analysis['largest_clusters']}")
    
    return analysis_data

# Analyze results if we have sequences
if sequences and clustering_results:
    print("=== DETAILED CLUSTERING ANALYSIS ===")
    analysis_data = analyze_clustering_results(clustering_results, len(sequences))
else:
    print("No clustering results to analyze")
    analysis_data = []


# <------ CELL 7 ------>
def create_clustering_visualizations(analysis_data, results_dir):
    """Create comprehensive visualizations of clustering results"""
    
    if not analysis_data:
        print("No data to visualize")
        return
    
    df = pd.DataFrame(analysis_data)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Number of clusters vs threshold
    ax1.plot(df['threshold']*100, df['num_clusters'], 'bo-', linewidth=3, markersize=10)
    ax1.set_xlabel('Identity Threshold (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax1.set_title('Clusters vs Identity Threshold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot 2: Average cluster size vs threshold
    ax2.plot(df['threshold']*100, df['avg_cluster_size'], 'ro-', linewidth=3, markersize=10)
    ax2.set_xlabel('Identity Threshold (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Cluster Size', fontsize=12, fontweight='bold')
    ax2.set_title('Average Cluster Size vs Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    # Plot 3: Singleton clusters
    bars = ax3.bar(df['threshold']*100, df['singletons'], alpha=0.8, color='green', width=2)
    ax3.set_xlabel('Identity Threshold (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Singleton Clusters', fontsize=12, fontweight='bold')
    ax3.set_title('Singleton Clusters vs Threshold', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Reduction ratio
    ax4.plot(df['threshold']*100, df['reduction_ratio'], 'mo-', linewidth=3, markersize=10)
    ax4.set_xlabel('Identity Threshold (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Data Reduction (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Data Reduction vs Threshold', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    plot_file = os.path.join(results_dir, "vsearch_clustering_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {plot_file}")
    plt.show()
    
    # Create cluster size distribution plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for threshold, results in clustering_results.items():
        cluster_sizes = results['cluster_sizes']
        if cluster_sizes:
            # Create histogram
            bins = range(1, max(cluster_sizes) + 2)
            counts, _ = np.histogram(cluster_sizes, bins=bins)
            ax.plot(bins[:-1], counts, 'o-', label=f'{threshold*100}% identity', 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Cluster Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # Save cluster distribution plot
    dist_plot_file = os.path.join(results_dir, "cluster_size_distribution.png")
    plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Cluster distribution plot saved to: {dist_plot_file}")
    plt.show()

# Create visualizations
if analysis_data:
    create_clustering_visualizations(analysis_data, results_dir)
else:
    print("No data available for visualization")


# <------ CELL 8 ------>
def generate_comprehensive_report(analysis_data, sequences, seq_lengths, results_dir):
    """Generate a comprehensive analysis report"""
    
    report_file = os.path.join(results_dir, "VSEARCH_clustering_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("VSEARCH SEQUENCE CLUSTERING ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Input data summary
        f.write("INPUT DATA SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input file: {input_file}\n")
        if sequences:
            f.write(f"Total sequences: {len(sequences)}\n")
            f.write(f"Sequence length range: {min(seq_lengths)}-{max(seq_lengths)} bp\n")
            f.write(f"Average sequence length: {sum(seq_lengths)/len(seq_lengths):.2f} bp\n")
            f.write(f"Median sequence length: {np.median(seq_lengths):.2f} bp\n")
        f.write("\n")
        
        # Clustering results
        f.write("CLUSTERING RESULTS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        if analysis_data:
            for data in analysis_data:
                f.write(f"Identity Threshold: {data['threshold']*100}%\n")
                f.write(f"  Number of clusters: {data['num_clusters']}\n")
                f.write(f"  Cluster size range: {data['min_cluster_size']}-{data['max_cluster_size']}\n")
                f.write(f"  Average cluster size: {data['avg_cluster_size']:.2f}\n")
                f.write(f"  Singleton clusters: {data['singletons']}\n")
                f.write(f"  Data reduction: {data['reduction_ratio']:.1f}%\n")
                f.write(f"  Top 5 cluster sizes: {data['largest_clusters']}\n\n")
        else:
            f.write("No clustering results available.\n\n")
        
        # Files generated
        f.write("OUTPUT FILES GENERATED:\n")
        f.write("-" * 25 + "\n")
        for threshold in identity_thresholds:
            prefix = f"vsearch_{int(threshold*100)}"
            f.write(f"• {prefix}_centroids.fasta - Representative sequences at {threshold*100}% identity\n")
            f.write(f"• {prefix}_clusters.uc - Cluster membership information\n")
        f.write("• vsearch_clustering_analysis.png - Analysis visualization\n")
        f.write("• cluster_size_distribution.png - Cluster size distribution\n")
        f.write("• VSEARCH_clustering_report.txt - This comprehensive report\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        if analysis_data:
            # Find optimal threshold
            best_threshold = None
            best_score = 0
            for data in analysis_data:
                # Score based on reasonable number of clusters and good reduction
                score = data['reduction_ratio'] * (1 - abs(data['num_clusters'] - len(sequences)*0.1)/len(sequences))
                if score > best_score:
                    best_score = score
                    best_threshold = data['threshold']
            
            if best_threshold:
                f.write(f"• Recommended threshold: {best_threshold*100}% identity\n")
                f.write(f"  - Provides good balance of clustering and data reduction\n")
            
            f.write("• Use higher thresholds (95%) for very similar sequences\n")
            f.write("• Use lower thresholds (80-85%) for more diverse clustering\n")
        else:
            f.write("• Check input file path and VSEARCH installation\n")
            f.write("• Ensure sequences are in proper FASTA format\n")
        
        f.write(f"\nAnalysis completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✓ Comprehensive report saved to: {report_file}")

# Create summary statistics table
def create_summary_table(analysis_data, results_dir):
    """Create a CSV summary table"""
    if not analysis_data:
        return
    
    df = pd.DataFrame(analysis_data)
    summary_file = os.path.join(results_dir, "clustering_summary_table.csv")
    df.to_csv(summary_file, index=False)
    print(f"✓ Summary table saved to: {summary_file}")

# Generate final report and summary
if sequences:
    generate_comprehensive_report(analysis_data, sequences, seq_lengths, results_dir)
    create_summary_table(analysis_data, results_dir)
else:
    print("Cannot generate report - no sequence data available")

print(f"\n{'='*60}")
print("VSEARCH CLUSTERING ANALYSIS COMPLETED!")
print(f"{'='*60}")
print(f"All results saved in: {results_dir}")
print("\nGenerated files:")
print("• Clustered FASTA files (centroids) at different identity thresholds")
print("• Cluster membership files (.uc format)")
print("• Analysis visualizations (PNG)")
print("• Comprehensive report (TXT)")
print("• Summary table (CSV)")
print(f"{'='*60}")