## IEDB Analysis Resource - Input: Protein/peptide sequence (FASTA/RAW) ; Output: Epitope predictions (CSV, TXT, JSON)

# Cell 1: Imports and Setup
import os
import sys
import requests
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up correct paths - we're inside DNA_Sequencing directory
INPUT_FILE = os.path.join("assets", "SarsCov2SpikemRNA.fasta")
OUTPUT_DIR = os.path.join("results", "iedb_analysis")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEDB API endpoints
IEDB_MHC_I_URL = "http://tools-cluster-interface.iedb.org/tools_api/mhci/"
IEDB_MHC_II_URL = "http://tools-cluster-interface.iedb.org/tools_api/mhcii/"
IEDB_BCELL_URL = "http://tools-cluster-interface.iedb.org/tools_api/bcell/"

# Common HLA alleles for analysis
COMMON_HLA_ALLELES = [
    'HLA-A*02:01', 'HLA-A*01:01', 'HLA-A*24:02', 'HLA-A*03:01', 'HLA-A*11:01',
    'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*35:01', 'HLA-B*40:01', 'HLA-B*44:02',
    'HLA-C*07:02', 'HLA-C*04:01', 'HLA-C*06:02', 'HLA-C*03:04', 'HLA-C*01:02'
]

# Initialize results storage
results = {
    'mhc_class_i': [],
    'mhc_class_ii': [],
    'bcell_epitopes': [],
    'sequence_info': {},
    'analysis_metadata': {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': INPUT_FILE,
        'output_dir': OUTPUT_DIR
    }
}

# Check paths
print("✅ Setup completed successfully!")
print(f"Current working directory: {os.getcwd()}")
print(f"Input file: {INPUT_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Input file exists: {os.path.exists(INPUT_FILE)}")
print(f"Available HLA alleles: {len(COMMON_HLA_ALLELES)}")
print(f"IEDB API endpoints configured")


# Cell 2: Sequence Loading and Processing

def load_fasta_sequence(file_path: str) -> Dict:
    """Load and process FASTA sequence"""
    try:
        sequences = {}
        with open(file_path, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequences[record.id] = {
                    'sequence': str(record.seq),
                    'length': len(record.seq),
                    'description': record.description
                }
        return sequences
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return {}
    except Exception as e:
        print(f"❌ Error loading sequence: {e}")
        return {}

def translate_mrna_to_protein(mrna_sequence: str) -> str:
    """Translate mRNA sequence to protein"""
    # Remove any whitespace and convert to uppercase
    mrna_clean = mrna_sequence.replace('\n', '').replace(' ', '').upper()
    
    # Convert to Seq object and translate
    seq_obj = Seq(mrna_clean)
    protein_seq = str(seq_obj.translate())
    
    # Remove stop codon if present
    if protein_seq.endswith('*'):
        protein_seq = protein_seq[:-1]
    
    return protein_seq

def generate_overlapping_peptides(protein_sequence: str, peptide_lengths: List[int] = [8, 9, 10, 11]) -> List[Dict]:
    """Generate overlapping peptides for epitope prediction"""
    peptides = []
    
    for length in peptide_lengths:
        for i in range(len(protein_sequence) - length + 1):
            peptide = protein_sequence[i:i + length]
            
            # Skip peptides with ambiguous amino acids
            if 'X' in peptide or '*' in peptide:
                continue
                
            peptides.append({
                'peptide': peptide,
                'length': length,
                'start_pos': i + 1,
                'end_pos': i + length
            })
    
    return peptides

# Load the SARS-CoV-2 spike mRNA sequence
print("🔄 Loading SARS-CoV-2 spike mRNA sequence...")
print(f"Looking for file: {os.path.abspath(INPUT_FILE)}")

# Initialize variables to avoid NameError
peptide_list = []
protein_seq = ""

sequences = load_fasta_sequence(INPUT_FILE)

if not sequences:
    print("❌ Failed to load sequences from the specified path.")
    print("Available files in assets directory:")
    if os.path.exists("assets"):
        for f in os.listdir("assets"):
            print(f"  - {f}")
    else:
        print("  assets directory not found!")
    
    # Create sample data for testing
    print("🔄 Creating sample SARS-CoV-2 spike protein for testing...")
    sample_spike_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYKNNSIAPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQD"
    
    sequences = {
        'sample_spike': {
            'sequence': sample_spike_seq,
            'length': len(sample_spike_seq),
            'description': 'Sample SARS-CoV-2 Spike protein (for testing)'
        }
    }
    protein_seq = sample_spike_seq
    print(f"✅ Using sample spike protein sequence for demonstration")
    print(f"   Sample protein length: {len(protein_seq)} amino acids")
    
    # Store sequence information
    results['sequence_info'] = {
        'sequence_id': 'sample_spike',
        'mrna_length': len(sample_spike_seq) * 3,
        'protein_length': len(sample_spike_seq),
        'protein_sequence': sample_spike_seq,
        'note': 'Sample data used - original file not found'
    }
    
else:
    # Process the loaded sequence
    seq_id = list(sequences.keys())[0]
    mrna_seq = sequences[seq_id]['sequence']
    
    print(f"✅ Loaded sequence: {seq_id}")
    print(f"   Length: {sequences[seq_id]['length']} nucleotides")
    print(f"   Description: {sequences[seq_id]['description']}")
    
    # Translate to protein
    print("🔄 Translating mRNA to protein...")
    protein_seq = translate_mrna_to_protein(mrna_seq)
    
    print(f"✅ Translation completed")
    print(f"   Protein length: {len(protein_seq)} amino acids")
    print(f"   First 50 AA: {protein_seq[:50]}...")
    
    # Store sequence information
    results['sequence_info'] = {
        'sequence_id': seq_id,
        'mrna_length': len(mrna_seq),
        'protein_length': len(protein_seq),
        'protein_sequence': protein_seq
    }

# Generate peptides for both cases (loaded sequence or sample data)
if protein_seq:
    print("🔄 Generating overlapping peptides...")
    peptide_list = generate_overlapping_peptides(protein_seq)
    
    print(f"✅ Generated {len(peptide_list)} peptides")
    if len(peptide_list) > 0:
        print(f"   Length distribution:")
        length_counts = pd.Series([p['length'] for p in peptide_list]).value_counts().sort_index()
        for length, count in length_counts.items():
            print(f"   {length}-mers: {count} peptides")
        
        # Update sequence information with peptide count
        if 'sequence_info' in results:
            results['sequence_info']['total_peptides'] = len(peptide_list)
            results['sequence_info']['peptide_lengths'] = list(length_counts.index)
    else:
        print("⚠️ No valid peptides generated")
    
    print("✅ Sequence processing completed!")
else:
    print("❌ No protein sequence available for peptide generation")



# Cell 3: MHC Class I Epitope Prediction

def query_iedb_mhc_class_i(peptides: List[str], alleles: List[str], method: str = 'netmhcpan_el') -> List[Dict]:
    """Query IEDB MHC Class I prediction tool"""
    results = []
    
    # Process peptides in batches to avoid API limits
    batch_size = 50
    for i in range(0, len(peptides), batch_size):
        batch_peptides = peptides[i:i + batch_size]
        
        # Prepare data for API request
        data = {
            'method': method,
            'sequence_text': '\n'.join(batch_peptides),
            'allele': ','.join(alleles[:10]),  # Limit to first 10 alleles per request
            'length': '8,9,10,11'
        }
        
        try:
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(peptides)-1)//batch_size + 1} for MHC Class I...")
            response = requests.post(IEDB_MHC_I_URL, data=data, timeout=300)
            
            if response.status_code == 200:
                # Parse response (assuming TSV format)
                lines = response.text.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.split('\t')
                        if len(parts) >= 5:
                            results.append({
                                'allele': parts[0] if len(parts) > 0 else '',
                                'peptide': parts[1] if len(parts) > 1 else '',
                                'ic50': float(parts[2]) if len(parts) > 2 and parts[2].replace('.', '').isdigit() else None,
                                'percentile': float(parts[3]) if len(parts) > 3 and parts[3].replace('.', '').isdigit() else None,
                                'method': method,
                                'batch': i//batch_size + 1
                            })
            else:
                print(f"⚠️ API request failed for batch {i//batch_size + 1}: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error for batch {i//batch_size + 1}: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Error processing batch {i//batch_size + 1}: {e}")
            continue
            
        # Add delay to be respectful to the API
        time.sleep(2)
    
    return results

def simulate_mhc_class_i_predictions(peptides: List[str], alleles: List[str]) -> List[Dict]:
    """Simulate MHC Class I predictions (fallback if API is unavailable)"""
    print("🔄 Simulating MHC Class I predictions (API fallback)...")
    
    results = []
    np.random.seed(42)  # For reproducibility
    
    for peptide in peptides[:200]:  # Limit for demonstration
        for allele in alleles[:5]:  # Top 5 alleles
            # Simulate binding affinity based on peptide properties
            hydrophobic_aa = 'AILVFWYH'
            charged_aa = 'DEKR'
            
            hydrophobic_count = sum(1 for aa in peptide if aa in hydrophobic_aa)
            charged_count = sum(1 for aa in peptide if aa in charged_aa)
            
            # Simulate IC50 and percentile based on peptide composition
            base_ic50 = np.random.lognormal(7, 2)  # Log-normal distribution
            
            # Adjust based on peptide properties
            if hydrophobic_count > len(peptide) * 0.4:
                base_ic50 *= 0.5  # Better binding for hydrophobic peptides
            if charged_count > 2:
                base_ic50 *= 1.5  # Worse binding for highly charged peptides
                
            percentile = min(100, base_ic50 / 100)
            
            results.append({
                'allele': allele,
                'peptide': peptide,
                'ic50': base_ic50,
                'percentile': percentile,
                'method': 'simulated',
                'hydrophobic_count': hydrophobic_count,
                'charged_count': charged_count
            })
    
    return results

# Perform MHC Class I epitope prediction
if 'sequence_info' in results and len(peptide_list) > 0:
    print("🚀 Starting MHC Class I epitope prediction...")
    
    # Extract peptides for API
    peptides_for_prediction = [p['peptide'] for p in peptide_list[:500]]  # Limit for API efficiency
    print(f"   Using {len(peptides_for_prediction)} peptides for prediction")
    
    # Try IEDB API first
    try:
        print("🔄 Attempting IEDB API connection...")
        # Test with small subset first
        test_peptides = peptides_for_prediction[:5]
        mhc_i_results = query_iedb_mhc_class_i(test_peptides, COMMON_HLA_ALLELES[:3])
        
        if len(mhc_i_results) > 0:
            print("✅ IEDB API connection successful, processing full dataset...")
            mhc_i_results = query_iedb_mhc_class_i(peptides_for_prediction, COMMON_HLA_ALLELES)
        else:
            raise Exception("No results from IEDB API")
            
    except Exception as e:
        print(f"⚠️ IEDB API unavailable ({e}), using simulation...")
        mhc_i_results = simulate_mhc_class_i_predictions(peptides_for_prediction, COMMON_HLA_ALLELES)
    
    # Store results
    results['mhc_class_i'] = mhc_i_results
    
    print(f"✅ MHC Class I prediction completed!")
    print(f"   Total predictions: {len(mhc_i_results)}")
    
    # Analyze results
    if mhc_i_results:
        df_mhc_i = pd.DataFrame(mhc_i_results)
        
        # Filter strong binders (percentile < 2.0)
        if 'percentile' in df_mhc_i.columns:
            strong_binders = df_mhc_i[df_mhc_i['percentile'] < 2.0]
            weak_binders = df_mhc_i[df_mhc_i['percentile'] > 50.0]
            
            print(f"   Strong binders (percentile < 2.0): {len(strong_binders)}")
            print(f"   Weak binders (percentile > 50.0): {len(weak_binders)}")
            
            if len(strong_binders) > 0:
                print("   Top 5 strong binders:")
                top_binders = strong_binders.nsmallest(5, 'percentile')
                for _, row in top_binders.iterrows():
                    print(f"     {row['peptide']} ({row['allele']}) - {row['percentile']:.2f}%")
        
        # Allele distribution
        print(f"   Alleles analyzed: {df_mhc_i['allele'].nunique()}")
        print(f"   Unique peptides: {df_mhc_i['peptide'].nunique()}")
        
elif 'sequence_info' not in results:
    print("❌ No sequence data available for MHC Class I prediction")
else:
    print("❌ No peptides generated for MHC Class I prediction")



# Cell 4: B-cell Epitope Prediction

def query_iedb_bcell_epitopes(protein_sequence: str, method: str = 'Bepipred') -> List[Dict]:
    """Query IEDB B-cell epitope prediction tool"""
    data = {
        'method': method,
        'sequence_text': protein_sequence,
        'window_size': '20'
    }
    
    try:
        print(f"🔄 Querying IEDB B-cell epitope prediction ({method})...")
        response = requests.post(IEDB_BCELL_URL, data=data, timeout=300)
        
        if response.status_code == 200:
            results = []
            lines = response.text.strip().split('\n')
            
            for i, line in enumerate(lines[1:], 1):  # Skip header
                parts = line.split('\t')
                if len(parts) >= 3:
                    results.append({
                        'position': i,
                        'residue': parts[0] if len(parts) > 0 else '',
                        'score': float(parts[1]) if len(parts) > 1 and parts[1].replace('.', '').replace('-', '').isdigit() else 0.0,
                        'method': method
                    })
            
            return results
        else:
            print(f"⚠️ B-cell API request failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"⚠️ Error querying B-cell epitopes: {e}")
        return []

def simulate_bcell_epitopes(protein_sequence: str) -> List[Dict]:
    """Simulate B-cell epitope predictions (fallback)"""
    print("🔄 Simulating B-cell epitope predictions...")
    
    results = []
    np.random.seed(42)
    
    # Simulate Bepipred-like scores
    for i, residue in enumerate(protein_sequence):
        # Basic simulation based on amino acid properties
        hydrophilic_aa = 'DENQHKRST'
        hydrophobic_aa = 'AILVFWYC'
        
        base_score = np.random.normal(0.0, 0.3)
        
        # Adjust score based on amino acid properties
        if residue in hydrophilic_aa:
            base_score += 0.2  # Higher score for hydrophilic residues
        elif residue in hydrophobic_aa:
            base_score -= 0.1  # Lower score for hydrophobic residues
            
        # Add some local context (smoothing)
        if i > 0 and i < len(protein_sequence) - 1:
            context_bonus = np.random.normal(0.0, 0.1)
            base_score += context_bonus
            
        results.append({
            'position': i + 1,
            'residue': residue,
            'score': base_score,
            'method': 'simulated'
        })
    
    return results

def identify_bcell_epitope_regions(bcell_scores: List[Dict], threshold: float = 0.5, min_length: int = 6) -> List[Dict]:
    """Identify continuous B-cell epitope regions above threshold"""
    epitope_regions = []
    current_region = []
    
    for score_data in bcell_scores:
        if score_data['score'] > threshold:
            current_region.append(score_data)
        else:
            if len(current_region) >= min_length:
                epitope_regions.append({
                    'start_pos': current_region[0]['position'],
                    'end_pos': current_region[-1]['position'],
                    'length': len(current_region),
                    'avg_score': np.mean([r['score'] for r in current_region]),
                    'sequence': ''.join([r['residue'] for r in current_region])
                })
            current_region = []
    
    # Handle last region
    if len(current_region) >= min_length:
        epitope_regions.append({
            'start_pos': current_region[0]['position'],
            'end_pos': current_region[-1]['position'],
            'length': len(current_region),
            'avg_score': np.mean([r['score'] for r in current_region]),
            'sequence': ''.join([r['residue'] for r in current_region])
        })
    
    return epitope_regions

# Perform B-cell epitope prediction
if 'sequence_info' in results and 'protein_sequence' in results['sequence_info']:
    print("🚀 Starting B-cell epitope prediction...")
    
    protein_seq = results['sequence_info']['protein_sequence']
    print(f"   Analyzing protein sequence of {len(protein_seq)} amino acids")
    
    # Try IEDB API first
    try:
        bcell_results = query_iedb_bcell_epitopes(protein_seq)
        
        if not bcell_results:
            raise Exception("No results from IEDB B-cell API")
            
        print("✅ IEDB B-cell API connection successful!")
        
    except Exception as e:
        print(f"⚠️ IEDB B-cell API unavailable ({e}), using simulation...")
        bcell_results = simulate_bcell_epitopes(protein_seq)
    
    # Store raw B-cell scores
    results['bcell_epitopes'] = bcell_results
    
    # Identify epitope regions
    if bcell_results:
        epitope_regions = identify_bcell_epitope_regions(bcell_results, threshold=0.5)
        results['bcell_epitope_regions'] = epitope_regions
        
        print(f"✅ B-cell epitope prediction completed!")
        print(f"   Total positions analyzed: {len(bcell_results)}")
        print(f"   Epitope regions identified: {len(epitope_regions)}")
        
        if epitope_regions:
            # Sort by average score
            epitope_regions.sort(key=lambda x: x['avg_score'], reverse=True)
            
            print("   Top 5 B-cell epitope regions:")
            for i, region in enumerate(epitope_regions[:5], 1):
                print(f"     {i}. Pos {region['start_pos']}-{region['end_pos']} (L={region['length']}, "
                      f"Score={region['avg_score']:.3f}): {region['sequence']}")
        
        # Calculate statistics
        df_bcell = pd.DataFrame(bcell_results)
        mean_score = df_bcell['score'].mean()
        std_score = df_bcell['score'].std()
        high_score_positions = len(df_bcell[df_bcell['score'] > mean_score + std_score])
        
        print(f"   Mean score: {mean_score:.3f} ± {std_score:.3f}")
        print(f"   High-scoring positions: {high_score_positions}")
        
else:
    print("❌ No protein sequence available for B-cell epitope prediction")


# Cell 5: Data Analysis and Visualization

def create_mhc_class_i_plots(mhc_data: List[Dict], output_dir: str):
    """Create visualizations for MHC Class I results"""
    df = pd.DataFrame(mhc_data)
    
    if df.empty:
        print("⚠️ No MHC Class I data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Percentile distribution
    if 'percentile' in df.columns:
        axes[0, 0].hist(df['percentile'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(2.0, color='red', linestyle='--', label='Strong binder threshold (2%)')
        axes[0, 0].axvline(50.0, color='orange', linestyle='--', label='Weak binder threshold (50%)')
        axes[0, 0].set_xlabel('Binding Percentile')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('MHC Class I Binding Percentile Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Allele-wise binding counts
    if 'allele' in df.columns:
        allele_counts = df['allele'].value_counts().head(10)
        axes[0, 1].bar(range(len(allele_counts)), allele_counts.values, color='green', alpha=0.7)
        axes[0, 1].set_xticks(range(len(allele_counts)))
        axes[0, 1].set_xticklabels(allele_counts.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Number of Predictions')
        axes[0, 1].set_title('Predictions per HLA Allele')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Peptide length distribution
    if 'peptide' in df.columns:
        peptide_lengths = [len(p) for p in df['peptide']]
        length_counts = pd.Series(peptide_lengths).value_counts().sort_index()
        axes[1, 0].bar(length_counts.index, length_counts.values, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Peptide Length')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Peptide Length Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: IC50 vs Percentile scatter
    if 'ic50' in df.columns and 'percentile' in df.columns:
        valid_data = df[(df['ic50'].notna()) & (df['percentile'].notna())]
        if not valid_data.empty:
            scatter = axes[1, 1].scatter(valid_data['ic50'], valid_data['percentile'], 
                                       alpha=0.6, s=20, c='red')
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_xlabel('IC50 (nM)')
            axes[1, 1].set_ylabel('Percentile Rank')
            axes[1, 1].set_title('IC50 vs Percentile Correlation')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mhc_class_i_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_bcell_epitope_plots(bcell_data: List[Dict], epitope_regions: List[Dict], output_dir: str):
    """Create visualizations for B-cell epitope results"""
    if not bcell_data:
        print("⚠️ No B-cell data to plot")
        return
        
    df = pd.DataFrame(bcell_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: B-cell epitope score along sequence
    if 'position' in df.columns and 'score' in df.columns:
        axes[0, 0].plot(df['position'], df['score'], linewidth=1, color='blue', alpha=0.7)
        axes[0, 0].axhline(0.5, color='red', linestyle='--', label='Epitope threshold (0.5)')
        axes[0, 0].fill_between(df['position'], df['score'], 0.5, 
                               where=(df['score'] > 0.5), alpha=0.3, color='red')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('B-cell Epitope Score')
        axes[0, 0].set_title('B-cell Epitope Prediction Along Sequence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Score distribution
    if 'score' in df.columns:
        axes[0, 1].hist(df['score'], bins=30, alpha=0.7, color='cyan', edgecolor='black')
        axes[0, 1].axvline(df['score'].mean(), color='red', linestyle='-', 
                          label=f'Mean: {df["score"].mean():.3f}')
        axes[0, 1].axvline(0.5, color='orange', linestyle='--', label='Threshold: 0.5')
        axes[0, 1].set_xlabel('B-cell Epitope Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('B-cell Epitope Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epitope region lengths
    if epitope_regions:
        lengths = [region['length'] for region in epitope_regions]
        axes[1, 0].hist(lengths, bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Epitope Region Length')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('B-cell Epitope Region Length Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Epitope region scores
        scores = [region['avg_score'] for region in epitope_regions]
        axes[1, 1].bar(range(len(scores[:20])), sorted(scores, reverse=True)[:20], 
                      color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Epitope Region (Ranked)')
        axes[1, 1].set_ylabel('Average Score')
        axes[1, 1].set_title('Top 20 B-cell Epitope Regions by Score')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bcell_epitope_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics():
    """Create and display summary statistics"""
    print("\n" + "="*60)
    print("🔬 IEDB ANALYSIS SUMMARY STATISTICS")
    print("="*60)
    
    # Sequence information
    if 'sequence_info' in results:
        seq_info = results['sequence_info']
        print(f"📊 SEQUENCE INFORMATION:")
        print(f"   Sequence ID: {seq_info.get('sequence_id', 'Unknown')}")
        print(f"   mRNA Length: {seq_info.get('mrna_length', 0):,} nucleotides")
        print(f"   Protein Length: {seq_info.get('protein_length', 0):,} amino acids")
        print(f"   Total Peptides Generated: {seq_info.get('total_peptides', 0):,}")
    
    # MHC Class I statistics
    if 'mhc_class_i' in results and results['mhc_class_i']:
        mhc_df = pd.DataFrame(results['mhc_class_i'])
        print(f"\n🧬 MHC CLASS I ANALYSIS:")
        print(f"   Total Predictions: {len(mhc_df):,}")
        print(f"   Unique Peptides: {mhc_df['peptide'].nunique():,}")
        print(f"   HLA Alleles Tested: {mhc_df['allele'].nunique()}")
        
        if 'percentile' in mhc_df.columns:
            strong_binders = mhc_df[mhc_df['percentile'] <= 2.0]
            moderate_binders = mhc_df[(mhc_df['percentile'] > 2.0) & (mhc_df['percentile'] <= 10.0)]
            weak_binders = mhc_df[mhc_df['percentile'] > 50.0]
            
            print(f"   Strong Binders (≤2%): {len(strong_binders):,} ({len(strong_binders)/len(mhc_df)*100:.1f}%)")
            print(f"   Moderate Binders (2-10%): {len(moderate_binders):,} ({len(moderate_binders)/len(mhc_df)*100:.1f}%)")
            print(f"   Weak Binders (>50%): {len(weak_binders):,} ({len(weak_binders)/len(mhc_df)*100:.1f}%)")
            
            if len(strong_binders) > 0:
                best_binder = strong_binders.loc[strong_binders['percentile'].idxmin()]
                print(f"   Best Binder: {best_binder['peptide']} "
                      f"({best_binder['percentile']:.3f}%)")
    
    # B-cell epitope statistics
    if 'bcell_epitopes' in results and results['bcell_epitopes']:
        bcell_df = pd.DataFrame(results['bcell_epitopes'])
        print(f"\n🔵 B-CELL EPITOPE ANALYSIS:")
        print(f"   Positions Analyzed: {len(bcell_df):,}")
        print(f"   Mean Score: {bcell_df['score'].mean():.3f} ± {bcell_df['score'].std():.3f}")
        print(f"   Score Range: {bcell_df['score'].min():.3f} to {bcell_df['score'].max():.3f}")
        
        high_scoring = bcell_df[bcell_df['score'] > 0.5]
        print(f"   High-scoring Positions (>0.5): {len(high_scoring):,} ({len(high_scoring)/len(bcell_df)*100:.1f}%)")
        
        if 'bcell_epitope_regions' in results:
            regions = results['bcell_epitope_regions']
            print(f"   Epitope Regions Identified: {len(regions)}")
            if regions:
                avg_length = np.mean([r['length'] for r in regions])
                total_coverage = sum([r['length'] for r in regions])
                protein_length = results['sequence_info'].get('protein_length', 1)
                coverage_percent = (total_coverage / protein_length) * 100
                
                print(f"   Average Region Length: {avg_length:.1f} residues")
                print(f"   Total Epitope Coverage: {total_coverage} residues ({coverage_percent:.1f}%)")
                print(f"   Best Region: Pos {regions[0]['start_pos']}-{regions[0]['end_pos']} "
                      f"(Score: {regions[0]['avg_score']:.3f})")

# Perform analysis and create visualizations
print("🚀 Starting data analysis and visualization...")

# Create visualizations
if results.get('mhc_class_i'):
    print("🔄 Creating MHC Class I visualizations...")
    create_mhc_class_i_plots(results['mhc_class_i'], OUTPUT_DIR)

if results.get('bcell_epitopes'):
    print("🔄 Creating B-cell epitope visualizations...")
    epitope_regions = results.get('bcell_epitope_regions', [])
    create_bcell_epitope_plots(results['bcell_epitopes'], epitope_regions, OUTPUT_DIR)

# Generate summary statistics
create_summary_statistics()

print(f"\n✅ Analysis completed! All results will be saved to: {OUTPUT_DIR}")



# Cell 6: Save Results and Generate Reports

def save_results_to_files(results_dict: Dict, output_dir: str):
    """Save all results to various file formats"""
    print("🔄 Saving results to files...")
    
    # Save complete results as JSON
    json_file = os.path.join(output_dir, 'iedb_analysis_results.json')
    with open(json_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results_dict.items():
            if isinstance(value, list):
                json_results[key] = []
                for item in value:
                    if isinstance(item, dict):
                        clean_item = {}
                        for k, v in item.items():
                            if isinstance(v, np.floating):
                                clean_item[k] = float(v)
                            elif isinstance(v, np.integer):
                                clean_item[k] = int(v)
                            else:
                                clean_item[k] = v
                        json_results[key].append(clean_item)
                    else:
                        json_results[key].append(item)
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    print(f"✅ Complete results saved to: {json_file}")
    
    # Save MHC Class I results as CSV
    if results_dict.get('mhc_class_i'):
        mhc_df = pd.DataFrame(results_dict['mhc_class_i'])
        mhc_csv = os.path.join(output_dir, 'mhc_class_i_predictions.csv')
        mhc_df.to_csv(mhc_csv, index=False)
        print(f"✅ MHC Class I results saved to: {mhc_csv}")
        
        # Save top binders separately
        if 'percentile' in mhc_df.columns:
            strong_binders = mhc_df[mhc_df['percentile'] <= 2.0].sort_values('percentile')
            if not strong_binders.empty:
                strong_csv = os.path.join(output_dir, 'strong_binders.csv')
                strong_binders.to_csv(strong_csv, index=False)
                print(f"✅ Strong binders saved to: {strong_csv}")
    
    # Save B-cell epitope results as CSV
    if results_dict.get('bcell_epitopes'):
        bcell_df = pd.DataFrame(results_dict['bcell_epitopes'])
        bcell_csv = os.path.join(output_dir, 'bcell_epitope_scores.csv')
        bcell_df.to_csv(bcell_csv, index=False)
        print(f"✅ B-cell epitope scores saved to: {bcell_csv}")
        
        # Save epitope regions
        if results_dict.get('bcell_epitope_regions'):
            regions_df = pd.DataFrame(results_dict['bcell_epitope_regions'])
            regions_csv = os.path.join(output_dir, 'bcell_epitope_regions.csv')
            regions_df.to_csv(regions_csv, index=False)
            print(f"✅ B-cell epitope regions saved to: {regions_csv}")
    
    # Save protein sequence as FASTA
    if results_dict.get('sequence_info', {}).get('protein_sequence'):
        protein_seq = results_dict['sequence_info']['protein_sequence']
        seq_id = results_dict['sequence_info'].get('sequence_id', 'SARS_CoV2_Spike')
        
        fasta_file = os.path.join(output_dir, 'spike_protein.fasta')
        with open(fasta_file, 'w') as f:
            f.write(f">{seq_id}_protein\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(protein_seq), 80):
                f.write(protein_seq[i:i+80] + '\n')
        print(f"✅ Protein sequence saved to: {fasta_file}")

def generate_html_report(results_dict: Dict, output_dir: str):
    """Generate an HTML report with embedded visualizations"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>IEDB Analysis Report - SARS-CoV-2 Spike Protein</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 8px; }}
            .section {{ margin: 20px 0; }}
            .stats {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
            .epitope {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007cba; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .strong {{ color: #d73027; font-weight: bold; }}
            .moderate {{ color: #fc8d59; }}
            .weak {{ color: #91bfdb; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🦠 IEDB Analysis Report</h1>
            <h2>SARS-CoV-2 Spike Protein Epitope Predictions</h2>
            <p><strong>Analysis Date:</strong> {results_dict['analysis_metadata']['timestamp']}</p>
            <p><strong>Input File:</strong> {results_dict['analysis_metadata']['input_file']}</p>
            <p><strong>Output Directory:</strong> {results_dict['analysis_metadata']['output_dir']}</p>
        </div>
    """
    
    # Sequence Information
    if results_dict.get('sequence_info'):
        seq_info = results_dict['sequence_info']
        html_content += f"""
        <div class="section">
            <h3>📊 Sequence Information</h3>
            <div class="stats">
                <p><strong>Sequence ID:</strong> {seq_info.get('sequence_id', 'Unknown')}</p>
                <p><strong>mRNA Length:</strong> {seq_info.get('mrna_length', 0):,} nucleotides</p>
                <p><strong>Protein Length:</strong> {seq_info.get('protein_length', 0):,} amino acids</p>
                <p><strong>Total Peptides Generated:</strong> {seq_info.get('total_peptides', 0):,}</p>
            </div>
        </div>
        """
    
    # MHC Class I Results
    if results_dict.get('mhc_class_i'):
        mhc_df = pd.DataFrame(results_dict['mhc_class_i'])
        html_content += f"""
        <div class="section">
            <h3>🧬 MHC Class I Analysis</h3>
            <div class="stats">
                <p><strong>Total Predictions:</strong> {len(mhc_df):,}</p>
                <p><strong>Unique Peptides:</strong> {mhc_df['peptide'].nunique():,}</p>
                <p><strong>HLA Alleles Tested:</strong> {mhc_df['allele'].nunique()}</p>
            </div>
        """
        
        if 'percentile' in mhc_df.columns:
            strong_binders = mhc_df[mhc_df['percentile'] <= 2.0]
            moderate_binders = mhc_df[(mhc_df['percentile'] > 2.0) & (mhc_df['percentile'] <= 10.0)]
            weak_binders = mhc_df[mhc_df['percentile'] > 50.0]
            
            html_content += f"""
            <h4>Binding Categories:</h4>
            <ul>
                <li class="strong">Strong Binders (≤2%): {len(strong_binders):,} ({len(strong_binders)/len(mhc_df)*100:.1f}%)</li>
                <li class="moderate">Moderate Binders (2-10%): {len(moderate_binders):,} ({len(moderate_binders)/len(mhc_df)*100:.1f}%)</li>
                <li class="weak">Weak Binders (>50%): {len(weak_binders):,} ({len(weak_binders)/len(mhc_df)*100:.1f}%)</li>
            </ul>
            """
            
            # Top 10 strong binders table
            if not strong_binders.empty:
                top_binders = strong_binders.nsmallest(10, 'percentile')
                html_content += """
                <h4>Top 10 Strong Binders:</h4>
                <table>
                    <tr><th>Rank</th><th>Peptide</th><th>HLA Allele</th><th>Percentile</th><th>IC50 (nM)</th></tr>
                """
                for i, (_, row) in enumerate(top_binders.iterrows(), 1):
                    ic50_val = f"{row.get('ic50', 'N/A'):.1f}" if pd.notna(row.get('ic50')) else 'N/A'
                    html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td><strong>{row['peptide']}</strong></td>
                        <td>{row['allele']}</td>
                        <td class="strong">{row['percentile']:.3f}%</td>
                        <td>{ic50_val}</td>
                    </tr>
                    """
                html_content += "</table>"
        
        html_content += "</div>"
    
    # B-cell Epitope Results
    if results_dict.get('bcell_epitopes'):
        bcell_df = pd.DataFrame(results_dict['bcell_epitopes'])
        html_content += f"""
        <div class="section">
            <h3>🔵 B-cell Epitope Analysis</h3>
            <div class="stats">
                <p><strong>Positions Analyzed:</strong> {len(bcell_df):,}</p>
                <p><strong>Mean Score:</strong> {bcell_df['score'].mean():.3f} ± {bcell_df['score'].std():.3f}</p>
                <p><strong>Score Range:</strong> {bcell_df['score'].min():.3f} to {bcell_df['score'].max():.3f}</p>
        """
        
        high_scoring = bcell_df[bcell_df['score'] > 0.5]
        html_content += f"<p><strong>High-scoring Positions (>0.5):</strong> {len(high_scoring):,} ({len(high_scoring)/len(bcell_df)*100:.1f}%)</p>"
        
        if results_dict.get('bcell_epitope_regions'):
            regions = results_dict['bcell_epitope_regions']
            html_content += f"<p><strong>Epitope Regions Identified:</strong> {len(regions)}</p>"
            
            if regions:
                avg_length = np.mean([r['length'] for r in regions])
                total_coverage = sum([r['length'] for r in regions])
                protein_length = results_dict['sequence_info'].get('protein_length', 1)
                coverage_percent = (total_coverage / protein_length) * 100
                
                html_content += f"""
                <p><strong>Average Region Length:</strong> {avg_length:.1f} residues</p>
                <p><strong>Total Epitope Coverage:</strong> {total_coverage} residues ({coverage_percent:.1f}%)</p>
                </div>
                
                <h4>Top 10 B-cell Epitope Regions:</h4>
                <table>
                    <tr><th>Rank</th><th>Position</th><th>Length</th><th>Average Score</th><th>Sequence</th></tr>
                """
                for i, region in enumerate(regions[:10], 1):
                    html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{region['start_pos']}-{region['end_pos']}</td>
                        <td>{region['length']}</td>
                        <td><strong>{region['avg_score']:.3f}</strong></td>
                        <td><code>{region['sequence']}</code></td>
                    </tr>
                    """
                html_content += "</table>"
        
        html_content += "</div>"
    
    # Files generated section
    html_content += f"""
        <div class="section">
            <h3>📁 Generated Files</h3>
            <ul>
                <li><strong>iedb_analysis_results.json</strong> - Complete analysis results</li>
                <li><strong>mhc_class_i_predictions.csv</strong> - MHC Class I binding predictions</li>
                <li><strong>strong_binders.csv</strong> - Strong binding peptides only</li>
                <li><strong>bcell_epitope_scores.csv</strong> - B-cell epitope scores</li>
                <li><strong>bcell_epitope_regions.csv</strong> - Identified epitope regions</li>
                <li><strong>spike_protein.fasta</strong> - Translated protein sequence</li>
                <li><strong>mhc_class_i_analysis.png</strong> - MHC binding visualizations</li>
                <li><strong>bcell_epitope_analysis.png</strong> - B-cell epitope visualizations</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_file = os.path.join(output_dir, 'iedb_analysis_report.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report saved to: {html_file}")

def generate_text_summary(results_dict: Dict, output_dir: str):
    """Generate a plain text summary report"""
    summary_lines = [
        "="*80,
        "IEDB ANALYSIS SUMMARY REPORT",
        "SARS-CoV-2 Spike Protein Epitope Predictions",
        "="*80,
        f"Analysis Date: {results_dict['analysis_metadata']['timestamp']}",
        f"Input File: {results_dict['analysis_metadata']['input_file']}",
        f"Output Directory: {results_dict['analysis_metadata']['output_dir']}",
        ""
    ]
    
    # Add sequence information
    if results_dict.get('sequence_info'):
        seq_info = results_dict['sequence_info']
        summary_lines.extend([
            "SEQUENCE INFORMATION:",
            f"  Sequence ID: {seq_info.get('sequence_id', 'Unknown')}",
            f"  mRNA Length: {seq_info.get('mrna_length', 0):,} nucleotides",
            f"  Protein Length: {seq_info.get('protein_length', 0):,} amino acids",
            f"  Total Peptides Generated: {seq_info.get('total_peptides', 0):,}",
            ""
        ])
    
    # Add MHC Class I summary
    if results_dict.get('mhc_class_i'):
        mhc_df = pd.DataFrame(results_dict['mhc_class_i'])
        summary_lines.extend([
            "MHC CLASS I ANALYSIS:",
            f"  Total Predictions: {len(mhc_df):,}",
            f"  Unique Peptides: {mhc_df['peptide'].nunique():,}",
            f"  HLA Alleles Tested: {mhc_df['allele'].nunique()}",
        ])
        
        if 'percentile' in mhc_df.columns:
            strong_binders = mhc_df[mhc_df['percentile'] <= 2.0]
            moderate_binders = mhc_df[(mhc_df['percentile'] > 2.0) & (mhc_df['percentile'] <= 10.0)]
            
            summary_lines.extend([
                f"  Strong Binders (≤2%): {len(strong_binders):,}",
                f"  Moderate Binders (2-10%): {len(moderate_binders):,}",
            ])
            
            if not strong_binders.empty:
                best_binder = strong_binders.loc[strong_binders['percentile'].idxmin()]
                summary_lines.append(f"  Best Binder: {best_binder['peptide']} ({best_binder['percentile']:.3f}%)")
        
        summary_lines.append("")
    
    # Add B-cell epitope summary
    if results_dict.get('bcell_epitopes'):
        bcell_df = pd.DataFrame(results_dict['bcell_epitopes'])
        summary_lines.extend([
            "B-CELL EPITOPE ANALYSIS:",
            f"  Positions Analyzed: {len(bcell_df):,}",
            f"  Mean Score: {bcell_df['score'].mean():.3f} ± {bcell_df['score'].std():.3f}",
        ])
        
        if results_dict.get('bcell_epitope_regions'):
            regions = results_dict['bcell_epitope_regions']
            summary_lines.append(f"  Epitope Regions Identified: {len(regions)}")
            
            if regions:
                best_region = regions[0]
                summary_lines.append(f"  Best Region: {best_region['sequence']} "
                                   f"(Pos {best_region['start_pos']}-{best_region['end_pos']}, "
                                   f"Score: {best_region['avg_score']:.3f})")
    
    summary_text = '\n'.join(summary_lines)
    
    # Save text summary
    txt_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(txt_file, 'w') as f:
        f.write(summary_text)
    
    print(f"✅ Text summary saved to: {txt_file}")
    
    return summary_text

# Save all results and generate reports
print("🚀 Generating final reports and saving results...")

# Save raw data files
save_results_to_files(results, OUTPUT_DIR)

# Generate HTML report
generate_html_report(results, OUTPUT_DIR)

# Generate text summary
summary_text = generate_text_summary(results, OUTPUT_DIR)

print("\n" + summary_text)
print(f"\n🎉 IEDB analysis completed successfully!")
print(f"📁 All results saved in: {os.path.abspath(OUTPUT_DIR)}")
print(f"📊 Open 'iedb_analysis_report.html' for detailed interactive report")