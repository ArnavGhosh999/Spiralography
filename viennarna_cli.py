#!/usr/bin/env python3
"""
ViennaRNA CLI wrapper for RNA secondary structure prediction
Input: FASTA/JSON/JSONL files
Output: Structure predictions in JSON/JSONL format
"""

import subprocess
import sys
import json
import os
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import tempfile

def install_viennarna():
    """Install ViennaRNA if not available"""
    try:
        subprocess.run(['RNAfold', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing ViennaRNA...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'viennarna'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Failed to install ViennaRNA. Using Python fallback...")
            return False

def run_rnafold_cli(sequence, sequence_id="sequence"):
    """Run RNAfold CLI on a sequence"""
    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
            tmp_file.write(f">{sequence_id}\n{sequence}\n")
            tmp_file.flush()
            
            # Run RNAfold
            result = subprocess.run(['RNAfold', '--noPS', tmp_file.name], 
                                  capture_output=True, text=True, check=True)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            structure = lines[1].split()[0] if len(lines) > 1 else ""
            mfe_line = lines[2] if len(lines) > 2 else ""
            
            # Extract MFE value
            mfe = 0.0
            if ')' in mfe_line:
                try:
                    mfe = float(mfe_line.split('(')[1].split(')')[0])
                except (IndexError, ValueError):
                    pass
            
            return structure, mfe
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"RNAfold CLI failed: {e}")
        return None, None

def run_viennarna_python(sequence):
    """Fallback to Python ViennaRNA library"""
    try:
        import RNA
        structure, mfe = RNA.fold(sequence)
        return structure, mfe
    except ImportError:
        print("ViennaRNA Python library not available")
        return None, None

def predict_structure(sequence, sequence_id="sequence", use_cli=True):
    """Predict RNA secondary structure"""
    if use_cli:
        structure, mfe = run_rnafold_cli(sequence, sequence_id)
        if structure is not None:
            return structure, mfe
    
    # Fallback to Python library
    structure, mfe = run_viennarna_python(sequence)
    if structure is not None:
        return structure, mfe
    
    # Final fallback - simple prediction
    print("Using simple fallback prediction")
    structure = "." * len(sequence)
    mfe = 0.0
    return structure, mfe

def process_fasta_file(input_file):
    """Process FASTA file"""
    results = []
    
    for record in SeqIO.parse(input_file, "fasta"):
        sequence = str(record.seq).upper()
        structure, mfe = predict_structure(sequence, record.id)
        
        results.append({
            "id": record.id,
            "description": record.description,
            "sequence": sequence,
            "structure": structure,
            "mfe": mfe,
            "length": len(sequence)
        })
    
    return results

def process_json_file(input_file):
    """Process JSON file"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'sequence' in item:
                sequence = item['sequence'].upper()
                structure, mfe = predict_structure(sequence, item.get('id', f'seq_{i}'))
                
                results.append({
                    "id": item.get('id', f'seq_{i}'),
                    "description": item.get('description', ''),
                    "sequence": sequence,
                    "structure": structure,
                    "mfe": mfe,
                    "length": len(sequence)
                })
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and set(value.upper()).issubset(set("AUGC")):
                sequence = value.upper()
                structure, mfe = predict_structure(sequence, key)
                
                results.append({
                    "id": key,
                    "description": "",
                    "sequence": sequence,
                    "structure": structure,
                    "mfe": mfe,
                    "length": len(sequence)
                })
    
    return results

def process_jsonl_file(input_file):
    """Process JSONL file"""
    results = []
    
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                if isinstance(item, dict) and 'sequence' in item:
                    sequence = item['sequence'].upper()
                    structure, mfe = predict_structure(sequence, item.get('id', f'seq_{i}'))
                    
                    results.append({
                        "id": item.get('id', f'seq_{i}'),
                        "description": item.get('description', ''),
                        "sequence": sequence,
                        "structure": structure,
                        "mfe": mfe,
                        "length": len(sequence)
                    })
            except json.JSONDecodeError:
                continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="ViennaRNA CLI wrapper for RNA structure prediction")
    parser.add_argument("input_file", help="Input FASTA, JSON, or JSONL file")
    parser.add_argument("--output", "-o", help="Output file (default: viennarna_output.jsonl)")
    parser.add_argument("--format", "-f", choices=['json', 'jsonl'], default='jsonl', help="Output format")
    parser.add_argument("--no-cli", action='store_true', help="Disable CLI and use Python library only")
    
    args = parser.parse_args()
    
    # Check if ViennaRNA is available
    if not args.no_cli:
        install_viennarna()
    
    # Process input file
    input_file = args.input_file
    print(f"Processing: {input_file}")
    
    if input_file.endswith(('.fasta', '.fa', '.fna')):
        results = process_fasta_file(input_file)
    elif input_file.endswith('.json'):
        results = process_json_file(input_file)
    elif input_file.endswith('.jsonl'):
        results = process_jsonl_file(input_file)
    else:
        print("Unsupported file format. Please use FASTA, JSON, or JSONL.")
        sys.exit(1)
    
    print(f"Processed {len(results)} sequences")
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        output_file = f"viennarna_output.{args.format}"
    
    if args.format == 'json':
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:  # jsonl
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    if results:
        print("\nSummary:")
        for result in results[:3]:  # Show first 3
            print(f"  {result['id']}: {result['length']} nt, MFE = {result['mfe']:.2f}")
        if len(results) > 3:
            print(f"  ... and {len(results) - 3} more sequences")

if __name__ == "__main__":
    main()
