#!/usr/bin/env python3
"""
DNAChisel CLI wrapper for DNA sequence optimization
Input: FASTA/JSON/JSONL files
Output: Optimized sequences in JSON/JSONL format
"""

import subprocess
import sys
import json
import os
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import tempfile

def run_dnachisel_cli(sequence, sequence_id="sequence", output_file=None):
    """Run DNAChisel CLI on a sequence"""
    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
            tmp_file.write(f">{sequence_id}\n{sequence}\n")
            tmp_file.flush()
            
            if output_file is None:
                output_file = tempfile.mktemp(suffix='.fasta')
            
            # Run DNAChisel with basic optimization
            cmd = [
                'dnachisel',
                'optimize',
                tmp_file.name,
                '--constraints', 'AvoidPattern("GGTCTC")', 'AvoidPattern("GAGACC")',
                '--objectives', 'MaximizeCAI(species="e_coli")',
                '--output', output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Read optimized sequence
            optimized_seq = None
            if os.path.exists(output_file):
                for record in SeqIO.parse(output_file, "fasta"):
                    optimized_seq = str(record.seq)
                    break
                os.unlink(output_file)
            
            # Clean up input file
            os.unlink(tmp_file.name)
            
            return optimized_seq, result.stdout
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"DNAChisel CLI failed: {e}")
        return None, str(e)

def optimize_sequence_python(sequence, sequence_id="sequence"):
    """Fallback to Python DNAChisel library"""
    try:
        from dnachisel import DnaOptimizationProblem, AvoidPattern, EnforceTranslation, MaximizeCAI
        
        # Create optimization problem
        constraints = [
            AvoidPattern("GGTCTC"),
            AvoidPattern("GAGACC"),
        ]
        
        objectives = []
        if len(sequence) % 3 == 0 and sequence.upper().startswith('ATG'):
            constraints.append(EnforceTranslation())
            objectives.append(MaximizeCAI(species="e_coli"))
        
        problem = DnaOptimizationProblem(
            sequence=sequence,
            constraints=constraints,
            objectives=objectives
        )
        
        # Resolve constraints
        problem.resolve_constraints()
        
        # Optimize if objectives exist
        if objectives:
            problem.optimize()
        
        return problem.sequence, "Optimization completed"
        
    except ImportError:
        print("DNAChisel Python library not available")
        return None, "Library not available"
    except Exception as e:
        print(f"DNAChisel optimization failed: {e}")
        return None, str(e)

def optimize_sequence(sequence, sequence_id="sequence", use_cli=True):
    """Optimize DNA sequence"""
    if use_cli:
        optimized, log = run_dnachisel_cli(sequence, sequence_id)
        if optimized is not None:
            return optimized, log
    
    # Fallback to Python library
    optimized, log = optimize_sequence_python(sequence, sequence_id)
    if optimized is not None:
        return optimized, log
    
    # Final fallback - return original sequence
    print("Using original sequence as fallback")
    return sequence, "No optimization performed"

def calculate_gc_content(sequence):
    """Calculate GC content percentage"""
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return (gc_count / len(sequence)) * 100 if len(sequence) > 0 else 0

def process_fasta_file(input_file, use_cli=True):
    """Process FASTA file"""
    results = []
    
    for record in SeqIO.parse(input_file, "fasta"):
        sequence = str(record.seq).upper()
        optimized, log = optimize_sequence(sequence, record.id, use_cli)
        
        results.append({
            "id": record.id,
            "description": record.description,
            "original_sequence": sequence,
            "optimized_sequence": optimized,
            "original_gc_content": calculate_gc_content(sequence),
            "optimized_gc_content": calculate_gc_content(optimized),
            "optimization_log": log,
            "length": len(sequence)
        })
    
    return results

def process_json_file(input_file, use_cli=True):
    """Process JSON file"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'sequence' in item:
                sequence = item['sequence'].upper()
                optimized, log = optimize_sequence(sequence, item.get('id', f'seq_{i}'), use_cli)
                
                results.append({
                    "id": item.get('id', f'seq_{i}'),
                    "description": item.get('description', ''),
                    "original_sequence": sequence,
                    "optimized_sequence": optimized,
                    "original_gc_content": calculate_gc_content(sequence),
                    "optimized_gc_content": calculate_gc_content(optimized),
                    "optimization_log": log,
                    "length": len(sequence)
                })
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and set(value.upper()).issubset(set("ATGC")):
                sequence = value.upper()
                optimized, log = optimize_sequence(sequence, key, use_cli)
                
                results.append({
                    "id": key,
                    "description": "",
                    "original_sequence": sequence,
                    "optimized_sequence": optimized,
                    "original_gc_content": calculate_gc_content(sequence),
                    "optimized_gc_content": calculate_gc_content(optimized),
                    "optimization_log": log,
                    "length": len(sequence)
                })
    
    return results

def process_jsonl_file(input_file, use_cli=True):
    """Process JSONL file"""
    results = []
    
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                if isinstance(item, dict) and 'sequence' in item:
                    sequence = item['sequence'].upper()
                    optimized, log = optimize_sequence(sequence, item.get('id', f'seq_{i}'), use_cli)
                    
                    results.append({
                        "id": item.get('id', f'seq_{i}'),
                        "description": item.get('description', ''),
                        "original_sequence": sequence,
                        "optimized_sequence": optimized,
                        "original_gc_content": calculate_gc_content(sequence),
                        "optimized_gc_content": calculate_gc_content(optimized),
                        "optimization_log": log,
                        "length": len(sequence)
                    })
            except json.JSONDecodeError:
                continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="DNAChisel CLI wrapper for DNA sequence optimization")
    parser.add_argument("input_file", help="Input FASTA, JSON, or JSONL file")
    parser.add_argument("--output", "-o", help="Output file (default: dnachisel_output.jsonl)")
    parser.add_argument("--format", "-f", choices=['json', 'jsonl'], default='jsonl', help="Output format")
    parser.add_argument("--no-cli", action='store_true', help="Disable CLI and use Python library only")
    
    args = parser.parse_args()
    
    # Process input file
    input_file = args.input_file
    print(f"Processing: {input_file}")
    
    if input_file.endswith(('.fasta', '.fa', '.fna')):
        results = process_fasta_file(input_file, not args.no_cli)
    elif input_file.endswith('.json'):
        results = process_json_file(input_file, not args.no_cli)
    elif input_file.endswith('.jsonl'):
        results = process_jsonl_file(input_file, not args.no_cli)
    else:
        print("Unsupported file format. Please use FASTA, JSON, or JSONL.")
        sys.exit(1)
    
    print(f"Processed {len(results)} sequences")
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        output_file = f"dnachisel_output.{args.format}"
    
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
            print(f"  {result['id']}: {result['length']} bp, GC: {result['original_gc_content']:.1f}% -> {result['optimized_gc_content']:.1f}%")
        if len(results) > 3:
            print(f"  ... and {len(results) - 3} more sequences")

if __name__ == "__main__":
    main()
