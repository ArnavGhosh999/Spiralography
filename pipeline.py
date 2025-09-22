#!/usr/bin/env python3
"""
Comprehensive DNA/RNA Analysis Pipeline
Integrates multiple bioinformatics tools with CLI support where available
"""

import subprocess
import os
import sys
import shutil
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# Get the current directory as base directory
BASE_DIR = os.getcwd()


def get_python_executable() -> str:
    """Return project venv python if present, otherwise current interpreter."""
    venv_python = os.path.join(BASE_DIR, '.venv', 'bin', 'python')
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable

def run_script(script_name: str, input_file: str, output_file: Optional[str] = None) -> str:
    """Run a Python script with input file"""
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    print(f"🔄 Running {script_name} on input {input_file}...")
    
    cmd = [get_python_executable(), script_path, input_file]
    if output_file:
        cmd.extend(['--output', output_file])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {script_name} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return output_file or get_default_output(script_name.split('.')[0])
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed: {e}")
        print(f"   Error: {e.stderr}")
        raise

def run_cli_tool(tool_name: str, input_file: str, output_file: Optional[str] = None) -> str:
    """Run a CLI tool with input file"""
    print(f"🔄 Running {tool_name} CLI on input {input_file}...")
    
    if output_file is None:
        output_file = get_default_output(tool_name)
    
    cmd = [tool_name, input_file, '--output', output_file]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {tool_name} CLI completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return output_file
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️ {tool_name} CLI not available, falling back to Python script...")
        return run_script(f"{tool_name}_cli.py", input_file, output_file)

def get_default_output(tool_name: str) -> str:
    """Get default output filename for a tool"""
    output_mapping = {
        'biopython': 'output.jsonl',
        'crispr': 'crispr_guides.jsonl',
        'mrnaid': 'optimized_mrna.jsonl',
        'rbscalc': 'rbs_output.jsonl',
        'viennarna': 'viennarna_output.jsonl',
        'dnachisel': 'dnachisel_output.jsonl',
        'copasi': 'copasi_output.jsonl',
        'iedb': 'iedb_output.jsonl'
    }
    return output_mapping.get(tool_name, f'{tool_name}_output.jsonl')

def check_tool_availability(tool_name: str) -> bool:
    """Check if a CLI tool is available"""
    try:
        subprocess.run([tool_name, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def copy_to_downloads(filepath: str) -> None:
    """Copy file to Downloads directory"""
    downloads = str(Path.home() / "Downloads")
    destination = os.path.join(downloads, os.path.basename(filepath))
    shutil.copy(filepath, destination)
    print(f"📁 Copied final output to {destination}")

def create_pipeline_report(results: Dict) -> None:
    """Create a comprehensive pipeline report"""
    report_file = os.path.join(BASE_DIR, 'pipeline_report.json')
    
    report = {
        'pipeline_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_directory': BASE_DIR,
            'python_version': sys.version,
            'tools_used': list(results.keys())
        },
        'results': results
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Pipeline report saved to: {report_file}")

def therapeutics_pipeline(initial_fasta: str) -> None:
    """
    Complete therapeutics pipeline with CLI support where available
    """
    print("🚀 Starting Comprehensive Therapeutics Pipeline")
    print("=" * 60)
    
    # Ensure input file exists
    input_path = os.path.join(BASE_DIR, initial_fasta)
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    results = {}
    current_file = input_path
    
    try:
        # Step 1: Biopython - Sequence parsing and basic analysis
        print("\n📋 Step 1: Biopython Analysis")
        print("-" * 30)
        biopython_output = run_script('biopython.py', current_file, 'output.jsonl')
        results['biopython'] = {
            'input': current_file,
            'output': biopython_output,
            'status': 'completed'
        }
        current_file = biopython_output
        
        # Step 2: CRISPR Guide Design
        print("\n✂️ Step 2: CRISPR Guide Design")
        print("-" * 30)
        crispr_output = run_script('crispr.py', current_file, 'crispr_guides.jsonl')
        results['crispr'] = {
            'input': current_file,
            'output': crispr_output,
            'status': 'completed'
        }
        current_file = crispr_output
        
        # Step 3: mRNA Optimization
        print("\n🧬 Step 3: mRNA Optimization")
        print("-" * 30)
        mrnaid_output = run_script('mrnaid.py', current_file, 'optimized_mrna.jsonl')
        results['mrnaid'] = {
            'input': current_file,
            'output': mrnaid_output,
            'status': 'completed'
        }
        current_file = mrnaid_output
        
        # Step 4: RBS Calculator
        print("\n🧮 Step 4: RBS Calculator")
        print("-" * 30)
        rbscalc_output = run_script('rbscalc.py', current_file, 'rbs_output.jsonl')
        results['rbscalc'] = {
            'input': current_file,
            'output': rbscalc_output,
            'status': 'completed'
        }
        current_file = rbscalc_output
        
        # Step 5: ViennaRNA Structure Prediction
        print("\n🔬 Step 5: ViennaRNA Structure Prediction")
        print("-" * 30)
        if check_tool_availability('RNAfold'):
            viennarna_output = run_cli_tool('viennarna', current_file, 'viennarna_output.jsonl')
        else:
            viennarna_output = run_script('viennarna_cli.py', current_file, 'viennarna_output.jsonl')
        results['viennarna'] = {
            'input': current_file,
            'output': viennarna_output,
            'status': 'completed'
        }
        current_file = viennarna_output
        
        # Step 6: DNAChisel Optimization
        print("\n⚡ Step 6: DNAChisel Optimization")
        print("-" * 30)
        if check_tool_availability('dnachisel'):
            dnachisel_output = run_cli_tool('dnachisel', current_file, 'dnachisel_output.jsonl')
        else:
            dnachisel_output = run_script('dnachisel_cli.py', current_file, 'dnachisel_output.jsonl')
        results['dnachisel'] = {
            'input': current_file,
            'output': dnachisel_output,
            'status': 'completed'
        }
        current_file = dnachisel_output

        # Constrained optimization to satisfy CAI/GC targets
        print("\n🧬 Step 6b: Constrained Optimization (GC 40–60%, CAI≥0.7)")
        print("-" * 30)
        constrained_output = run_script('constrained_opt.py', 'optimized_mrna.jsonl', 'optimized_mrna_constrained.jsonl')
        results['constrained_opt'] = {
            'input': 'optimized_mrna.jsonl',
            'output': constrained_output,
            'status': 'completed'
        }
        
        # Step 7: COPASI Biochemical Simulation
        print("\n🧪 Step 7: COPASI Biochemical Simulation")
        print("-" * 30)
        if check_tool_availability('copasi'):
            copasi_output = run_cli_tool('copasi', current_file, 'copasi_output.jsonl')
        else:
            copasi_output = run_script('copasi_cli.py', current_file, 'copasi_output.jsonl')
        results['copasi'] = {
            'input': current_file,
            'output': copasi_output,
            'status': 'completed'
        }
        current_file = copasi_output
        
        # Step 8: IEDB Epitope Analysis
        print("\n🦠 Step 8: IEDB Epitope Analysis")
        print("-" * 30)
        iedb_output = run_script('iedb_analysis.py', current_file, 'iedb_output.jsonl')
        results['iedb'] = {
            'input': current_file,
            'output': iedb_output,
            'status': 'completed'
        }

        # Step 9: Compliance Filters (based on available outputs)
        print("\n✅ Step 9: Compliance Filters")
        print("-" * 30)
        compliance_output = 'compliance_report.json'
        try:
            _ = run_script('compliance_filters.py', 'optimized_mrna.jsonl')
        except Exception:
            # Fallback: run explicitly with python and flags
            try:
                result = subprocess.run([sys.executable, os.path.join(BASE_DIR, 'compliance_filters.py'),
                                         '--optimized_jsonl', 'optimized_mrna.jsonl',
                                         '--mhc_csv', os.path.join('results', 'iedb_analysis', 'mhc_class_i_predictions.csv'),
                                         '--out', compliance_output], check=True, capture_output=True, text=True)
                print(result.stdout.strip())
            except Exception as e:
                print(f"⚠️ Compliance filters failed: {e}")
        results['compliance'] = {
            'input': 'optimized_mrna.jsonl',
            'output': compliance_output,
            'status': 'completed'
        }
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 60)
        print(f"📁 Final output: {current_file}")
        
        # Create pipeline report
        create_pipeline_report(results)
        
        # Copy final output to downloads
        copy_to_downloads(current_file)
        
        # Print summary
        print("\n📊 Pipeline Summary:")
        for step, data in results.items():
            status_icon = "✅" if data['status'] == 'completed' else "❌"
            print(f"  {status_icon} {step}: {data['output']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed at step: {e}")
        print("Creating partial results report...")
        create_pipeline_report(results)
        raise

def main():
    """Main function with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive DNA/RNA Analysis Pipeline")
    parser.add_argument("input_file", nargs='?', default='sequence_5.fasta', 
                       help="Input FASTA file (default: sequence_5.fasta)")
    parser.add_argument("--list-tools", action='store_true', 
                       help="List available tools and their status")
    
    args = parser.parse_args()
    
    if args.list_tools:
        print("🔧 Available Tools Status:")
        print("-" * 30)
        tools = ['RNAfold', 'dnachisel', 'copasi']
        for tool in tools:
            status = "✅ Available" if check_tool_availability(tool) else "❌ Not available"
            print(f"  {tool}: {status}")
        return
    
    therapeutics_pipeline(args.input_file)

if __name__ == "__main__":
    main()
