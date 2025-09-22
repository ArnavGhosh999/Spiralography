#!/usr/bin/env python3
"""
COPASI CLI wrapper for biochemical network simulation
Input: FASTA/JSON/JSONL files (converted to protein sequences)
Output: Simulation results in JSON/JSONL format
"""

import subprocess
import sys
import json
import os
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import tempfile
import time

def install_copasi():
    """Install COPASI if not available"""
    try:
        subprocess.run(['copasi', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("COPASI CLI not found. Using Python fallback...")
        return False

def translate_mrna_to_protein(mrna_sequence):
    """Translate mRNA sequence to protein"""
    mrna_clean = mrna_sequence.replace('\n', '').replace(' ', '').upper()
    seq_obj = Seq(mrna_clean)
    protein_seq = str(seq_obj.translate())
    if protein_seq.endswith('*'):
        protein_seq = protein_seq[:-1]
    return protein_seq

def create_simple_network_model(protein_sequence, output_file):
    """Create a simple biochemical network model"""
    # This is a simplified model for demonstration
    # In practice, you would create a more complex SBML model
    
    model_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="spike_protein_network" name="SARS-CoV-2 Spike Protein Network">
    <listOfCompartments>
      <compartment id="extracellular" name="Extracellular" size="1"/>
      <compartment id="membrane" name="Membrane" size="0.1"/>
      <compartment id="intracellular" name="Intracellular" size="1"/>
    </listOfCompartments>
    
    <listOfSpecies>
      <species id="Spike_mRNA" name="Spike mRNA" compartment="intracellular" initialConcentration="100"/>
      <species id="Spike_Protein" name="Spike Protein" compartment="membrane" initialConcentration="0"/>
      <species id="Spike_Trimer" name="Spike Trimer" compartment="membrane" initialConcentration="0"/>
      <species id="ACE2_Receptor" name="ACE2 Receptor" compartment="membrane" initialConcentration="50"/>
      <species id="Virus_ACE2_Complex" name="Virus-ACE2 Complex" compartment="membrane" initialConcentration="0"/>
      <species id="Internalized_Virus" name="Internalized Virus" compartment="intracellular" initialConcentration="0"/>
    </listOfSpecies>
    
    <listOfReactions>
      <reaction id="mRNA_Translation" name="mRNA Translation" reversible="false">
        <listOfReactants>
          <speciesReference species="Spike_mRNA" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Spike_Protein" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <ci> k1 * Spike_mRNA </ci>
          </math>
          <listOfParameters>
            <parameter id="k1" value="0.1"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
      
      <reaction id="Protein_Trimerization" name="Protein Trimerization" reversible="true">
        <listOfReactants>
          <speciesReference species="Spike_Protein" stoichiometry="3"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Spike_Trimer" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <ci> k2 * Spike_Protein^3 - k3 * Spike_Trimer </ci>
          </math>
          <listOfParameters>
            <parameter id="k2" value="0.01"/>
            <parameter id="k3" value="0.001"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
      
      <reaction id="Receptor_Binding" name="Receptor Binding" reversible="true">
        <listOfReactants>
          <speciesReference species="Spike_Trimer" stoichiometry="1"/>
          <speciesReference species="ACE2_Receptor" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Virus_ACE2_Complex" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <ci> k4 * Spike_Trimer * ACE2_Receptor - k5 * Virus_ACE2_Complex </ci>
          </math>
          <listOfParameters>
            <parameter id="k4" value="0.05"/>
            <parameter id="k5" value="0.01"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
      
      <reaction id="Virus_Internalization" name="Virus Internalization" reversible="false">
        <listOfReactants>
          <speciesReference species="Virus_ACE2_Complex" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Internalized_Virus" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <ci> k6 * Virus_ACE2_Complex </ci>
          </math>
          <listOfParameters>
            <parameter id="k6" value="0.02"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
    
    with open(output_file, 'w') as f:
        f.write(model_content)

def run_copasi_simulation(sbml_file, output_dir):
    """Run COPASI simulation"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run time course simulation
        cmd = [
            'copasi',
            '--import', sbml_file,
            '--export', os.path.join(output_dir, 'simulation_results.csv'),
            '--run'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse results (simplified)
        results = {
            'simulation_status': 'completed',
            'output_file': os.path.join(output_dir, 'simulation_results.csv'),
            'log': result.stdout,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return results
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"COPASI simulation failed: {e}")
        return {
            'simulation_status': 'failed',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

def simulate_biochemical_network_python(protein_sequence):
    """Fallback to Python simulation"""
    try:
        import numpy as np
        from scipy.integrate import odeint
        
        # Simple ODE model
        def model(y, t, params):
            mrna, protein, trimer, receptor, complex_v, internalized = y
            k1, k2, k3, k4, k5, k6 = params
            
            dydt = [
                -k1 * mrna,  # mRNA degradation
                k1 * mrna - 3*k2 * protein**3 + k3 * trimer,  # Protein
                3*k2 * protein**3 - k3 * trimer - k4 * trimer * receptor + k5 * complex_v,  # Trimer
                -k4 * trimer * receptor + k5 * complex_v,  # Receptor
                k4 * trimer * receptor - k5 * complex_v - k6 * complex_v,  # Complex
                k6 * complex_v  # Internalized
            ]
            return dydt
        
        # Parameters
        params = [0.1, 0.01, 0.001, 0.05, 0.01, 0.02]
        
        # Initial conditions
        y0 = [100, 0, 0, 50, 0, 0]  # mRNA, protein, trimer, receptor, complex, internalized
        
        # Time points
        t = np.linspace(0, 100, 1000)
        
        # Solve ODE
        solution = odeint(model, y0, t, args=(params,))
        
        # Extract final values
        final_values = solution[-1]
        
        return {
            'simulation_status': 'completed',
            'final_concentrations': {
                'mRNA': float(final_values[0]),
                'Protein': float(final_values[1]),
                'Trimer': float(final_values[2]),
                'Receptor': float(final_values[3]),
                'Complex': float(final_values[4]),
                'Internalized': float(final_values[5])
            },
            'time_points': t.tolist(),
            'concentrations': solution.tolist(),
            'method': 'python_ode'
        }
        
    except ImportError:
        print("SciPy not available for simulation")
        return {
            'simulation_status': 'failed',
            'error': 'SciPy not available',
            'method': 'python_fallback'
        }
    except Exception as e:
        print(f"Python simulation failed: {e}")
        return {
            'simulation_status': 'failed',
            'error': str(e),
            'method': 'python_fallback'
        }

def process_fasta_file(input_file, use_cli=True):
    """Process FASTA file"""
    results = []
    
    for record in SeqIO.parse(input_file, "fasta"):
        sequence = str(record.seq).upper()
        
        # Translate to protein if it looks like mRNA
        if set(sequence).issubset(set("ATGC")):
            protein_seq = translate_mrna_to_protein(sequence)
        else:
            protein_seq = sequence
        
        # Run simulation
        if use_cli:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_file:
                create_simple_network_model(protein_seq, tmp_file.name)
                
                simulation_result = run_copasi_simulation(tmp_file.name, f"copasi_output_{record.id}")
                os.unlink(tmp_file.name)
        else:
            simulation_result = simulate_biochemical_network_python(protein_seq)
        
        results.append({
            "id": record.id,
            "description": record.description,
            "sequence": sequence,
            "protein_sequence": protein_seq,
            "simulation_results": simulation_result,
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
                
                # Translate to protein if it looks like mRNA
                if set(sequence).issubset(set("ATGC")):
                    protein_seq = translate_mrna_to_protein(sequence)
                else:
                    protein_seq = sequence
                
                # Run simulation
                if use_cli:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_file:
                        create_simple_network_model(protein_seq, tmp_file.name)
                        
                        simulation_result = run_copasi_simulation(tmp_file.name, f"copasi_output_{item.get('id', f'seq_{i}')}")
                        os.unlink(tmp_file.name)
                else:
                    simulation_result = simulate_biochemical_network_python(protein_seq)
                
                results.append({
                    "id": item.get('id', f'seq_{i}'),
                    "description": item.get('description', ''),
                    "sequence": sequence,
                    "protein_sequence": protein_seq,
                    "simulation_results": simulation_result,
                    "length": len(sequence)
                })
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                sequence = value.upper()
                
                # Translate to protein if it looks like mRNA
                if set(sequence).issubset(set("ATGC")):
                    protein_seq = translate_mrna_to_protein(sequence)
                else:
                    protein_seq = sequence
                
                # Run simulation
                if use_cli:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_file:
                        create_simple_network_model(protein_seq, tmp_file.name)
                        
                        simulation_result = run_copasi_simulation(tmp_file.name, f"copasi_output_{key}")
                        os.unlink(tmp_file.name)
                else:
                    simulation_result = simulate_biochemical_network_python(protein_seq)
                
                results.append({
                    "id": key,
                    "description": "",
                    "sequence": sequence,
                    "protein_sequence": protein_seq,
                    "simulation_results": simulation_result,
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
                    
                    # Translate to protein if it looks like mRNA
                    if set(sequence).issubset(set("ATGC")):
                        protein_seq = translate_mrna_to_protein(sequence)
                    else:
                        protein_seq = sequence
                    
                    # Run simulation
                    if use_cli:
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_file:
                            create_simple_network_model(protein_seq, tmp_file.name)
                            
                            simulation_result = run_copasi_simulation(tmp_file.name, f"copasi_output_{item.get('id', f'seq_{i}')}")
                            os.unlink(tmp_file.name)
                    else:
                        simulation_result = simulate_biochemical_network_python(protein_seq)
                    
                    results.append({
                        "id": item.get('id', f'seq_{i}'),
                        "description": item.get('description', ''),
                        "sequence": sequence,
                        "protein_sequence": protein_seq,
                        "simulation_results": simulation_result,
                        "length": len(sequence)
                    })
            except json.JSONDecodeError:
                continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="COPASI CLI wrapper for biochemical network simulation")
    parser.add_argument("input_file", help="Input FASTA, JSON, or JSONL file")
    parser.add_argument("--output", "-o", help="Output file (default: copasi_output.jsonl)")
    parser.add_argument("--format", "-f", choices=['json', 'jsonl'], default='jsonl', help="Output format")
    parser.add_argument("--no-cli", action='store_true', help="Disable CLI and use Python simulation only")
    
    args = parser.parse_args()
    
    # Check if COPASI is available
    if not args.no_cli:
        install_copasi()
    
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
        output_file = f"copasi_output.{args.format}"
    
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
            status = result['simulation_results'].get('simulation_status', 'unknown')
            print(f"  {result['id']}: {result['length']} bp, Status: {status}")
        if len(results) > 3:
            print(f"  ... and {len(results) - 3} more sequences")

if __name__ == "__main__":
    main()
