# 2nd tools- •	COPASI;	Input: Biochemical network model (SBML/XML/CSV); Output: Simulation results (time-course data, steady-state analysis, plots, CSV)

# Cell 1: Imports and Setup
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Install and import COPASI
try:
    import COPASI as copasi
    print("✅ COPASI imported successfully")
except ImportError:
    print("❌ COPASI not found. Installing python-copasi...")
    os.system("pip install python-copasi")
    import COPASI as copasi
    print("✅ COPASI installed and imported")

# Set up paths
INPUT_FILE = os.path.join("assets", "SarsCov2SpikemRNA.fasta")
OUTPUT_DIR = os.path.join("results", "copasi_results")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COPASI simulation parameters
SIMULATION_PARAMS = {
    'duration': 100.0,      # Simulation time (arbitrary units)
    'step_size': 0.1,       # Time step
    'intervals': 1000,      # Number of intervals
    'output_events': True,
    'output_steady_state': True
}

# Initialize results storage
results = {
    'sequence_info': {},
    'model_info': {},
    'time_course_data': {},
    'steady_state_data': {},
    'parameter_analysis': {},
    'analysis_metadata': {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': INPUT_FILE,
        'output_dir': OUTPUT_DIR,
        'copasi_version': copasi.CVersion.VERSION.getVersionDevel()
    }
}

print("✅ Setup completed successfully!")
print(f"Current working directory: {os.getcwd()}")
print(f"Input file: {INPUT_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"COPASI version: {copasi.CVersion.VERSION.getVersionDevel()}")
print(f"Simulation duration: {SIMULATION_PARAMS['duration']} time units")
print(f"Step size: {SIMULATION_PARAMS['step_size']}")


# Cell 2: Sequence Loading and Model Creation

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
    except Exception as e:
        print(f"❌ Error loading sequence: {e}")
        return {}

def translate_mrna_to_protein(mrna_sequence: str) -> str:
    """Translate mRNA sequence to protein"""
    mrna_clean = mrna_sequence.replace('\n', '').replace(' ', '').upper()
    seq_obj = Seq(mrna_clean)
    protein_seq = str(seq_obj.translate())
    if protein_seq.endswith('*'):
        protein_seq = protein_seq[:-1]
    return protein_seq

def create_spike_protein_model(protein_sequence: str) -> copasi.CDataModel:
    """Create a biochemical network model for spike protein interactions"""
    print("🔄 Creating COPASI biochemical network model...")
    
    # Create a new COPASI data model
    data_model = copasi.CRootContainer.addDatamodel()
    
    # Get the model from the data model
    model = data_model.getModel()
    model.setObjectName("SARS_CoV2_Spike_Protein_Network")
    model.setTitle("SARS-CoV-2 Spike Protein Interaction Network")
    
    # Set up model units
    model.setTimeUnit(copasi.CUnit.s)  # seconds
    model.setVolumeUnit(copasi.CUnit.l)  # liters
    model.setQuantityUnit(copasi.CUnit.mol)  # moles
    
    # Create compartments
    compartments = model.getCompartments()
    
    # Extracellular compartment
    extracellular = compartments.createCompartment("extracellular", 1.0)
    extracellular.setStatus(copasi.CModelEntity.Status_FIXED)
    
    # Cell membrane compartment  
    membrane = compartments.createCompartment("membrane", 0.1)
    membrane.setStatus(copasi.CModelEntity.Status_FIXED)
    
    # Intracellular compartment
    intracellular = compartments.createCompartment("intracellular", 1.0)
    intracellular.setStatus(copasi.CModelEntity.Status_FIXED)
    
    # Create species (molecular entities)
    species = model.getMetabolites()
    
    # mRNA species
    spike_mRNA = species.createMetabolite("Spike_mRNA", "intracellular", 100.0, copasi.CModelEntity.Status_REACTIONS)
    
    # Protein species
    spike_protein = species.createMetabolite("Spike_Protein", "membrane", 0.0, copasi.CModelEntity.Status_REACTIONS)
    spike_trimer = species.createMetabolite("Spike_Trimer", "membrane", 0.0, copasi.CModelEntity.Status_REACTIONS)
    
    # Host cell receptor
    ace2_receptor = species.createMetabolite("ACE2_Receptor", "membrane", 50.0, copasi.CModelEntity.Status_REACTIONS)
    
    # Virus-receptor complex
    virus_complex = species.createMetabolite("Virus_ACE2_Complex", "membrane", 0.0, copasi.CModelEntity.Status_REACTIONS)
    
    # Internalized virus
    internalized_virus = species.createMetabolite("Internalized_Virus", "intracellular", 0.0, copasi.CModelEntity.Status_REACTIONS)
    
    # Create reactions
    reactions = model.getReactions()
    
    # Reaction 1: mRNA translation (Spike_mRNA -> Spike_Protein)
    translation = reactions.createReaction("mRNA_Translation")
    translation.addSubstrate(spike_mRNA.getKey(), 1.0)
    translation.addProduct(spike_protein.getKey(), 1.0)
    translation.setReversible(False)
    
    # Reaction 2: Protein trimerization (3 Spike_Protein -> Spike_Trimer)
    trimerization = reactions.createReaction("Protein_Trimerization")
    trimerization.addSubstrate(spike_protein.getKey(), 3.0)
    trimerization.addProduct(spike_trimer.getKey(), 1.0)
    trimerization.setReversible(True)
    
    # Reaction 3: Receptor binding (Spike_Trimer + ACE2_Receptor -> Virus_ACE2_Complex)
    binding = reactions.createReaction("Receptor_Binding")
    binding.addSubstrate(spike_trimer.getKey(), 1.0)
    binding.addSubstrate(ace2_receptor.getKey(), 1.0)
    binding.addProduct(virus_complex.getKey(), 1.0)
    binding.setReversible(True)
    
    # Reaction 4: Virus internalization (Virus_ACE2_Complex -> Internalized_Virus)
    internalization = reactions.createReaction("Virus_Internalization")
    internalization.addSubstrate(virus_complex.getKey(), 1.0)
    internalization.addProduct(internalized_virus.getKey(), 1.0)
    internalization.setReversible(False)
    
    # Set kinetic parameters
    set_reaction_kinetics(translation, "translation", k1=0.1)
    set_reaction_kinetics(trimerization, "trimerization", k1=0.01, k2=0.001)
    set_reaction_kinetics(binding, "binding", k1=0.05, k2=0.01)  
    set_reaction_kinetics(internalization, "internalization", k1=0.02)
    
    print("✅ Biochemical network model created successfully")
    print(f"   Compartments: {compartments.size()}")
    print(f"   Species: {species.size()}")
    print(f"   Reactions: {reactions.size()}")
    
    return data_model

def set_reaction_kinetics(reaction, reaction_type: str, k1: float, k2: float = None):
    """Set kinetic law for reactions"""
    kinetic_law = reaction.getKineticLaw()
    
    if reaction_type == "translation":
        # First-order kinetics: v = k1 * [mRNA]
        kinetic_law.setObjectName("Mass Action")
        kinetic_law.addParameterDescription("k1", k1, copasi.CUnit.s_1)
        
    elif reaction_type == "trimerization":
        # Forward: v = k1 * [Protein]^3, Reverse: v = k2 * [Trimer]
        kinetic_law.setObjectName("Mass Action")
        kinetic_law.addParameterDescription("k1", k1, copasi.CUnit.l2_per_mol2_per_s)
        if k2:
            kinetic_law.addParameterDescription("k2", k2, copasi.CUnit.s_1)
            
    elif reaction_type == "binding":
        # Forward: v = k1 * [Trimer] * [ACE2], Reverse: v = k2 * [Complex]
        kinetic_law.setObjectName("Mass Action") 
        kinetic_law.addParameterDescription("k1", k1, copasi.CUnit.l_per_mol_per_s)
        if k2:
            kinetic_law.addParameterDescription("k2", k2, copasi.CUnit.s_1)
            
    elif reaction_type == "internalization":
        # First-order kinetics: v = k1 * [Complex]
        kinetic_law.setObjectName("Mass Action")
        kinetic_law.addParameterDescription("k1", k1, copasi.CUnit.s_1)

# Load the SARS-CoV-2 spike mRNA sequence
print("🔄 Loading SARS-CoV-2 spike mRNA sequence...")

sequences = {}
protein_seq = ""

if os.path.exists(INPUT_FILE):
    print(f"✅ Found file: {INPUT_FILE}")
    sequences = load_fasta_sequence(INPUT_FILE)
    
    if sequences:
        # Process the first sequence
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
            'protein_sequence': protein_seq,
            'mrna_sequence': mrna_seq[:200] + "..." if len(mrna_seq) > 200 else mrna_seq
        }
    else:
        print("❌ Failed to load sequences from file")
else:
    print(f"❌ File not found: {INPUT_FILE}")
    print("Using sample data for demonstration...")
    
    # Use sample spike protein sequence
    protein_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYKNNSIAPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQD"
    
    results['sequence_info'] = {
        'sequence_id': 'sample_spike_protein',
        'protein_length': len(protein_seq),
        'protein_sequence': protein_seq,
        'note': 'Sample data used - original file not found'
    }

# Create the biochemical network model
if protein_seq:
    try:
        copasi_model = create_spike_protein_model(protein_seq)
        
        # Store model information
        model = copasi_model.getModel()
        results['model_info'] = {
            'model_name': model.getObjectName(),
            'model_title': model.getTitle(),
            'compartments': model.getCompartments().size(),
            'species': model.getMetabolites().size(),
            'reactions': model.getReactions().size(),
            'parameters': len(SIMULATION_PARAMS)
        }
        
        print("✅ Sequence processing and model creation completed!")
        
    except Exception as e:
        print(f"❌ Error creating COPASI model: {e}")
        copasi_model = None
else:
    print("❌ No protein sequence available for model creation")
    copasi_model = None


# Cell 3: Time Course Simulation

def run_time_course_simulation(data_model: copasi.CDataModel, duration: float, intervals: int) -> Dict:
    """Run time course simulation and return results"""
    print("🔄 Running time course simulation...")
    
    try:
        # Get the trajectory task
        task_list = data_model.getTaskList()
        trajectory_task = task_list.get("Time-Course")
        
        if trajectory_task is None:
            print("❌ Time-Course task not found")
            return {}
        
        # Set up the trajectory task
        trajectory_task.setMethodType(copasi.CTaskEnum.Method_deterministic)
        
        # Get the method and set parameters
        method = trajectory_task.getMethod()
        
        # Set simulation parameters
        problem = trajectory_task.getProblem()
        problem.setModel(data_model.getModel())
        
        # Set time course parameters
        problem.setStepNumber(intervals)
        problem.setDuration(duration)
        problem.setTimeSeriesRequested(True)
        
        # Set automatic step size
        problem.setAutomaticStepSize(True)
        
        # Run the simulation
        print(f"   Duration: {duration} time units")
        print(f"   Intervals: {intervals}")
        
        result = trajectory_task.process(True)
        
        if not result:
            print("❌ Time course simulation failed")
            print(f"Error: {copasi.CCopasiMessage.getAllMessageText()}")
            return {}
        
        # Get time series data
        time_series = trajectory_task.getTimeSeries()
        
        if time_series is None:
            print("❌ No time series data generated")
            return {}
        
        # Extract time series data
        num_variables = time_series.getNumVariables()
        num_steps = time_series.getRecordedSteps()
        
        print(f"✅ Simulation completed successfully")
        print(f"   Variables: {num_variables}")
        print(f"   Time steps: {num_steps}")
        
        # Create data structure for results
        time_course_data = {
            'time': [],
            'species_data': {},
            'metadata': {
                'num_variables': num_variables,
                'num_steps': num_steps,
                'duration': duration,
                'intervals': intervals
            }
        }
        
        # Get variable titles
        titles = []
        for i in range(num_variables):
            titles.append(time_series.getTitle(i))
        
        # Extract data for each time point
        for step in range(num_steps):
            # Get time point
            time_point = time_series.getConcentrationData(step, 0)  # First column is time
            time_course_data['time'].append(time_point)
            
            # Get species concentrations
            for var_idx in range(1, num_variables):  # Skip time column
                var_name = titles[var_idx]
                concentration = time_series.getConcentrationData(step, var_idx)
                
                if var_name not in time_course_data['species_data']:
                    time_course_data['species_data'][var_name] = []
                
                time_course_data['species_data'][var_name].append(concentration)
        
        # Print summary of key species
        print("   Key species final concentrations:")
        for species_name, concentrations in time_course_data['species_data'].items():
            if len(concentrations) > 0:
                final_conc = concentrations[-1]
                print(f"     {species_name}: {final_conc:.6f}")
        
        return time_course_data
        
    except Exception as e:
        print(f"❌ Error in time course simulation: {e}")
        return {}

def analyze_time_course_results(time_course_data: Dict) -> Dict:
    """Analyze time course simulation results"""
    print("🔄 Analyzing time course results...")
    
    if not time_course_data or 'species_data' not in time_course_data:
        print("❌ No time course data to analyze")
        return {}
    
    analysis = {
        'peak_concentrations': {},
        'final_concentrations': {},
        'time_to_peak': {},
        'area_under_curve': {},
        'half_life': {}
    }
    
    time_points = np.array(time_course_data['time'])
    
    for species_name, concentrations in time_course_data['species_data'].items():
        conc_array = np.array(concentrations)
        
        # Peak concentration
        max_conc = np.max(conc_array)
        max_idx = np.argmax(conc_array)
        time_to_peak = time_points[max_idx]
        
        analysis['peak_concentrations'][species_name] = max_conc
        analysis['time_to_peak'][species_name] = time_to_peak
        
        # Final concentration
        analysis['final_concentrations'][species_name] = conc_array[-1]
        
        # Area under curve (simple trapezoidal rule)
        if len(conc_array) > 1:
            auc = np.trapz(conc_array, time_points)
            analysis['area_under_curve'][species_name] = auc
        
        # Half-life estimation (time to reach 50% of peak)
        if max_conc > 0:
            half_max = max_conc * 0.5
            # Find first time point after peak where concentration drops to half-max
            post_peak_idx = max_idx
            half_life_idx = None
            
            for i in range(post_peak_idx, len(conc_array)):
                if conc_array[i] <= half_max:
                    half_life_idx = i
                    break
            
            if half_life_idx is not None:
                analysis['half_life'][species_name] = time_points[half_life_idx] - time_points[max_idx]
            else:
                analysis['half_life'][species_name] = None
    
    print("✅ Time course analysis completed")
    return analysis

# Run time course simulation if model exists
if 'copasi_model' in locals() and copasi_model is not None:
    print("🚀 Starting time course simulation...")
    
    # Run the simulation
    time_course_results = run_time_course_simulation(
        copasi_model, 
        SIMULATION_PARAMS['duration'], 
        SIMULATION_PARAMS['intervals']
    )
    
    if time_course_results:
        # Store results
        results['time_course_data'] = time_course_results
        
        # Analyze results  
        time_course_analysis = analyze_time_course_results(time_course_results)
        results['time_course_analysis'] = time_course_analysis
        
        # Print key findings
        print("\n📊 TIME COURSE SIMULATION RESULTS:")
        print("="*50)
        
        if 'metadata' in time_course_results:
            metadata = time_course_results['metadata']
            print(f"Simulation duration: {metadata['duration']} time units")
            print(f"Number of time points: {metadata['num_steps']}")
            print(f"Variables tracked: {metadata['num_variables']}")
        
        print("\nPeak Concentrations:")
        if 'peak_concentrations' in time_course_analysis:
            for species, peak in time_course_analysis['peak_concentrations'].items():
                time_to_peak = time_course_analysis.get('time_to_peak', {}).get(species, 0)
                print(f"  {species}: {peak:.6f} (at t={time_to_peak:.2f})")
        
        print("\nFinal Concentrations:")
        if 'final_concentrations' in time_course_analysis:
            for species, final in time_course_analysis['final_concentrations'].items():
                print(f"  {species}: {final:.6f}")
        
    else:
        print("❌ Time course simulation failed")
        
else:
    print("❌ No COPASI model available for simulation")


# Cell 4: Steady State Analysis

def run_steady_state_analysis(data_model: copasi.CDataModel) -> Dict:
    """Run steady state analysis and return results"""
    print("🔄 Running steady state analysis...")
    
    try:
        # Get the steady state task
        task_list = data_model.getTaskList()
        steady_state_task = task_list.get("Steady-State")
        
        if steady_state_task is None:
            print("❌ Steady-State task not found")
            return {}
        
        # Set up the steady state task
        steady_state_task.setMethodType(copasi.CTaskEnum.Method_Newton)
        
        # Get the problem and set parameters
        problem = steady_state_task.getProblem()
        problem.setModel(data_model.getModel())
        
        # Run the steady state analysis
        print("   Calculating steady state...")
        result = steady_state_task.process(True)
        
        if not result:
            print("❌ Steady state analysis failed")
            print(f"Error: {copasi.CCopasiMessage.getAllMessageText()}")
            return {}
        
        # Get steady state results
        steady_state = data_model.getModel().getSteadyState()
        
        steady_state_data = {
            'species_concentrations': {},
            'reaction_fluxes': {},
            'stability_analysis': {},
            'metadata': {
                'status': 'completed',
                'resolution': steady_state.getResolution(),
                'max_time': steady_state.getTime()
            }
        }
        
        # Get species concentrations at steady state
        metabolites = data_model.getModel().getMetabolites()
        print("   Species concentrations at steady state:")
        
        for i in range(metabolites.size()):
            metabolite = metabolites.get(i)
            name = metabolite.getObjectName()
            concentration = metabolite.getConcentration()
            
            steady_state_data['species_concentrations'][name] = concentration
            print(f"     {name}: {concentration:.6e}")
        
        # Get reaction fluxes at steady state
        reactions = data_model.getModel().getReactions()
        print("   Reaction fluxes at steady state:")
        
        for i in range(reactions.size()):
            reaction = reactions.get(i)
            name = reaction.getObjectName()
            flux = reaction.getFlux()
            
            steady_state_data['reaction_fluxes'][name] = flux
            print(f"     {name}: {flux:.6e}")
        
        # Get eigenvalues for stability analysis if available
        try:
            eigenvalues_real = steady_state.getEigenValuesReal()
            eigenvalues_imag = steady_state.getEigenValuesImag()
            
            if eigenvalues_real.size() > 0:
                print("   Eigenvalues (stability analysis):")
                stability_analysis = {}
                
                for i in range(eigenvalues_real.size()):
                    real_part = eigenvalues_real[i]
                    imag_part = eigenvalues_imag[i] if i < eigenvalues_imag.size() else 0.0
                    
                    eigenvalue_name = f"eigenvalue_{i+1}"
                    stability_analysis[eigenvalue_name] = {
                        'real': real_part,
                        'imaginary': imag_part,
                        'stable': real_part < 0
                    }
                    
                    stability_indicator = "stable" if real_part < 0 else "unstable"
                    print(f"     λ{i+1}: {real_part:.6e} + {imag_part:.6e}i ({stability_indicator})")
                
                steady_state_data['stability_analysis'] = stability_analysis
                
                # Overall stability assessment
                all_stable = all(ev['stable'] for ev in stability_analysis.values())
                steady_state_data['overall_stability'] = 'stable' if all_stable else 'unstable'
                
        except Exception as e:
            print(f"   Stability analysis not available: {e}")
            steady_state_data['stability_analysis'] = {}
        
        print("✅ Steady state analysis completed")
        return steady_state_data
        
    except Exception as e:
        print(f"❌ Error in steady state analysis: {e}")
        return {}

def perform_parameter_sensitivity_analysis(data_model: copasi.CDataModel) -> Dict:
    """Perform parameter sensitivity analysis"""
    print("🔄 Running parameter sensitivity analysis...")
    
    try:
        # Get the sensitivity task
        task_list = data_model.getTaskList()
        sensitivity_task = task_list.get("Sensitivities")
        
        if sensitivity_task is None:
            print("⚠️ Sensitivity task not available in this COPASI version")
            return {}
        
        # Set up sensitivity analysis
        problem = sensitivity_task.getProblem()
        problem.setModel(data_model.getModel())
        
        # Run sensitivity analysis
        result = sensitivity_task.process(True)
        
        if not result:
            print("❌ Sensitivity analysis failed")
            return {}
        
        # Extract sensitivity results (simplified)
        sensitivity_data = {
            'status': 'completed',
            'note': 'Sensitivity analysis completed - detailed results would require specific COPASI API calls'
        }
        
        print("✅ Parameter sensitivity analysis completed")
        return sensitivity_data
        
    except Exception as e:
        print(f"⚠️ Sensitivity analysis not available: {e}")
        return {'status': 'not_available', 'reason': str(e)}

def analyze_reaction_network(data_model: copasi.CDataModel) -> Dict:
    """Analyze the biochemical reaction network properties"""
    print("🔄 Analyzing reaction network properties...")
    
    model = data_model.getModel()
    
    # Get network components
    compartments = model.getCompartments()
    metabolites = model.getMetabolites()
    reactions = model.getReactions()
    
    network_analysis = {
        'network_size': {
            'compartments': compartments.size(),
            'species': metabolites.size(),
            'reactions': reactions.size()
        },
        'compartment_info': {},
        'species_info': {},
        'reaction_info': {}
    }
    
    # Analyze compartments
    print("   Analyzing compartments...")
    for i in range(compartments.size()):
        comp = compartments.get(i)
        comp_name = comp.getObjectName()
        network_analysis['compartment_info'][comp_name] = {
            'volume': comp.getInitialValue(),
            'dimensionality': comp.getDimensionality()
        }
    
    # Analyze species
    print("   Analyzing species...")
    for i in range(metabolites.size()):
        metab = metabolites.get(i)
        species_name = metab.getObjectName()
        compartment = metab.getCompartment().getObjectName()
        
        network_analysis['species_info'][species_name] = {
            'compartment': compartment,
            'initial_concentration': metab.getInitialConcentration(),
            'status': metab.getStatus()
        }
    
    # Analyze reactions
    print("   Analyzing reactions...")
    for i in range(reactions.size()):
        reaction = reactions.get(i)
        reaction_name = reaction.getObjectName()
        
        # Get substrates and products
        substrates = []
        products = []
        
        # Get chemical equation info
        equation = reaction.getChemEq()
        
        network_analysis['reaction_info'][reaction_name] = {
            'reversible': reaction.isReversible(),
            'chemical_equation': equation.getChemEqString(),
            'substrates_count': equation.getSubstrates().size(),
            'products_count': equation.getProducts().size()
        }
    
    print("✅ Network analysis completed")
    return network_analysis

# Run steady state analysis if model exists
if 'copasi_model' in locals() and copasi_model is not None:
    print("🚀 Starting steady state analysis...")
    
    # Run steady state analysis
    steady_state_results = run_steady_state_analysis(copasi_model)
    
    if steady_state_results:
        results['steady_state_data'] = steady_state_results
        
        # Run parameter sensitivity analysis
        sensitivity_results = perform_parameter_sensitivity_analysis(copasi_model)
        results['parameter_analysis'] = sensitivity_results
        
        # Analyze reaction network
        network_analysis = analyze_reaction_network(copasi_model)
        results['network_analysis'] = network_analysis
        
        # Print key findings
        print("\n📊 STEADY STATE ANALYSIS RESULTS:")
        print("="*50)
        
        if 'overall_stability' in steady_state_results:
            print(f"System stability: {steady_state_results['overall_stability']}")
        
        print("\nSpecies at equilibrium:")
        for species, conc in steady_state_results.get('species_concentrations', {}).items():
            print(f"  {species}: {conc:.6e}")
        
        print("\nReaction fluxes:")
        for reaction, flux in steady_state_results.get('reaction_fluxes', {}).items():
            print(f"  {reaction}: {flux:.6e}")
        
        if steady_state_results.get('stability_analysis'):
            print("\nStability eigenvalues:")
            for ev_name, ev_data in steady_state_results['stability_analysis'].items():
                real_part = ev_data['real']
                status = "stable" if ev_data['stable'] else "unstable"
                print(f"  {ev_name}: {real_part:.6e} ({status})")
        
        print("\n📊 NETWORK ANALYSIS:")
        print("="*30)
        if 'network_size' in network_analysis:
            size_info = network_analysis['network_size']
            print(f"Network components:")
            print(f"  Compartments: {size_info['compartments']}")
            print(f"  Species: {size_info['species']}")
            print(f"  Reactions: {size_info['reactions']}")
        
    else:
        print("❌ Steady state analysis failed")
        
else:
    print("❌ No COPASI model available for steady state analysis")


# Cell 5: Data Visualization

def create_time_course_plots(time_course_data: Dict, output_dir: str):
    """Create time course visualization plots"""
    if not time_course_data or 'species_data' not in time_course_data:
        print("⚠️ No time course data to plot")
        return
    
    print("🔄 Creating time course plots...")
    
    time_points = np.array(time_course_data['time'])
    species_data = time_course_data['species_data']
    
    # Create subplots for different categories of species
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define colors for different species
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # Plot 1: All species concentrations over time
    ax1 = axes[0, 0]
    for i, (species_name, concentrations) in enumerate(species_data.items()):
        color = colors[i % len(colors)]
        ax1.plot(time_points, concentrations, label=species_name, color=color, linewidth=2)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Concentration')
    ax1.set_title('All Species Concentrations Over Time')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale plot for better visibility of small concentrations
    ax2 = axes[0, 1]
    for i, (species_name, concentrations) in enumerate(species_data.items()):
        color = colors[i % len(colors)]
        # Add small constant to avoid log(0)
        log_conc = np.array(concentrations) + 1e-10
        ax2.semilogy(time_points, log_conc, label=species_name, color=color, linewidth=2)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Concentration (log scale)')
    ax2.set_title('Species Concentrations (Log Scale)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Focus on key virus-related species
    ax3 = axes[1, 0]
    virus_species = ['Spike_Protein', 'Spike_Trimer', 'Virus_ACE2_Complex', 'Internalized_Virus']
    
    for species_name in virus_species:
        if species_name in species_data:
            ax3.plot(time_points, species_data[species_name], 
                    label=species_name, linewidth=2, marker='o', markersize=3)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Concentration')
    ax3.set_title('Viral Species Dynamics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Phase plot (if we have at least 2 species)
    ax4 = axes[1, 1]
    species_names = list(species_data.keys())
    if len(species_names) >= 2:
        x_species = species_names[0]
        y_species = species_names[1]
        
        ax4.plot(species_data[x_species], species_data[y_species], 
                color='red', linewidth=2, alpha=0.7)
        ax4.scatter(species_data[x_species][0], species_data[y_species][0], 
                   color='green', s=100, label='Start', zorder=5)
        ax4.scatter(species_data[x_species][-1], species_data[y_species][-1], 
                   color='red', s=100, label='End', zorder=5)
        
        ax4.set_xlabel(f'{x_species} Concentration')
        ax4.set_ylabel(f'{y_species} Concentration')
        ax4.set_title(f'Phase Plot: {y_species} vs {x_species}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Phase plot requires\nat least 2 species', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Phase Plot (Not Available)')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, 'time_course_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Time course plots saved: {plot_file}")

def create_steady_state_plots(steady_state_data: Dict, output_dir: str):
    """Create steady state analysis plots"""
    if not steady_state_data:
        print("⚠️ No steady state data to plot")
        return
    
    print("🔄 Creating steady state plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Species concentrations at steady state
    if 'species_concentrations' in steady_state_data:
        species_names = list(steady_state_data['species_concentrations'].keys())
        concentrations = list(steady_state_data['species_concentrations'].values())
        
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(species_names)), concentrations, 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        ax1.set_xlabel('Species')
        ax1.set_ylabel('Steady State Concentration')
        ax1.set_title('Species Concentrations at Steady State')
        ax1.set_xticks(range(len(species_names)))
        ax1.set_xticklabels(species_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, concentrations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Reaction fluxes at steady state
    if 'reaction_fluxes' in steady_state_data:
        reaction_names = list(steady_state_data['reaction_fluxes'].keys())
        fluxes = list(steady_state_data['reaction_fluxes'].values())
        
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(reaction_names)), fluxes, 
                      color=['orange', 'purple', 'brown', 'pink'])
        ax2.set_xlabel('Reactions')
        ax2.set_ylabel('Steady State Flux')
        ax2.set_title('Reaction Fluxes at Steady State')
        ax2.set_xticks(range(len(reaction_names)))
        ax2.set_xticklabels(reaction_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, fluxes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Eigenvalues (stability analysis)
    if 'stability_analysis' in steady_state_data and steady_state_data['stability_analysis']:
        eigenvalues = steady_state_data['stability_analysis']
        ev_names = list(eigenvalues.keys())
        real_parts = [ev['real'] for ev in eigenvalues.values()]
        imag_parts = [ev['imaginary'] for ev in eigenvalues.values()]
        
        ax3 = axes[1, 0]
        
        # Create scatter plot of eigenvalues in complex plane
        colors = ['red' if real < 0 else 'blue' for real in real_parts]
        scatter = ax3.scatter(real_parts, imag_parts, c=colors, s=100, alpha=0.7)
        
        # Add vertical line at Re = 0
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax3.set_xlabel('Real Part')
        ax3.set_ylabel('Imaginary Part')
        ax3.set_title('Eigenvalues (Stability Analysis)')
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        ax3.scatter([], [], c='red', s=100, alpha=0.7, label='Stable (Re < 0)')
        ax3.scatter([], [], c='blue', s=100, alpha=0.7, label='Unstable (Re > 0)')
        ax3.legend()
        
        # Annotate eigenvalues
        for i, (real, imag) in enumerate(zip(real_parts, imag_parts)):
            ax3.annotate(f'λ{i+1}', (real, imag), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
    else:
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, 'Stability analysis\nnot available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Eigenvalues (Not Available)')
    
    # Plot 4: Network summary
    ax4 = axes[1, 1]
    
    # Create a simple network diagram representation
    if 'species_concentrations' in steady_state_data:
        species_list = list(steady_state_data['species_concentrations'].keys())
        n_species = len(species_list)
        
        # Create circular layout for species
        angles = np.linspace(0, 2*np.pi, n_species, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Plot species as circles
        for i, (x, y, species) in enumerate(zip(x_pos, y_pos, species_list)):
            circle = plt.Circle((x, y), 0.1, color=f'C{i}', alpha=0.7)
            ax4.add_patch(circle)
            ax4.text(x, y-0.25, species, ha='center', va='top', fontsize=8)
        
        # Draw connections (simplified - just connect consecutive species)
        for i in range(n_species - 1):
            ax4.plot([x_pos[i], x_pos[i+1]], [y_pos[i], y_pos[i+1]], 
                    'k-', alpha=0.3, linewidth=1)
        
        ax4.set_xlim(-1.5, 1.5)
        ax4.set_ylim(-1.5, 1.5)
        ax4.set_aspect('equal')
        ax4.set_title('Network Topology (Simplified)')
        ax4.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, 'steady_state_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Steady state plots saved: {plot_file}")

def create_summary_dashboard(results: Dict, output_dir: str):
    """Create a summary dashboard with key metrics"""
    print("🔄 Creating summary dashboard...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('COPASI Biochemical Network Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Summary statistics boxes
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    
    summary_text = f"""
    MODEL INFORMATION
    ═══════════════════════════════
    """
    
    if 'model_info' in results:
        model_info = results['model_info']
        summary_text += f"""
    Model: {model_info.get('model_name', 'Unknown')}
    Compartments: {model_info.get('compartments', 0)}
    Species: {model_info.get('species', 0)}
    Reactions: {model_info.get('reactions', 0)}
    """
    
    if 'sequence_info' in results:
        seq_info = results['sequence_info']
        summary_text += f"""
    
    SEQUENCE INFORMATION
    ═══════════════════════════════
    Sequence ID: {seq_info.get('sequence_id', 'Unknown')}
    Protein Length: {seq_info.get('protein_length', 0)} amino acids
    """
    
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Time course summary (if available)
    if 'time_course_analysis' in results:
        ax2 = fig.add_subplot(gs[0, 2:])
        tc_analysis = results['time_course_analysis']
        
        if 'peak_concentrations' in tc_analysis:
            species = list(tc_analysis['peak_concentrations'].keys())[:5]  # Top 5
            peaks = [tc_analysis['peak_concentrations'][s] for s in species]
            
            bars = ax2.bar(range(len(species)), peaks, color='lightcoral')
            ax2.set_title('Peak Concentrations (Time Course)')
            ax2.set_xlabel('Species')
            ax2.set_ylabel('Peak Concentration')
            ax2.set_xticks(range(len(species)))
            ax2.set_xticklabels([s.replace('_', '\n') for s in species], fontsize=8)
            
            for bar, value in zip(bars, peaks):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Steady state summary (if available)
    if 'steady_state_data' in results:
        ax3 = fig.add_subplot(gs[1, :2])
        ss_data = results['steady_state_data']
        
        if 'species_concentrations' in ss_data:
            species = list(ss_data['species_concentrations'].keys())
            concentrations = list(ss_data['species_concentrations'].values())
            
            bars = ax3.bar(range(len(species)), concentrations, color='lightgreen')
            ax3.set_title('Steady State Concentrations')
            ax3.set_xlabel('Species')
            ax3.set_ylabel('Concentration')
            ax3.set_xticks(range(len(species)))
            ax3.set_xticklabels([s.replace('_', '\n') for s in species], fontsize=8)
            
            for bar, value in zip(bars, concentrations):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    # System stability indicator
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis('off')
    
    stability_text = "SYSTEM STABILITY\n═══════════════════════"
    
    if 'steady_state_data' in results and 'overall_stability' in results['steady_state_data']:
        stability = results['steady_state_data']['overall_stability']
        color = 'green' if stability == 'stable' else 'red'
        stability_text += f"\n\nStatus: {stability.upper()}"
        
        if 'stability_analysis' in results['steady_state_data']:
            n_eigenvalues = len(results['steady_state_data']['stability_analysis'])
            stability_text += f"\nEigenvalues analyzed: {n_eigenvalues}"
    else:
        stability_text += "\n\nStatus: Unknown"
        color = 'gray'
    
    ax4.text(0.5, 0.5, stability_text, transform=ax4.transAxes, fontsize=12,
             ha='center', va='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # Analysis timestamp
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    if 'analysis_metadata' in results:
        metadata = results['analysis_metadata']
        timestamp = metadata.get('timestamp', 'Unknown')
        copasi_version = metadata.get('copasi_version', 'Unknown')
        
        footer_text = f"Analysis completed: {timestamp} | COPASI version: {copasi_version}"
        ax5.text(0.5, 0.5, footer_text, transform=ax5.transAxes, fontsize=10,
                ha='center', va='center', style='italic')
    
    # Save the dashboard
    dashboard_file = os.path.join(output_dir, 'copasi_dashboard.png')
    plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Summary dashboard saved: {dashboard_file}")

# Create visualizations if data is available
print("🚀 Starting data visualization...")

if results.get('time_course_data'):
    create_time_course_plots(results['time_course_data'], OUTPUT_DIR)

if results.get('steady_state_data'):
    create_steady_state_plots(results['steady_state_data'], OUTPUT_DIR)

# Always create summary dashboard
create_summary_dashboard(results, OUTPUT_DIR)

print("✅ All visualizations completed!")


# Cell 6: Save Results and Generate Reports

def export_copasi_model(data_model: copasi.CDataModel, output_dir: str):
    """Export COPASI model to various formats"""
    print("🔄 Exporting COPASI model...")
    
    try:
        # Export to COPASI format (.cps)
        cps_file = os.path.join(output_dir, 'spike_protein_model.cps')
        success = data_model.saveModel(cps_file, True)
        if success:
            print(f"✅ COPASI model saved: {cps_file}")
        else:
            print(f"❌ Failed to save COPASI model")
        
        # Export to SBML format
        sbml_file = os.path.join(output_dir, 'spike_protein_model.xml')
        success = data_model.exportSBML(sbml_file, True)
        if success:
            print(f"✅ SBML model saved: {sbml_file}")
        else:
            print(f"❌ Failed to export SBML model")
            
    except Exception as e:
        print(f"❌ Error exporting model: {e}")

def save_time_course_data(time_course_data: Dict, output_dir: str):
    """Save time course data to CSV"""
    if not time_course_data or 'species_data' not in time_course_data:
        print("⚠️ No time course data to save")
        return
    
    print("🔄 Saving time course data...")
    
    # Create DataFrame
    df_data = {'Time': time_course_data['time']}
    df_data.update(time_course_data['species_data'])
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, 'time_course_data.csv')
    df.to_csv(csv_file, index=False)
    print(f"✅ Time course data saved: {csv_file}")
    
    # Save analysis results
    if 'time_course_analysis' in results:
        analysis_data = results['time_course_analysis']
        
        # Create summary DataFrame
        species_names = list(analysis_data.get('peak_concentrations', {}).keys())
        
        analysis_df = pd.DataFrame({
            'Species': species_names,
            'Peak_Concentration': [analysis_data['peak_concentrations'].get(s, 0) for s in species_names],
            'Time_to_Peak': [analysis_data['time_to_peak'].get(s, 0) for s in species_names],
            'Final_Concentration': [analysis_data['final_concentrations'].get(s, 0) for s in species_names],
            'Area_Under_Curve': [analysis_data['area_under_curve'].get(s, 0) for s in species_names],
            'Half_Life': [analysis_data['half_life'].get(s, 'N/A') for s in species_names]
        })
        
        analysis_csv = os.path.join(output_dir, 'time_course_analysis.csv')
        analysis_df.to_csv(analysis_csv, index=False)
        print(f"✅ Time course analysis saved: {analysis_csv}")

def save_steady_state_data(steady_state_data: Dict, output_dir: str):
    """Save steady state data to CSV"""
    if not steady_state_data:
        print("⚠️ No steady state data to save")
        return
    
    print("🔄 Saving steady state data...")
    
    # Save species concentrations
    if 'species_concentrations' in steady_state_data:
        species_df = pd.DataFrame([
            {'Species': species, 'Steady_State_Concentration': conc}
            for species, conc in steady_state_data['species_concentrations'].items()
        ])
        
        species_csv = os.path.join(output_dir, 'steady_state_concentrations.csv')
        species_df.to_csv(species_csv, index=False)
        print(f"✅ Steady state concentrations saved: {species_csv}")
    
    # Save reaction fluxes
    if 'reaction_fluxes' in steady_state_data:
        fluxes_df = pd.DataFrame([
            {'Reaction': reaction, 'Steady_State_Flux': flux}
            for reaction, flux in steady_state_data['reaction_fluxes'].items()
        ])
        
        fluxes_csv = os.path.join(output_dir, 'steady_state_fluxes.csv')
        fluxes_df.to_csv(fluxes_csv, index=False)
        print(f"✅ Steady state fluxes saved: {fluxes_csv}")
    
    # Save eigenvalues
    if 'stability_analysis' in steady_state_data and steady_state_data['stability_analysis']:
        eigenvalues_df = pd.DataFrame([
            {
                'Eigenvalue': ev_name,
                'Real_Part': ev_data['real'],
                'Imaginary_Part': ev_data['imaginary'],
                'Stable': ev_data['stable']
            }
            for ev_name, ev_data in steady_state_data['stability_analysis'].items()
        ])
        
        eigenvalues_csv = os.path.join(output_dir, 'eigenvalues.csv')
        eigenvalues_df.to_csv(eigenvalues_csv, index=False)
        print(f"✅ Eigenvalues saved: {eigenvalues_csv}")

def save_all_results_json(results_dict: Dict, output_dir: str):
    """Save complete results as JSON"""
    print("🔄 Saving complete results...")
    
    # Clean results for JSON serialization
    json_results = {}
    
    for key, value in results_dict.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.floating, np.integer)):
                    json_results[key][k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, np.ndarray):
                    json_results[key][k] = v.tolist()
                else:
                    json_results[key][k] = v
        else:
            json_results[key] = value
    
    # Save JSON file
    json_file = os.path.join(output_dir, 'copasi_analysis_results.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"✅ Complete results saved: {json_file}")

def generate_html_report(results_dict: Dict, output_dir: str):
    """Generate comprehensive HTML report"""
    print("🔄 Generating HTML report...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>COPASI Biochemical Network Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 8px; }}
            .section {{ margin: 20px 0; }}
            .stats {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
            .model-info {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007cba; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .stable {{ color: #28a745; font-weight: bold; }}
            .unstable {{ color: #dc3545; font-weight: bold; }}
            .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 COPASI Biochemical Network Analysis Report</h1>
            <h2>SARS-CoV-2 Spike Protein Interaction Network</h2>
            <p><strong>Analysis Date:</strong> {results_dict['analysis_metadata']['timestamp']}</p>
            <p><strong>COPASI Version:</strong> {results_dict['analysis_metadata']['copasi_version']}</p>
        </div>
    """
    
    # Model Information Section
    if 'model_info' in results_dict:
        model_info = results_dict['model_info']
        html_content += f"""
        <div class="section">
            <h3>📊 Model Information</h3>
            <div class="stats">
                <p><strong>Model Name:</strong> {model_info.get('model_name', 'Unknown')}</p>
                <p><strong>Model Title:</strong> {model_info.get('model_title', 'Unknown')}</p>
                <p><strong>Compartments:</strong> {model_info.get('compartments', 0)}</p>
                <p><strong>Species:</strong> {model_info.get('species', 0)}</p>
                <p><strong>Reactions:</strong> {model_info.get('reactions', 0)}</p>
            </div>
        </div>
        """
    
    # Sequence Information Section
    if 'sequence_info' in results_dict:
        seq_info = results_dict['sequence_info']
        html_content += f"""
        <div class="section">
            <h3>🧬 Sequence Information</h3>
            <div class="stats">
                <p><strong>Sequence ID:</strong> {seq_info.get('sequence_id', 'Unknown')}</p>
                <p><strong>Protein Length:</strong> {seq_info.get('protein_length', 0):,} amino acids</p>
        """
        if 'mrna_length' in seq_info:
            html_content += f"<p><strong>mRNA Length:</strong> {seq_info.get('mrna_length', 0):,} nucleotides</p>"
        html_content += "</div></div>"
    
    # Time Course Results Section
    if 'time_course_analysis' in results_dict:
        tc_analysis = results_dict['time_course_analysis']
        html_content += f"""
        <div class="section">
            <h3>📈 Time Course Analysis Results</h3>
            <div class="stats">
                <p><strong>Simulation Duration:</strong> {results_dict.get('time_course_data', {}).get('metadata', {}).get('duration', 'N/A')} time units</p>
                <p><strong>Time Steps:</strong> {results_dict.get('time_course_data', {}).get('metadata', {}).get('num_steps', 'N/A')}</p>
            </div>
        """
        
        # Peak Concentrations Table
        if 'peak_concentrations' in tc_analysis:
            html_content += """
            <h4>Peak Concentrations</h4>
            <table>
                <tr><th>Species</th><th>Peak Concentration</th><th>Time to Peak</th><th>Final Concentration</th></tr>
            """
            for species in tc_analysis['peak_concentrations'].keys():
                peak = tc_analysis['peak_concentrations'].get(species, 0)
                time_to_peak = tc_analysis.get('time_to_peak', {}).get(species, 0)
                final = tc_analysis.get('final_concentrations', {}).get(species, 0)
                
                html_content += f"""
                <tr>
                    <td><strong>{species}</strong></td>
                    <td>{peak:.6e}</td>
                    <td>{time_to_peak:.2f}</td>
                    <td>{final:.6e}</td>
                </tr>
                """
            html_content += "</table>"
        
        html_content += "</div>"
    
    # Steady State Results Section
    if 'steady_state_data' in results_dict:
        ss_data = results_dict['steady_state_data']
        html_content += f"""
        <div class="section">
            <h3>⚖️ Steady State Analysis Results</h3>
        """
        
        # System Stability
        if 'overall_stability' in ss_data:
            stability = ss_data['overall_stability']
            stability_class = 'stable' if stability == 'stable' else 'unstable'
            html_content += f"""
            <div class="highlight">
                <p><strong>System Stability:</strong> <span class="{stability_class}">{stability.upper()}</span></p>
            </div>
            """
        
        # Species Concentrations Table
        if 'species_concentrations' in ss_data:
            html_content += """
            <h4>Species Concentrations at Steady State</h4>
            <table>
                <tr><th>Species</th><th>Steady State Concentration</th></tr>
            """
            for species, conc in ss_data['species_concentrations'].items():
                html_content += f"""
                <tr>
                    <td><strong>{species}</strong></td>
                    <td>{conc:.6e}</td>
                </tr>
                """
            html_content += "</table>"
        
        # Reaction Fluxes Table
        if 'reaction_fluxes' in ss_data:
            html_content += """
            <h4>Reaction Fluxes at Steady State</h4>
            <table>
                <tr><th>Reaction</th><th>Steady State Flux</th></tr>
            """
            for reaction, flux in ss_data['reaction_fluxes'].items():
                html_content += f"""
                <tr>
                    <td><strong>{reaction}</strong></td>
                    <td>{flux:.6e}</td>
                </tr>
                """
            html_content += "</table>"
        
        # Eigenvalues Table
        if 'stability_analysis' in ss_data and ss_data['stability_analysis']:
            html_content += """
            <h4>Stability Eigenvalues</h4>
            <table>
                <tr><th>Eigenvalue</th><th>Real Part</th><th>Imaginary Part</th><th>Stability</th></tr>
            """
            for ev_name, ev_data in ss_data['stability_analysis'].items():
                stability_class = 'stable' if ev_data['stable'] else 'unstable'
                stability_text = 'Stable' if ev_data['stable'] else 'Unstable'
                
                html_content += f"""
                <tr>
                    <td><strong>{ev_name}</strong></td>
                    <td>{ev_data['real']:.6e}</td>
                    <td>{ev_data['imaginary']:.6e}</td>
                    <td><span class="{stability_class}">{stability_text}</span></td>
                </tr>
                """
            html_content += "</table>"
        
        html_content += "</div>"
    
    # Network Analysis Section
    if 'network_analysis' in results_dict:
        network = results_dict['network_analysis']
        html_content += f"""
        <div class="section">
            <h3>🔗 Network Analysis</h3>
            <div class="stats">
        """
        if 'network_size' in network:
            size_info = network['network_size']
            html_content += f"""
                <p><strong>Network Components:</strong></p>
                <ul>
                    <li>Compartments: {size_info['compartments']}</li>
                    <li>Species: {size_info['species']}</li>
                    <li>Reactions: {size_info['reactions']}</li>
                </ul>
            """
        html_content += "</div></div>"
    
    # Files Generated Section
    html_content += f"""
        <div class="section">
            <h3>📁 Generated Files</h3>
            <ul>
                <li><strong>copasi_analysis_results.json</strong> - Complete analysis results</li>
                <li><strong>time_course_data.csv</strong> - Time course simulation data</li>
                <li><strong>time_course_analysis.csv</strong> - Time course analysis summary</li>
                <li><strong>steady_state_concentrations.csv</strong> - Steady state concentrations</li>
                <li><strong>steady_state_fluxes.csv</strong> - Steady state reaction fluxes</li>
                <li><strong>eigenvalues.csv</strong> - Stability eigenvalues</li>
                <li><strong>spike_protein_model.cps</strong> - COPASI model file</li>
                <li><strong>spike_protein_model.xml</strong> - SBML model file</li>
                <li><strong>time_course_analysis.png</strong> - Time course plots</li>
                <li><strong>steady_state_analysis.png</strong> - Steady state plots</li>
                <li><strong>copasi_dashboard.png</strong> - Summary dashboard</li>
            </ul>
        </div>
        
        <div class="section">
            <h3>📝 Notes</h3>
            <div class="model-info">
                <p><strong>Model Description:</strong> This biochemical network model represents the key interactions in SARS-CoV-2 spike protein processing, including mRNA translation, protein trimerization, receptor binding, and virus internalization.</p>
                <p><strong>Simulation Method:</strong> Deterministic time course simulation using COPASI's built-in ODE solver.</p>
                <p><strong>Steady State Method:</strong> Newton method for finding equilibrium concentrations and stability analysis.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_file = os.path.join(output_dir, 'copasi_analysis_report.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report saved: {html_file}")

def generate_text_summary(results_dict: Dict, output_dir: str):
    """Generate plain text summary report"""
    summary_lines = [
        "="*80,
        "COPASI BIOCHEMICAL NETWORK ANALYSIS SUMMARY",
        "SARS-CoV-2 Spike Protein Interaction Network",
        "="*80,
        f"Analysis Date: {results_dict['analysis_metadata']['timestamp']}",
        f"COPASI Version: {results_dict['analysis_metadata']['copasi_version']}",
        ""
    ]
    
    # Model Information
    if 'model_info' in results_dict:
        model_info = results_dict['model_info']
        summary_lines.extend([
            "MODEL INFORMATION:",
            f"  Model Name: {model_info.get('model_name', 'Unknown')}",
            f"  Compartments: {model_info.get('compartments', 0)}",
            f"  Species: {model_info.get('species', 0)}",
            f"  Reactions: {model_info.get('reactions', 0)}",
            ""
        ])
    
    # Sequence Information
    if 'sequence_info' in results_dict:
        seq_info = results_dict['sequence_info']
        summary_lines.extend([
            "SEQUENCE INFORMATION:",
            f"  Sequence ID: {seq_info.get('sequence_id', 'Unknown')}",
            f"  Protein Length: {seq_info.get('protein_length', 0):,} amino acids",
        ])
        if 'mrna_length' in seq_info:
            summary_lines.append(f"  mRNA Length: {seq_info.get('mrna_length', 0):,} nucleotides")
        summary_lines.append("")
    
    # Time Course Analysis
    if 'time_course_analysis' in results_dict:
        tc_analysis = results_dict['time_course_analysis']
        summary_lines.extend([
            "TIME COURSE ANALYSIS:",
            f"  Simulation Duration: {results_dict.get('time_course_data', {}).get('metadata', {}).get('duration', 'N/A')} time units",
            f"  Time Steps: {results_dict.get('time_course_data', {}).get('metadata', {}).get('num_steps', 'N/A')}",
        ])
        
        if 'peak_concentrations' in tc_analysis:
            summary_lines.append("  Peak Concentrations:")
            for species, peak in tc_analysis['peak_concentrations'].items():
                time_to_peak = tc_analysis.get('time_to_peak', {}).get(species, 0)
                summary_lines.append(f"    {species}: {peak:.6e} (at t={time_to_peak:.2f})")
        summary_lines.append("")
    
    # Steady State Analysis
    if 'steady_state_data' in results_dict:
        ss_data = results_dict['steady_state_data']
        summary_lines.extend([
            "STEADY STATE ANALYSIS:",
        ])
        
        if 'overall_stability' in ss_data:
            summary_lines.append(f"  System Stability: {ss_data['overall_stability'].upper()}")
        
        if 'species_concentrations' in ss_data:
            summary_lines.append("  Species Concentrations:")
            for species, conc in ss_data['species_concentrations'].items():
                summary_lines.append(f"    {species}: {conc:.6e}")
        
        if 'reaction_fluxes' in ss_data:
            summary_lines.append("  Reaction Fluxes:")
            for reaction, flux in ss_data['reaction_fluxes'].items():
                summary_lines.append(f"    {reaction}: {flux:.6e}")
        
        summary_lines.append("")
    
    summary_text = '\n'.join(summary_lines)
    
    # Save text summary
    txt_file = os.path.join(output_dir, 'copasi_analysis_summary.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✅ Text summary saved: {txt_file}")
    return summary_text

# Save all results and generate reports
print("🚀 Saving results and generating reports...")

# Export COPASI model if available
if 'copasi_model' in locals() and copasi_model is not None:
    export_copasi_model(copasi_model, OUTPUT_DIR)

# Save time course data
if results.get('time_course_data'):
    save_time_course_data(results['time_course_data'], OUTPUT_DIR)

# Save steady state data
if results.get('steady_state_data'):
    save_steady_state_data(results['steady_state_data'], OUTPUT_DIR)

# Save complete results as JSON
save_all_results_json(results, OUTPUT_DIR)

# Generate HTML report
generate_html_report(results, OUTPUT_DIR)

# Generate text summary
summary_text = generate_text_summary(results, OUTPUT_DIR)

# Print summary
print("\n" + summary_text)

print(f"\n🎉 COPASI analysis completed successfully!")
print(f"📁 All results saved in: {os.path.abspath(OUTPUT_DIR)}")
print(f"📊 Open 'copasi_analysis_report.html' for detailed interactive report")
print(f"🔬 Model files: spike_protein_model.cps (COPASI) and spike_protein_model.xml (SBML)")