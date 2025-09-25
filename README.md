# 🧬 Spiralography

<div align="center">

**AI-Powered Bioinformatics Pipeline for Advanced Sequence Optimization**

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Models](https://img.shields.io/badge/AI-DistilGPT2%20%7C%20GPT--Neo%20%7C%20DialoGPT-orange)

</div>

## 🔬 Overview

<div align="justify">

**Spiralography** is a comprehensive bioinformatics pipeline that harnesses multiple AI language models to optimize biological sequences through sophisticated computational workflows. The system integrates over 20 specialized bioinformatics tools with machine learning models to deliver precision sequence analysis and optimization.

The pipeline combines traditional bioinformatics approaches with modern AI capabilities, enabling researchers to perform complex sequence manipulations, structural predictions, and optimization tasks. Built on robust foundations including Ensembl, DIAMOND, InterProScan, and Rfam databases, Spiralography provides a unified platform for advanced genomic analysis with automated AI-guided decision making throughout the analytical process.

</div>

## ✨ Key Features

<div align="justify">

The system leverages three primary AI models for intelligent sequence analysis: DistilGPT2 for code generation and genomic analysis, GPT-Neo-125M for specialized reasoning and optimization tasks, and DialoGPT for biological reasoning and molecular dynamics analysis. This multi-model approach ensures comprehensive coverage of different analytical perspectives while maintaining high computational efficiency.

</div>

- **Multi-Model AI Integration** - Three specialized language models working in orchestrated sequence
- **Comprehensive Tool Suite** - 20+ integrated bioinformatics tools from clustering to molecular dynamics
- **Advanced Optimization** - Intelligent codon usage, GC content balancing, and structure refinement
- **Real-time Analysis** - Interactive Jupyter notebook with live visualization and progress tracking
- **Publication-Ready Output** - High-resolution plots, detailed reports, and multiple export formats

## 🏗️ Architecture & Pipeline Flow

<div align="justify">

The pipeline operates through a systematic 20-stage execution sequence where each AI agent acts as an expert bioinformatics tool replacement. The workflow begins with input sequence loading and progresses through increasingly sophisticated analytical stages, with each agent receiving structured input from the previous stage and generating specialized output through AI-guided analysis.

**Stage 1-5: Foundation Analysis**
The pipeline initiates with **Input Sequence Loading** from FASTA files (typically SARS-CoV-2 spike mRNA sequences), followed by **Ensembl Agent** performing genomic annotation and gene model prediction. **Biopython Agent** handles sequence parsing and manipulation, converting between formats and performing basic sequence operations. **CD-HIT Agent** performs sequence clustering and redundancy removal to identify representative sequences. **DIAMOND Agent** conducts fast protein sequence alignment against viral protein databases, identifying homologous sequences and functional relationships.

**Stage 6-10: Functional Characterization**
**InterProScan Agent** analyzes protein sequences for functional domains and GO term assignments. **Rfam Agent** performs RNA family classification and secondary structure prediction using covariance models. **ViennaRNA Agent** calculates minimum free energy structures and base-pairing probabilities. **MRNAID Agent** optimizes mRNA sequences for enhanced translation efficiency and reduced immunogenicity. **oxDNA Agent** performs molecular dynamics simulations with detailed trajectory analysis including RMSD, radius of gyration, and energy landscapes.

**Stage 11-15: Advanced Optimization**
**COOL Agent** conducts RNA structure optimization with thermodynamic constraints. **KineFold Agent** analyzes RNA folding kinetics and pathway intermediates. **COPASI Agent** performs biochemical network modeling and metabolic flux analysis. **CRISPOR Agent** designs CRISPR guide RNAs with specificity scoring and off-target analysis. **IEDB Agent** predicts epitopes and evaluates immunogenicity potential across MHC-I, MHC-II, and B-cell epitopes.

**Stage 16-20: Design Integration**
**RBS Calculator Agent** optimizes ribosome binding sites for translation initiation. **DNA Chisel Agent** performs comprehensive sequence optimization with multiple constraints. **UniProt Agent** provides protein database searches and functional annotation. **AlphaFold Agent** predicts protein 3D structures and evaluates folding confidence. **Benchling Agent** creates final sequence designs with plasmid construction and cloning strategies.

The system maintains persistent pipeline data storage with step tracking (`pipeline_data['step']`), current tool identification (`pipeline_data['current_tool']`), and comprehensive metadata management throughout the execution flow.

</div>

## 📁 Project Structure

<div align="justify">

The project is organized into a hierarchical structure that separates input data, processing logic, and tool-specific outputs. The main pipeline notebook orchestrates all analyses while individual output directories maintain results from each specialized tool.

</div>

```
spiralography/
├── assets/                          # Input datasets and reference files
│   └── SarsCov2SpikemRNA.fasta     # SARS-CoV-2 spike protein sequences (2 sequences)
├── pipeline_outputs/                # Comprehensive tool outputs
│   ├── benchling/                   # Sequence design and plasmid annotation
│   ├── biopython/                   # Parsed sequences and translations
│   ├── cdhit/                       # Clustered representative sequences
│   ├── copasi/                      # Biochemical network models (SBML format)
│   ├── crispor/                     # CRISPR guide designs and specificity scores
│   ├── diamond/                     # Protein alignments (BLAST tabular format)
│   ├── dnachisel/                   # Optimized sequences with constraint reports
│   ├── ensembl/                     # Genomic annotations and gene models
│   ├── exported_data/               # Final optimized sequences and metadata
│   ├── final_analysis/              # Best sequence analysis and visualizations
│   ├── iedb/                        # Epitope predictions and immunogenicity
│   ├── integrated_results/          # Cross-tool analysis summaries
│   ├── interproscan/                # Protein domains and functional annotations
│   ├── kinefold/                    # RNA folding pathways and kinetics
│   ├── llamaaffinity/               # Binding affinity predictions
│   ├── mrnaid/                      # mRNA optimization reports
│   ├── oxdna/                       # MD trajectories and energy analysis
│   ├── rbs_calculator/              # Ribosome binding site optimization
│   ├── rfam/                        # RNA family classifications
│   ├── validation_reports/          # Quality control and validation
│   └── viennarna/                   # RNA secondary structure predictions
├── pipeline.ipynb                   # Main analysis notebook (20+ cells)
├── requirements.txt                 # Python dependencies
└── .venv/                          # Virtual environment isolation
```

<div align="justify">

Each output directory contains tool-specific results: `diamond/` stores protein alignments in BLAST tabular format with e-values and bit scores; `oxdna/` contains molecular dynamics trajectories in XYZ format with energy analysis; `copasi/` maintains SBML biochemical network models with time-course simulations; `benchling/` produces designed sequences with plasmid maps and cloning strategies.

</div>

## 🚀 Quick Start

<div align="justify">

Installation requires Python 3.8+ with substantial computational resources. GPU acceleration is strongly recommended for AI model inference, while 16GB+ RAM ensures smooth processing of large sequence datasets and molecular dynamics simulations.

</div>

### Installation & Setup

```bash
# Clone repository and navigate
git clone <repository-url>
cd spiralography

# Create isolated virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install comprehensive dependencies
pip install -r requirements.txt

# Verify GPU availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Execution Methods

<div align="justify">

The pipeline offers multiple execution approaches depending on user needs. The primary interface through Jupyter notebook provides interactive analysis with real-time visualization and step-by-step control. Alternative programmatic execution enables batch processing and automated workflows.

</div>

```bash
# Interactive notebook execution (recommended)
jupyter notebook pipeline.ipynb

# Programmatic batch execution
python -c "
import sys
sys.path.append('.')
from pipeline import analyze_sequences
results = analyze_sequences('assets/SarsCov2SpikemRNA.fasta')
print(f'Analysis complete: {len(results)} sequences processed')
"

# Custom analysis with specific parameters
python -c "
from pipeline import generate_llm_response, distilgpt2_model, distilgpt2_tokenizer
response = generate_llm_response(
    model=distilgpt2_model, 
    tokenizer=distilgpt2_tokenizer,
    prompt='Analyze this sequence: ATGCGATCG',
    max_length=200,
    temperature=0.5
)
print(response)
"
```

## 🔧 Configuration Parameters

<div align="justify">

The pipeline employs sophisticated parameter management across AI models, simulation engines, and bioinformatics tools. Configuration is handled automatically but can be customized for specific research requirements.

</div>

### AI Model Configuration

<div align="justify">

**Language Model Parameters** control the behavior and output quality of the three primary AI agents. The `max_length` parameter defines token limits for model responses, typically set between 400-1000 tokens depending on the complexity of the biological analysis required. The `temperature` parameter (default 0.7) balances response creativity and determinism—lower values (0.1-0.5) produce more consistent, focused outputs while higher values (0.8-1.2) increase diversity but may reduce accuracy.

</div>

```python
# Core AI model parameters in generate_llm_response()
max_length = 500          # Token limit for responses (400-1000 typical)
temperature = 0.7         # Creativity vs consistency balance (0.1-1.2)
do_sample = True          # Enable stochastic generation
num_return_sequences = 1  # Number of alternative outputs
torch_dtype = torch.float16  # Memory-efficient mixed precision
device_map = "auto"       # Automatic GPU/CPU allocation
pad_token_id = tokenizer.eos_token_id  # Consistent sequence handling
```

### Molecular Simulation Parameters

<div align="justify">

**oxDNA Molecular Dynamics** simulations employ comprehensive parameter sets for accurate nucleic acid modeling. Temperature is maintained at 300K (physiological conditions), while salt concentration at 0.15M mimics cellular ionic strength. Simulation time extends to 1000ns with 0.002ps timesteps, generating 500,000 total steps with data sampling every 100 steps for 5,000 trajectory frames.

</div>

```python
# Molecular dynamics simulation parameters
simulation_params = {
    'temperature': 300,           # Kelvin (physiological conditions)
    'salt_concentration': 0.15,   # Molarity (cellular ionic strength)
    'simulation_time': 1000,      # Nanoseconds (trajectory length)
    'timestep': 0.002,           # Picoseconds (integration precision)
    'total_steps': 500000,       # Total MD steps
    'sampling_frequency': 100,    # Data collection interval
    'trajectory_frames': 5000     # Frames stored for analysis
}
```

### Biochemical Network Parameters

<div align="justify">

**COPASI Simulations** model biochemical networks with default species concentrations at 10.0μM in 1.0L volumes. Time-course simulations span 100 seconds with 1000 time steps, providing detailed kinetic analysis. Temperature is set to 310.15K (37°C) for physiological relevance.

</div>

```python
# Biochemical network modeling parameters
DEFAULT_CONCENTRATION = 10.0     # μM (species concentration)
DEFAULT_VOLUME = 1.0            # Liters (reaction volume)
SIMULATION_TIME = 100.0         # Seconds (kinetic analysis duration)
TIME_STEPS = 1000              # Temporal resolution
temperature = 310.15           # Kelvin (37°C physiological)
```

### Sequence Analysis Parameters

<div align="justify">

**Quality Scoring Weights** determine multi-dimensional sequence evaluation criteria. GC content analysis receives 20% weight, sequence length 10%, structural complexity 20%, repeat content 20%, secondary structure prediction 15%, optimization metrics 10%, and expression efficiency 5%. These weights can be adjusted based on specific research priorities.

</div>

```python
# Quality scoring weight distribution
scoring_weights = {
    'gc_content': 0.2,        # GC composition analysis
    'length': 0.1,            # Sequence length consideration
    'complexity': 0.2,        # Structural complexity assessment
    'repeats': 0.2,           # Repetitive element evaluation
    'structure': 0.15,        # Secondary structure prediction
    'optimization': 0.1,      # Codon usage optimization
    'efficiency': 0.05        # Translation efficiency
}
```

### Input Processing Parameters

<div align="justify">

**Tokenization Settings** handle variable-length biological sequences with `max_length=1024` for input truncation and dynamic padding using end-of-sequence tokens. Attention masks are automatically generated to handle sequence boundaries, while `truncation=True` ensures consistent input dimensions across diverse sequence lengths.

</div>

```python
# Input tokenization and processing
inputs = tokenizer.encode(
    prompt, 
    return_tensors="pt", 
    max_length=1024,          # Maximum input length
    truncation=True           # Handle long sequences
)
```

## 📊 Output Analysis & Formats

<div align="justify">

Spiralography generates comprehensive results across multiple formats optimized for different analysis requirements. The system produces both raw computational data and publication-ready visualizations, ensuring compatibility with downstream analysis workflows and research publication standards.

**Primary Output Formats** include optimized sequences in standard FASTA format with detailed headers containing metadata and quality scores. JSON files provide structured data with comprehensive scoring information, optimization histories, and cross-references between analysis stages. High-resolution PNG/SVG visualizations are generated at 300 DPI resolution with professional color schemes and clear annotations suitable for publication.

**Specialized Formats** vary by tool: BLAST tabular format for alignment results with e-values and bit scores; SBML format for biochemical network models compatible with systems biology tools; XYZ trajectory files for molecular dynamics analysis; GFF3 format for genomic annotations; TSV files for statistical analysis and data import.

**Analysis Reports** combine quantitative metrics with qualitative assessments. Best sequence analysis includes comprehensive scoring across multiple dimensions: GC content analysis (optimal range 45-55%), secondary structure likelihood, codon usage optimization, translation efficiency predictions, and immunogenicity assessments. Quality control reports highlight potential issues such as repetitive sequences, restriction enzyme sites, and structural instabilities.

</div>

## 🎯 Research Applications

<div align="justify">

The system excels across diverse research domains requiring sophisticated sequence analysis and optimization. **Viral Research** applications include vaccine development through epitope prediction and immunogenicity optimization, therapeutic design for antiviral compounds, and evolutionary analysis of viral sequence variants. **Protein Engineering** capabilities encompass expression optimization through codon usage analysis, structural stability prediction via molecular dynamics, and functional domain identification through comprehensive database searches.

**Synthetic Biology** applications leverage the pipeline's design capabilities for synthetic gene construction, metabolic pathway optimization, and regulatory element design. **Pharmaceutical Development** benefits from epitope mapping for vaccine targets, immunogenicity assessment for therapeutic proteins, and molecular dynamics analysis for drug-target interactions.

**Diagnostic Development** utilizes CRISPR guide design capabilities for genome editing applications, PCR primer optimization for diagnostic assays, and biomarker identification through comparative genomics analysis. The flexible architecture accommodates both targeted studies focusing on specific sequences and large-scale genomic surveys requiring high-throughput analysis.

</div>

## 🤝 Contributing

<div align="justify">

We welcome contributions from the bioinformatics, computational biology, and AI communities. The project follows standard open source development practices with emphasis on code quality, comprehensive testing, and detailed documentation.

</div>

**Development Guidelines:**
- Fork repository and create feature branches for all changes
- Maintain compatibility with Python 3.8+ and current dependency versions
- Include comprehensive tests for new functionality
- Update documentation for API changes and new features
- Follow PEP 8 style guidelines with 88-character line limits

**Testing Requirements:**
- Unit tests for individual functions and methods
- Integration tests for pipeline stage interactions
- Performance benchmarks for computationally intensive operations
- Validation tests using known reference datasets

## 📄 License

<div align="justify">

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for complete terms and conditions. The MIT license permits commercial use, modification, distribution, and private use while requiring only attribution and license preservation.

</div>

---

<div align="center">

**🧬 Mapping the complexity of biological sequences through AI-powered analysis**

</div>