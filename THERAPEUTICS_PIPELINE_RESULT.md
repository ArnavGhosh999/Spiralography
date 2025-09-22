# THERAPEUTICS PIPELINE RESULT DOCX

## 8. COOL (RNA Secondary Structure Prediction)

**Input:** RNA sequences (FASTA/RAW/JSON/JSONL)
**Output:** Secondary structure predictions (dot-bracket notation, MFE values, JSON)
**Alternative used:** ViennaRNA Package (RNA.fold)
**Libraries Used:**
- ViennaRNA (v2.7.0) - RNA secondary structure prediction
- ipywidgets (v8.1.5) - Interactive file upload interface
- JSON - Data parsing and output formatting

**Parameters and Approach:**
- **Prediction Method:** ViennaRNA RNA.fold() function
- **File Format Support:** .fna, .fana, .fa, .fasta, .json, .jsonl
- **Sequence Processing:** 
  - Automatic DNA to RNA conversion (T → U)
  - Uppercase normalization
  - Whitespace and tab removal
- **Output Format:** Dot-bracket notation for secondary structure
- **Energy Calculation:** Minimum Free Energy (MFE) in kcal/mol
- **Batch Processing:** Multiple sequences processed simultaneously

**COOL Analysis Parameters:**
- **Sequence Validation:** AUGC nucleotide validation
- **Error Handling:** Graceful error handling for invalid sequences
- **Interactive Interface:** File upload widget with real-time processing
- **Multi-format Support:** FASTA, JSON, and JSONL input parsing
- **Results Display:** Formatted output with sequence ID, sequence, structure, and MFE

**OUTPUTS:**
- **JSON output:** cool_analysis_results.json
- **Structure files:** Dot-bracket notation (.txt)
- **Energy files:** MFE values (.txt)
- **Interactive results:** Real-time display in Jupyter notebook

---

## 9. DNAChisel (DNA Sequence Optimization)

**Input:** DNA sequences (FASTA/JSON/JSONL)
**Output:** Optimized DNA sequences (FASTA, JSON, optimization metrics)
**Alternative used:** DnaOptimizationProblem from dnachisel library
**Libraries Used:**
- dnachisel - DNA sequence optimization framework
- ipywidgets - Interactive sequence selection interface
- JSON - Data parsing and output formatting

**Parameters and Approach:**
- **Optimization Constraints:**
  - AvoidPattern("GGTCTC") - Avoid BsaI restriction sites
  - AvoidPattern("GAGACC") - Avoid BsmBI restriction sites
  - EnforceGCContent(mini=0.4, maxi=0.6) - Maintain 40-60% GC content
- **Optimization Objectives:**
  - EnforceTranslation() - Maintain coding sequence integrity
  - MaximizeCAI(species="e_coli") - Optimize codon usage for E. coli
- **Sequence Validation:** 
  - Length validation (must be divisible by 3 for coding sequences)
  - Start codon validation (must begin with ATG)
- **Optimization Method:** Constraint resolution followed by objective optimization

**DNAChisel Optimization Parameters:**
- **Constraint Resolution:** Automatic constraint satisfaction
- **Objective Optimization:** Codon Adaptation Index (CAI) maximization
- **Species-specific Optimization:** E. coli codon usage preferences
- **Restriction Site Avoidance:** Common cloning enzyme sites
- **GC Content Control:** Maintains optimal GC content range
- **Similarity Calculation:** Sequence similarity percentage post-optimization

**OUTPUTS:**
- **JSON output:** dnachisel_optimization_results.json
- **FASTA output:** Optimized sequences (.fasta)
- **Metrics files:** Similarity scores, GC content changes (.txt)
- **Optimization reports:** Detailed optimization summaries (.txt)

---

## 10. Kinefold_Substitute (AI-Powered RNA Folding Kinetics)

**Input:** RNA sequences (FASTA/JSON/JSONL)
**Output:** AI-predicted folding kinetics and pathways (text analysis, JSON)
**Alternative used:** BioMistral-7B-Bio-v0.1 (biomistral/biomistral-7b-bio-v0.1)
**Libraries Used:**
- transformers - Hugging Face model loading and text generation
- torch - PyTorch backend for model inference
- ipywidgets - Interactive file upload interface
- JSON - Data parsing and output formatting

**Parameters and Approach:**
- **AI Model:** biomistral/biomistral-7b-bio-v0.1
- **Model Configuration:**
  - Precision: torch.float16 (GPU) / torch.float32 (CPU)
  - Device mapping: Automatic GPU/CPU detection
  - Max new tokens: 250 for detailed analysis
- **Task Instruction:** Specialized prompt for RNA folding kinetics prediction
- **Sequence Processing:** 
  - Multi-format input support (FASTA, JSON, JSONL)
  - Automatic sequence extraction and validation
- **Analysis Focus:**
  - Folding intermediates identification
  - Key transition steps prediction
  - Folding pathway analysis
  - Kinetic model application

**Kinefold_Substitute Analysis Parameters:**
- **Model Loading:** AutoTokenizer and AutoModelForCausalLM
- **Pipeline Configuration:** Text generation with specialized bio prompts
- **Device Optimization:** CUDA acceleration when available
- **Batch Processing:** Multiple sequence analysis support
- **Output Formatting:** Structured text analysis with biological insights
- **Error Handling:** Graceful fallback for model loading issues

**OUTPUTS:**
- **JSON output:** kinefold_analysis_results.json
- **Text analysis:** Detailed folding kinetics predictions (.txt)
- **Pathway files:** Folding pathway descriptions (.txt)
- **Interactive results:** Real-time AI analysis display

---

## Summary of Therapeutics Pipeline Tools

### Input Format Support
All three tools support multiple input formats:
- **FASTA files:** .fa, .fna, .fas, .fana, .fasta
- **JSON files:** Structured sequence data
- **JSONL files:** Line-delimited JSON records

### Output Format Consistency
All tools generate standardized outputs:
- **JSON results:** Machine-readable analysis results
- **Text reports:** Human-readable analysis summaries
- **Interactive displays:** Real-time processing in Jupyter notebooks

### Computational Approaches
1. **COOL:** Physics-based RNA secondary structure prediction using ViennaRNA
2. **DNAChisel:** Constraint-based DNA sequence optimization with biological constraints
3. **Kinefold_Substitute:** AI-powered RNA folding kinetics using large language models

### Integration Capabilities
- **Sequential Processing:** Outputs from one tool can serve as inputs to others
- **Batch Processing:** Multiple sequences processed simultaneously
- **Format Conversion:** Automatic format detection and conversion
- **Error Handling:** Robust error handling across all tools

### Biological Applications
- **RNA Therapeutics:** Secondary structure prediction and folding analysis
- **DNA Engineering:** Sequence optimization for expression and cloning
- **Synthetic Biology:** Integrated design and analysis pipeline
- **Drug Development:** Multi-scale analysis from sequence to structure to kinetics
