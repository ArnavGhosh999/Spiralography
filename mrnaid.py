import json
from Bio import SeqIO
import argparse

# Example codon usage table for E. coli (simplified for demo)
codon_table = {
    "A": ["GCT","GCC","GCA","GCG"],
    "C": ["TGT","TGC"],
    "D": ["GAT","GAC"],
    "E": ["GAA","GAG"],
    "F": ["TTT","TTC"],
    "G": ["GGT","GGC","GGA","GGG"],
    "H": ["CAT","CAC"],
    "I": ["ATT","ATC","ATA"],
    "K": ["AAA","AAG"],
    "L": ["TTA","TTG","CTT","CTC","CTA","CTG"],
    "M": ["ATG"],
    "N": ["AAT","AAC"],
    "P": ["CCT","CCC","CCA","CCG"],
    "Q": ["CAA","CAG"],
    "R": ["CGT","CGC","CGA","CGG","AGA","AGG"],
    "S": ["TCT","TCC","TCA","TCG","AGT","AGC"],
    "T": ["ACT","ACC","ACA","ACG"],
    "V": ["GTT","GTC","GTA","GTG"],
    "W": ["TGG"],
    "Y": ["TAT","TAC"]
}

# Simple codon to amino acid mapping (DNA)
aa_table = {
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TGT":"C","TGC":"C",
    "GAT":"D","GAC":"D",
    "GAA":"E","GAG":"E",
    "TTT":"F","TTC":"F",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    "CAT":"H","CAC":"H",
    "ATT":"I","ATC":"I","ATA":"I",
    "AAA":"K","AAG":"K",
    "TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATG":"M",
    "AAT":"N","AAC":"N",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAA":"Q","CAG":"Q",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TGG":"W",
    "TAT":"Y","TAC":"Y"
}

def codon_to_aa(codon):
    return aa_table.get(codon, "X")  # "X" = unknown codons

# Function to optimize codons
def optimize_codon(seq):
    seq = seq.upper()
    optimized = ""
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        if len(codon) < 3:
            break
        aa = codon_to_aa(codon)
        optimized += codon_table.get(aa, [codon])[0]
    return optimized

# Function to calculate GC content
def gc_content(seq):
    seq = seq.upper()
    g = seq.count("G")
    c = seq.count("C")
    return round((g + c) / len(seq) * 100, 2) if len(seq) > 0 else 0


def process_file(input_file):
    json_data = []

    if input_file.endswith((".fasta", ".fa")):
        print("Detected FASTA format")
        records = list(SeqIO.parse(input_file, "fasta"))
        for r in records:
            optimized = optimize_codon(str(r.seq))
            json_data.append({
                "id": r.id,
                "description": r.description,
                "original_sequence": str(r.seq),
                "optimized_sequence": optimized,
                "gc_content_original": gc_content(str(r.seq)),
                "gc_content_optimized": gc_content(optimized)
            })

    elif input_file.endswith(".json"):
        print("Detected JSON format")
        with open(input_file) as f:
            raw_data = json.load(f)
            for entry in raw_data:
                seq = entry.get("sequence", "")
                optimized = optimize_codon(seq) if seq else ""
                json_data.append({
                    "id": entry.get("id", ""),
                    "description": entry.get("description", ""),
                    "original_sequence": seq,
                    "optimized_sequence": optimized,
                    "gc_content_original": gc_content(seq),
                    "gc_content_optimized": gc_content(optimized)
                })

    elif input_file.endswith(".jsonl"):
        print("Detected JSONL format")
        with open(input_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                seq = entry.get("sequence", "")
                optimized = optimize_codon(seq) if seq else ""
                json_data.append({
                    "id": entry.get("id", ""),
                    "description": entry.get("description", ""),
                    "original_sequence": seq,
                    "optimized_sequence": optimized,
                    "gc_content_original": gc_content(seq),
                    "gc_content_optimized": gc_content(optimized)
                })
    else:
        raise ValueError("Unsupported file type. Please provide FASTA, JSON, or JSONL.")

    return json_data


def main():
    parser = argparse.ArgumentParser(description="Optimize codon usage in mRNA sequences")
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument("--output", "-o", help="Output file (default: optimized_mrna.jsonl)")
    args = parser.parse_args()

    input_file = args.input_file
    print("Uploaded file:", input_file)

    json_data = process_file(input_file)

    print("Total records processed:", len(json_data))
    if json_data:
        print("\nFirst Record Preview:")
        print(json.dumps(json_data[0], indent=4)[:500])

    # Use provided output file or default
    output_file = args.output or "optimized_mrna.jsonl"
    output_json = output_file.replace('.jsonl', '.json')
    
    # Save JSON
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=4)

    # Save JSONL
    with open(output_file, "w") as f:
        for entry in json_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nCreated: {output_json} and {output_file}")


if __name__ == "__main__":
    main()
