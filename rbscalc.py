import json
import re
from Bio import SeqIO
import argparse


# Function to score RBS and suggest optimized RBS
def rbs_optimize(seq):
    """
    Estimate RBS strength and generate an optimized RBS sequence.
    Returns:
        translation_initiation_rate (numeric)
        optimized_rbs_sequence (string)
    """
    seq = seq.upper()
    sd_motif = "AGGAGG"

    # Find SD motif positions
    matches = [m.start() for m in re.finditer(sd_motif, seq)]

    # Default values if no motif found
    if not matches:
        tir = 0
        # Add canonical SD motif 8nt upstream of start codon (assuming seq starts at start codon)
        optimized_rbs = sd_motif + seq
    else:
        # Strongest motif = closest to start codon (position 0)
        closest = min(matches)
        # Simple scoring: 100 - distance to start codon
        tir = max(0, 100 - closest)
        # Optimized RBS: move canonical motif to 8nt upstream of start codon
        optimized_rbs = seq
        # Replace existing motif with canonical at optimal position
        if closest != 0:
            optimized_rbs = sd_motif + seq[closest + len(sd_motif):]

    return tir, optimized_rbs


def process_file(input_file):
    json_data = []

    if input_file.endswith((".fasta", ".fa")):
        print("Detected FASTA format")
        records = list(SeqIO.parse(input_file, "fasta"))
        for r in records:
            tir, optimized_rbs = rbs_optimize(str(r.seq))
            json_data.append({
                "id": r.id,
                "description": r.description,
                "original_sequence": str(r.seq),
                "translation_initiation_rate": tir,
                "optimized_rbs_sequence": optimized_rbs
            })

    elif input_file.endswith(".json"):
        print("Detected JSON format")
        with open(input_file) as f:
            raw_data = json.load(f)
            for entry in raw_data:
                seq = entry.get("sequence", "")
                tir, optimized_rbs = rbs_optimize(seq) if seq else (0, "")
                json_data.append({
                    "id": entry.get("id", ""),
                    "description": entry.get("description", ""),
                    "original_sequence": seq,
                    "translation_initiation_rate": tir,
                    "optimized_rbs_sequence": optimized_rbs
                })

    elif input_file.endswith(".jsonl"):
        print("Detected JSONL format")
        with open(input_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                seq = entry.get("sequence", "")
                tir, optimized_rbs = rbs_optimize(seq) if seq else (0, "")
                json_data.append({
                    "id": entry.get("id", ""),
                    "description": entry.get("description", ""),
                    "original_sequence": seq,
                    "translation_initiation_rate": tir,
                    "optimized_rbs_sequence": optimized_rbs
                })

    else:
        raise ValueError("Unsupported file type. Please provide FASTA, JSON, or JSONL.")

    return json_data


def main():
    parser = argparse.ArgumentParser(description="Optimize RBS sequences from FASTA, JSON, or JSONL")
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument("--output", "-o", help="Output file (default: rbs_output.jsonl)")
    args = parser.parse_args()

    input_file = args.input_file
    print("Uploaded file:", input_file)

    json_data = process_file(input_file)

    print("Total records parsed:", len(json_data))
    if json_data:
        print("\nFirst Record Preview:")
        print(json.dumps(json_data[0], indent=4)[:500])

    # Use provided output file or default
    output_file = args.output or "rbs_output.jsonl"
    output_json = output_file.replace('.jsonl', '.json')
    
    # Save results
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=4)
    with open(output_file, "w") as f:
        for entry in json_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nCreated: {output_json} and {output_file}")


if __name__ == "__main__":
    main()
