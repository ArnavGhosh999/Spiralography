import json
from Bio import SeqIO
import argparse

# Function to find potential SpCas9 guides (20nt upstream of NGG PAM)
def find_guides(seq):
    seq = seq.upper()
    guides = []
    for i in range(len(seq) - 23):
        candidate = seq[i:i+23]  # 20nt guide + NGG PAM
        if candidate[20:23] == "GG":  # NGG PAM
            guide_seq = candidate[:20]
            gc_content = (guide_seq.count("G") + guide_seq.count("C")) / 20
            efficiency = round(gc_content * 100, 2)  # simple GC-based score
            guides.append({
                "position": i+1,  # 1-based position
                "guide_sequence": guide_seq,
                "pam": "NGG",
                "gc_content": round(gc_content*100, 2),
                "efficiency_score": efficiency
            })
    return guides


def process_file(input_file):
    json_data = []

    if input_file.endswith((".fasta", ".fa")):
        print("Detected FASTA format")
        records = list(SeqIO.parse(input_file, "fasta"))
        for r in records:
            guides = find_guides(str(r.seq))
            json_data.append({
                "id": r.id,
                "description": r.description,
                "sequence": str(r.seq),
                "guides": guides
            })

    elif input_file.endswith(".json"):
        print("Detected JSON format")
        with open(input_file) as f:
            raw_data = json.load(f)
            for entry in raw_data:
                seq = entry.get("sequence", "")
                guides = find_guides(seq) if seq else []
                json_data.append({
                    "id": entry.get("id", ""),
                    "description": entry.get("description", ""),
                    "sequence": seq,
                    "guides": guides
                })

    elif input_file.endswith(".jsonl"):
        print("Detected JSONL format")
        with open(input_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                seq = entry.get("sequence", "")
                guides = find_guides(seq) if seq else []
                json_data.append({
                    "id": entry.get("id", ""),
                    "description": entry.get("description", ""),
                    "sequence": seq,
                    "guides": guides
                })

    else:
        raise ValueError("Unsupported file type. Please provide FASTA, JSON, or JSONL.")

    return json_data


def main():
    parser = argparse.ArgumentParser(description="Find SpCas9 CRISPR guides from FASTA, JSON, or JSONL")
    parser.add_argument("input_file", help="Path to input file")
    args = parser.parse_args()

    input_file = args.input_file
    print("Uploaded file:", input_file)

    json_data = process_file(input_file)

    print("Total records parsed:", len(json_data))
    if json_data:
        print("\nFirst Record Preview:")
        print(json.dumps(json_data[0], indent=4)[:500])

    # Save JSON
    with open("crispr_guides.json", "w") as f:
        json.dump(json_data, f, indent=4)

    # Save JSONL
    with open("crispr_guides.jsonl", "w") as f:
        for entry in json_data:
            f.write(json.dumps(entry) + "\n")

    print("\nCreated: crispr_guides.json and crispr_guides.jsonl")


if __name__ == "__main__":
    main()
