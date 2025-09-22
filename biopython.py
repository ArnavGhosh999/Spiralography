import json
import os
import zipfile
from Bio import SeqIO
import argparse

def process_file(input_file):
    records = []
    json_data = []
    processed_file_name = None

    if input_file.endswith(".zip"):
        print("Detected ZIP format. Attempting to extract FASTA.")
        try:
            with zipfile.ZipFile(input_file, 'r') as zip_ref:
                fasta_file_in_zip = None
                for name in zip_ref.namelist():
                    if name.endswith((".fasta", ".fa")):
                        fasta_file_in_zip = name
                        break

                if fasta_file_in_zip:
                    print(f"Found FASTA file within zip: {fasta_file_in_zip}")
                    zip_ref.extract(fasta_file_in_zip)
                    processed_file_name = fasta_file_in_zip

                    print("Detected FASTA format")
                    records = list(SeqIO.parse(processed_file_name, "fasta"))
                    json_data = [
                        {"id": r.id, "description": r.description, "sequence": str(r.seq)}
                        for r in records
                    ]
                else:
                    raise ValueError("No FASTA or .fa file found within the zip archive.")
        except zipfile.BadZipFile:
            raise ValueError("Uploaded file is not a valid ZIP file.")

    elif input_file.endswith((".fasta", ".fa")):
        print("Detected FASTA format")
        records = list(SeqIO.parse(input_file, "fasta"))
        json_data = [
            {"id": r.id, "description": r.description, "sequence": str(r.seq)}
            for r in records
        ]

    elif input_file.endswith(".json"):
        print("Detected JSON format")
        with open(input_file, "r") as f:
            json_data = json.load(f)

    elif input_file.endswith(".jsonl"):
        print("Detected JSONL format")
        json_data = []
        with open(input_file, "r") as f:
            for line in f:
                json_data.append(json.loads(line.strip()))

    else:
        raise ValueError("Unsupported file type. Please provide FASTA, JSON, JSONL, or ZIP containing FASTA.")

    return json_data, processed_file_name


def main():
    parser = argparse.ArgumentParser(description="Process FASTA, JSON, JSONL, or ZIP containing FASTA")
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument("--output", "-o", help="Output file (default: output.jsonl)")
    args = parser.parse_args()

    input_file = args.input_file
    print("Uploaded file:", input_file)

    json_data, processed_file_name = process_file(input_file)

    print("Total records parsed:", len(json_data))
    if json_data:
        print("\nFirst Record Preview:")
        print(json.dumps(json_data[0], indent=4)[:500])

    if json_data:
        # Use provided output file or default
        output_file = args.output or "output.jsonl"
        output_json = output_file.replace('.jsonl', '.json')
        
        with open(output_json, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"\nCreated: {output_json}")

        with open(output_file, "w") as f:
            for entry in json_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Created: {output_file}")

    if processed_file_name and os.path.exists(processed_file_name) and processed_file_name != input_file:
        os.remove(processed_file_name)


if __name__ == "__main__":
    main()
