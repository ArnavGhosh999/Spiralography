#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'viennarna', 'ipywidgets'])


# In[2]:


import RNA
import ipywidgets as widgets
# from IPython.display import display  # Not needed for standalone Python
import json

def parse_fasta_content(content):
    sequences = {}
    current_seq = ""
    current_name = None
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_name and current_seq:
                sequences[current_name] = current_seq.upper()
            current_name = line[1:].split()[0]
            current_seq = ""
        elif line and not line.startswith(';'):
            current_seq += line.replace(' ', '').replace('\t', '')
    if current_name and current_seq:
        sequences[current_name] = current_seq.upper()
    return sequences

def parse_jsonl_content(content):
    lines = content.strip().split('\n')
    items = []
    for line in lines:
        try:
            item = json.loads(line)
            items.append(item)
        except Exception:
            continue
    return items

def run_viennarna_on_sequences(seq_dict):
    results = []
    for label, s in seq_dict.items():
        try:
            struct, mfe = RNA.fold(s)
            results.append((label, s, struct, mfe))
        except Exception as e:
            results.append((label, s, "Error", str(e)))
    return results

def run_viennarna_on_jsonl(jsonl_items):
    results = []
    for idx, item in enumerate(jsonl_items):
        seq = None
        label = None
        if isinstance(item, dict):
            seq = item.get("sequence") or item.get("seq")
            label = item.get("id") or f"jsonl_record_{idx+1}"
        if seq:
            try:
                struct, mfe = RNA.fold(seq)
                results.append((label, seq, struct, mfe))
            except Exception as e:
                results.append((label, seq, "Error", str(e)))
    return results

def run_viennarna_on_json(json_data):
    results = []
    for k, v in json_data.items():
        if isinstance(v, str) and set(v.upper()).issubset(set("AUGC")):
            try:
                struct, mfe = RNA.fold(v)
                results.append((k, v, struct, mfe))
            except Exception as e:
                results.append((k, v, "Error", str(e)))
    return results

def upload_all_rna_widget():
    # Accepts .fa, .fna, .fas, .fana, and .fasta now!
    file_upload = widgets.FileUpload(
        accept='.fna,.fas,.fa,.fasta,.json,.jsonl,.fana',
        multiple=True,
        description='Upload All RNA Files'
    )
    status_html = widgets.HTML("<p><i>Upload your .fna/.fana/.fa/.fasta, .json, and .jsonl files&hellip;</i></p>")
    output = widgets.Output()
    rna_output = widgets.Output()

    uploaded_sequences = {}
    uploaded_json = {}
    uploaded_jsonl = []

    def on_upload(change):
        output.clear_output()
        rna_output.clear_output()
        status_html.value = "<p><i>Processing files...</i></p>"
        if not file_upload.value:
            status_html.value = "<p style='color:red;'>No file(s) selected!</p>"
            return
        try:
            uploaded_json.clear()
            uploaded_jsonl.clear()
            uploaded_sequences.clear()
            for f in file_upload.value:
                filename = f['name']
                content = bytes(f['content']).decode('utf-8')
                ext = filename.split('.')[-1].lower()
                # Accept .fa, .fna, .fas, .fana, .fasta
                if ext in ['fa', 'fna', 'fas', 'fana', 'fasta']:
                    seqs = parse_fasta_content(content)
                    uploaded_sequences.update(seqs)
                    with output:
                        print(f"Parsed sequences from: {filename}")
                        for k, v in seqs.items():
                            print(f"  - {k} ({len(v)} nt)")
                elif ext == 'json':
                    data = json.loads(content)
                    uploaded_json[filename] = data
                    with output:
                        print(f"Loaded JSON file: {filename}")
                        print(f"  Top-level keys: {list(data.keys())}")
                elif ext == 'jsonl':
                    items = parse_jsonl_content(content)
                    uploaded_jsonl.extend(items)
                    with output:
                        print(f"Loaded JSONL file: {filename} ({len(items)} records)")
                else:
                    with output:
                        print(f"Skipped unknown file type: {filename}")
            status_html.value = "<p style='color:green;'>Upload finished. Running ViennaRNA analysis...</p>"

            with rna_output:
                if uploaded_sequences:
                    print("\nFASTA/FA/FNA/FAS/FANA Sequences (ViennaRNA results):")
                    for label, seq, struct, mfe in run_viennarna_on_sequences(uploaded_sequences):
                        print(f"  > {label}\n    Seq: {seq}\n    Struct: {struct}\n    MFE: {mfe}\n")
                if uploaded_jsonl:
                    print("\nJSONL Sequences (ViennaRNA results):")
                    for label, seq, struct, mfe in run_viennarna_on_jsonl(uploaded_jsonl):
                        print(f"  > {label}\n    Seq: {seq}\n    Struct: {struct}\n    MFE: {mfe}\n")
                if uploaded_json:
                    print("\nJSON Sequences (ViennaRNA results):")
                    for fname, data in uploaded_json.items():
                        print(f"  File: {fname}")
                        for label, seq, struct, mfe in run_viennarna_on_json(data):
                            print(f"    > {label}\n      Seq: {seq}\n      Struct: {struct}\n      MFE: {mfe}\n")
                print("Done!")
            status_html.value = "<p style='color:green;'>RNA analysis complete.</p>"
        except Exception as e:
            status_html.value = f"<p style='color:red;'>❌ Error: {e}</p>"

    file_upload.observe(on_upload, names='value')

    vbox = widgets.VBox([
        widgets.HTML("<h3>🧬 RNA File Uploader (.fna/.fana/.fa/.fasta, .json, .jsonl files)</h3>"),
        status_html,
        file_upload,
        output,
        widgets.HTML("<b>Results:</b>"),
        rna_output
    ])
    print("Interactive widget created (display functionality not available in standalone mode)")
    return uploaded_sequences, uploaded_json, uploaded_jsonl

# Just call this in a notebook cell
uploaded_sequences, uploaded_json, uploaded_jsonl = upload_all_rna_widget()


# In[ ]:




