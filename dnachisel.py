#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dnachisel import *
import ipywidgets as widgets
from IPython.display import display
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
    items = [json.loads(line) for line in lines if line.strip()]
    return items

def upload_folder_widget():
    file_upload = widgets.FileUpload(
        accept='',  # Accept all file types in folder
        multiple=True,
        description='Upload Folder Files'
    )
    status_html = widgets.HTML("<p><i>Upload all files from your folder (.fna, .json, .jsonl)...</i></p>")
    output = widgets.Output()
    
    uploaded_sequences = {}
    uploaded_json = {}
    uploaded_jsonl = []

    def on_upload(change):
        output.clear_output()
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
                if ext in ['fa', 'fna', 'fas']:
                    seqs = parse_fasta_content(content)
                    uploaded_sequences.update(seqs)
                    with output:
                        print(f"Parsed sequences from: {filename}")
                        for k, v in seqs.items():
                            print(f"  - {k} ({len(v)} bp)")
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
            status_html.value = "<p style='color:green;'>Upload finished.</p>"
        except Exception as e:
            status_html.value = f"<p style='color:red;'>❌ Error: {e}</p>"

    file_upload.observe(on_upload, names='value')
    vbox = widgets.VBox([
        widgets.HTML("<h3>📁 Folder Upload: .fna, .json, .jsonl files</h3>"),
        status_html,
        file_upload,
        output
    ])
    display(vbox)
    return uploaded_sequences, uploaded_json, uploaded_jsonl

uploaded_sequences, uploaded_json, uploaded_jsonl = upload_folder_widget()


# In[2]:


def select_and_optimize_sequence(sequences):
    import ipywidgets as widgets
    from dnachisel import DnaOptimizationProblem, AvoidPattern, EnforceTranslation, EnforceGCContent, MaximizeCAI
    from IPython.display import display

    if not sequences:
        print("No sequences loaded yet.")
        return
    key_dropdown = widgets.Dropdown(
        options=[(f"{k} ({len(v)}bp)", k) for k, v in sequences.items()],
        description='Select sequence:',
        disabled=False
    )
    run_button = widgets.Button(description='Optimize', button_style='success')
    out = widgets.Output()

    def run_optimization(b):
        out.clear_output()
        key = key_dropdown.value
        seq = sequences[key]
        with out:
            print(f"Selected sequence: {key} ({len(seq)} bp)")
            try:
                constraints = [
                    AvoidPattern("GGTCTC"),
                    AvoidPattern("GAGACC"),
                    EnforceGCContent(mini=0.4, maxi=0.6)
                ]
                objectives = []
                if len(seq) % 3 == 0 and seq.upper().startswith('ATG'):
                    constraints.append(EnforceTranslation())
                    objectives.append(MaximizeCAI(species="e_coli"))
                problem = DnaOptimizationProblem(
                    sequence=seq,
                    constraints=constraints,
                    objectives=objectives
                )
                problem.resolve_constraints()
                if problem.objectives:
                    problem.optimize()
                optimized = problem.sequence
                print("\n--- Optimization complete ---")
                print(f"FASTA:\n>{key}_optimized\n{optimized}")
                print(f"\nSimilarity: {sum(a==b for a,b in zip(seq, optimized))/len(seq)*100:.1f} %")
                print(f"GC content: {100*sum([c in 'GC' for c in seq])/len(seq):.1f}% → {100*sum([c in 'GC' for c in optimized])/len(optimized):.1f}%")
            except Exception as e:
                print(f"❌ Optimization failed: {e}")

    run_button.on_click(run_optimization)
    vbox = widgets.VBox([
        widgets.HTML("<h3>🎯 Sequence Selection & Optimization</h3>"),
        key_dropdown, run_button, out
    ])
    display(vbox)

select_and_optimize_sequence(uploaded_sequences)


# In[ ]:




