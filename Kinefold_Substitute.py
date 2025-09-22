#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
        accept='.fna,.fana,.fa,.fasta,.json,.jsonl',  
        multiple=True,
        description='Upload Folder Files'
    )
    status_html = widgets.HTML(
        "<p><i>Upload files (.fna, .fana, .fa, .fasta, .json, .jsonl)...</i></p>"
    )
    output = widgets.Output()
    uploaded_sequences = {}
    uploaded_json = {}
    uploaded_jsonl = []
    def on_upload(change):
        output.clear_output()
        status_html.value = "<p><i>Processing files...</i></p>"
        try:
            uploaded_json.clear()
            uploaded_jsonl.clear()
            uploaded_sequences.clear()
            for filename, fileinfo in file_upload.value.items():
                content = fileinfo.get('content', None)
                with output:
                    print(f"---\nProcessing file: {filename}")
                    if content is None:
                        print("  ❌ No content field found!")
                        continue
                    # Robust decoding
                    if isinstance(content, bytes):
                        try:
                            content_str = content.decode('utf-8')
                        except Exception as e:
                            print(f"  ❌ Decode error: {e}")
                            continue
                    elif isinstance(content, str):
                        content_str = content
                    else:
                        print(f"  ❌ Unknown content type: {type(content)}")
                        continue
                    ext = filename.split('.')[-1].lower()
                    print(f"  Extension: .{ext}")
                    print(f"  First 80 chars:\n{content_str[:80]}")
                    if ext in ['fa', 'fna', 'fasta', 'fana']:
                        try:
                            seqs = parse_fasta_content(content_str)
                            if not seqs:
                                print("  ❌ No sequences parsed! Check FASTA format.")
                            else:
                                uploaded_sequences.update(seqs)
                                print(f"  🟢 Parsed sequences from: {filename}")
                                for k, v in seqs.items():
                                    print(f"    - {k} ({len(v)} bp)")
                        except Exception as e:
                            print(f"  ❌ FASTA parse error: {e}")
                    elif ext == 'json':
                        try:
                            data = json.loads(content_str)
                            uploaded_json[filename] = data
                            print(f"  🟢 Loaded JSON file: {filename}")
                            print(f"    Top-level keys: {list(data.keys())}")
                        except Exception as e:
                            print(f"  ❌ JSON parse error: {e}")
                    elif ext == 'jsonl':
                        try:
                            items = parse_jsonl_content(content_str)
                            uploaded_jsonl.extend(items)
                            print(f"  🟢 Loaded JSONL file: {filename} ({len(items)} records)")
                        except Exception as e:
                            print(f"  ❌ JSONL parse error: {e}")
                    else:
                        print(f"  ⏭️ Skipped unknown file type: {filename}")
            status_html.value = "<p style='color:green;'>Upload finished. All files processed.</p>"
        except Exception as e:
            status_html.value = f"<p style='color:red;'>❌ Error: {e}</p>"
    file_upload.observe(on_upload, names='value')
    vbox = widgets.VBox([
        widgets.HTML("<h3>📁 Folder Upload: .fna, .fana, .fa, .fasta, .json, .jsonl files</h3>"),
        status_html,
        file_upload,
        output
    ])
    display(vbox)
    return uploaded_sequences, uploaded_json, uploaded_jsonl

uploaded_sequences, uploaded_json, uploaded_jsonl = upload_folder_widget()


# In[4]:


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Example: pick the first uploaded sequence
if uploaded_sequences:
    sequence_name, sequence = next(iter(uploaded_sequences.items()))
else:
    raise ValueError("No sequences uploaded. Please upload a FASTA file first.")

model_ckpt = "biomistral/biomistral-7b-bio-v0.1"

task_instruction = (
    "Predict the RNA folding kinetics and pathways for the following RNA sequence "
    "using the same kinetic models. List folding intermediates, key transition steps, "
    "and comment on the folding pathway if possible.\n\n"
    f"RNA sequence:\n{sequence}\n\n"
)

print(f"Using sequence: {sequence_name} ({len(sequence)} bp)")
print("Querying model for folding kinetics and pathways...")

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_ckpt, torch_dtype=torch.float16, device_map="auto")

# Use CUDA if available
device = 0 if torch.cuda.is_available() else -1
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# Generate prediction
result = nlp(task_instruction, max_new_tokens=250)[0]['generated_text']

print("=== Model Output ===\n")
if task_instruction in result:
    print(result[len(task_instruction):].strip())
else:
    print(result.strip())


# In[ ]:




