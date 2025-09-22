#!/usr/bin/env python3
"""
Compliance filters for therapeutics pipeline outputs.

Implements threshold checks for:
- mRNAid-like metrics: CAI (approx), GC content
- IEDB epitope predictions: strong binders and allele coverage (Class I only, from existing results)

Other requested checks (InterProScan, Rfam/RNAcentral, CRISPOR/siSPOTR) are marked as not_evaluated
unless external tools are integrated.
"""

import argparse
import json
import os
import csv
from typing import Dict, Any


def load_jsonl(path: str):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def gc_content(seq: str) -> float:
    seq = (seq or "").upper()
    if not seq:
        return 0.0
    g = seq.count('G')
    c = seq.count('C')
    return 100.0 * (g + c) / len(seq)


# Very lightweight CAI approximation using a uniform preferred codon set (demo purpose)
PREFERRED_CODONS = {
    'A': 'GCT', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT',
    'I': 'ATT', 'K': 'AAA', 'L': 'CTG', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA',
    'R': 'CGT', 'S': 'TCT', 'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT'
}


def approximate_cai(dna_seq: str) -> float:
    seq = (dna_seq or "").upper()
    if len(seq) < 3:
        return 0.0
    weights = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if len(codon) < 3:
            break
        # Translate codon (DNA) to amino acid (rough mapping)
        aa = DNA_CODON_TO_AA.get(codon, None)
        if aa is None:
            continue
        preferred = PREFERRED_CODONS.get(aa)
        weights.append(1.0 if preferred == codon else 0.5)  # crude weighting
    if not weights:
        return 0.0
    return sum(weights) / len(weights)


DNA_CODON_TO_AA = {
    'GCT':'A','GCC':'A','GCA':'A','GCG':'A','TGT':'C','TGC':'C','GAT':'D','GAC':'D',
    'GAA':'E','GAG':'E','TTT':'F','TTC':'F','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
    'CAT':'H','CAC':'H','ATT':'I','ATC':'I','ATA':'I','AAA':'K','AAG':'K',
    'TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATG':'M','AAT':'N','AAC':'N',
    'CCT':'P','CCC':'P','CCA':'P','CCG':'P','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGA':'R','AGG':'R',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','AGT':'S','AGC':'S','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V','TGG':'W','TAT':'Y','TAC':'Y'
}


def check_mrnaid_constraints(optimized_items):
    # Use first optimized record if available
    result = {
        'evaluated': False,
        'passed': False,
        'details': {}
    }
    if not optimized_items:
        return result
    item = optimized_items[0]
    seq = item.get('optimized_sequence') or item.get('sequence') or ''
    gc = gc_content(seq)
    cai = approximate_cai(seq)
    # Basic checks available locally
    passed = (cai >= 0.7) and (40.0 <= gc <= 60.0)
    result['evaluated'] = True
    result['passed'] = passed
    result['details'] = {
        'cai': round(cai, 4),
        'gc_percent': round(gc, 2),
        'thresholds': {'cai_min': 0.7, 'gc_range': [40, 60]},
        'notes': 'Other criteria (UTR MFE, Kozak, uORFs, motifs) not evaluated here.'
    }
    return result


def load_mhc_class_i_csv(csv_path: str):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def check_epitope_constraints(mhc_csv_path: str):
    result = {
        'evaluated': False,
        'passed': False,
        'details': {}
    }
    rows = load_mhc_class_i_csv(mhc_csv_path)
    if not rows:
        return result

    strong = 0
    alleles = set()
    peptides = {}
    for r in rows:
        try:
            perc = float(r.get('percentile', '')) if r.get('percentile') not in (None, '') else None
        except Exception:
            perc = None
        allele = r.get('allele', '')
        peptide = r.get('peptide', '')
        if perc is not None and perc <= 2.0:
            strong += 1
            alleles.add(allele)
            peptides.setdefault(peptide, set()).add(allele)

    # Simple pass: at least one peptide binds to >=3 alleles (proxy for coverage)
    multi_binding = any(len(a_set) >= 3 for a_set in peptides.values())
    passed = strong > 0 and multi_binding

    result['evaluated'] = True
    result['passed'] = passed
    result['details'] = {
        'num_predictions': len(rows),
        'strong_binders': strong,
        'unique_alleles': len(alleles),
        'peptides_with_>=3_alleles': sum(1 for a in peptides.values() if len(a) >= 3),
        'thresholds': {
            'mhc_i_percentile_max': 2.0,
            'min_alleles_per_epitope': 3
        },
        'notes': 'Population coverage, proteasomal/TAP, conservation, toxicity not evaluated here.'
    }
    return result


def main():
    p = argparse.ArgumentParser(description='Apply therapeutics compliance filters.')
    p.add_argument('--optimized_jsonl', default='optimized_mrna.jsonl', help='Optimized mRNA JSONL path')
    p.add_argument('--mhc_csv', default=os.path.join('results', 'iedb_analysis', 'mhc_class_i_predictions.csv'), help='IEDB MHC-I CSV path')
    p.add_argument('--out', default='compliance_report.json', help='Output JSON report')
    args = p.parse_args()

    report: Dict[str, Any] = {
        'checks': {},
        'summary': {}
    }

    # mRNAid-like check
    optimized = load_jsonl(args.optimized_jsonl)
    if not optimized and os.path.exists('optimized_mrna_constrained.jsonl'):
        optimized = load_jsonl('optimized_mrna_constrained.jsonl')
    mrnaid_check = check_mrnaid_constraints(optimized)
    report['checks']['mrnaid'] = mrnaid_check

    # Epitope check (MHC class I only based on available output)
    epitope_check = check_epitope_constraints(args.mhc_csv)
    report['checks']['epitope_mhci'] = epitope_check

    # Not evaluated components placeholders
    report['checks']['interproscan_domains'] = {
        'evaluated': False,
        'passed': False,
        'details': {'reason': 'InterProScan CLI not integrated in this environment.'}
    }
    report['checks']['rfam_rnacentral'] = {
        'evaluated': False,
        'passed': False,
        'details': {'reason': 'Rfam/RNAcentral scans not integrated.'}
    }
    report['checks']['crispor_offtarget'] = {
        'evaluated': False,
        'passed': False,
        'details': {'reason': 'CRISPOR/CFD and genome-wide off-target scan not integrated.'}
    }
    report['checks']['siRNA_offtarget'] = {
        'evaluated': False,
        'passed': False,
        'details': {'reason': 'siSPOTR or equivalent not integrated.'}
    }

    # Summary
    pass_keys = [k for k, v in report['checks'].items() if v['evaluated'] and v['passed']]
    fail_keys = [k for k, v in report['checks'].items() if v['evaluated'] and not v['passed']]
    not_eval = [k for k, v in report['checks'].items() if not v['evaluated']]

    report['summary'] = {
        'passed_checks': pass_keys,
        'failed_checks': fail_keys,
        'not_evaluated': not_eval,
        'overall_pass': len(fail_keys) == 0 and len(pass_keys) > 0
    }

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()


