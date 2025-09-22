#!/usr/bin/env python3
"""
Constrained mRNA optimization to target CAI ≥ 0.7 and GC between 40–60%.
Input: FASTA/JSON/JSONL with fields containing DNA coding sequence (5'->3').
Output: JSON/JSONL with optimized sequence and metrics.
"""

import argparse
import json
import os
from typing import List, Dict

from Bio import SeqIO


def gc_content(seq: str) -> float:
    s = (seq or "").upper()
    if not s:
        return 0.0
    return 100.0 * (s.count('G') + s.count('C')) / len(s)


def optimize_sequence_dnachisel(seq: str) -> Dict:
    try:
        from dnachisel import (
            DnaOptimizationProblem,
            EnforceGCContent,
            AvoidPattern,
            EnforceTranslation,
            MaximizeCAI
        )
    except Exception as e:
        return {
            'optimized_sequence': seq,
            'notes': f'DNAChisel not available ({e}); returned original sequence.'
        }

    constraints = [
        EnforceGCContent(mini=0.40, maxi=0.60),
        AvoidPattern("GGTCTC"),
        AvoidPattern("GAGACC"),
        EnforceTranslation()
    ]
    objectives = [MaximizeCAI(species="e_coli")]

    problem = DnaOptimizationProblem(
        sequence=seq,
        constraints=constraints,
        objectives=objectives
    )
    problem.resolve_constraints()
    problem.optimize()
    optimized = problem.sequence

    # Approximate CAI through DNAChisel's objective score is non-trivial; leave as note
    return {
        'optimized_sequence': optimized,
        'notes': 'Optimized with EnforceGCContent(40–60%) and MaximizeCAI(species=e_coli).'
    }


def process_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser(description='Constrained mRNA optimization (GC 40–60%, CAI high).')
    p.add_argument('input_file', help='Input FASTA/JSON/JSONL')
    p.add_argument('--output', '-o', default='optimized_mrna_constrained.jsonl')
    args = p.parse_args()

    input_file = args.input_file
    results: List[Dict] = []

    if input_file.endswith(('.fasta', '.fa', '.fna')):
        for r in SeqIO.parse(input_file, 'fasta'):
            seq = str(r.seq)
            opt = optimize_sequence_dnachisel(seq)
            optimized = opt['optimized_sequence']
            results.append({
                'id': r.id,
                'description': r.description,
                'original_sequence': seq,
                'optimized_sequence': optimized,
                'gc_content_original': round(gc_content(seq), 2),
                'gc_content_optimized': round(gc_content(optimized), 2),
                'note': opt.get('notes', '')
            })
    elif input_file.endswith('.json'):
        data = json.load(open(input_file))
        for i, item in enumerate(data if isinstance(data, list) else [data]):
            seq = item.get('sequence') or item.get('optimized_sequence') or ''
            if not seq:
                continue
            opt = optimize_sequence_dnachisel(seq)
            optimized = opt['optimized_sequence']
            results.append({
                'id': item.get('id', f'seq_{i}') if isinstance(item, dict) else f'seq_{i}',
                'description': item.get('description', '') if isinstance(item, dict) else '',
                'original_sequence': seq,
                'optimized_sequence': optimized,
                'gc_content_original': round(gc_content(seq), 2),
                'gc_content_optimized': round(gc_content(optimized), 2),
                'note': opt.get('notes', '')
            })
    elif input_file.endswith('.jsonl'):
        for i, item in enumerate(process_jsonl(input_file)):
            seq = item.get('sequence') or item.get('optimized_sequence') or ''
            if not seq:
                continue
            opt = optimize_sequence_dnachisel(seq)
            optimized = opt['optimized_sequence']
            results.append({
                'id': item.get('id', f'seq_{i}'),
                'description': item.get('description', ''),
                'original_sequence': seq,
                'optimized_sequence': optimized,
                'gc_content_original': round(gc_content(seq), 2),
                'gc_content_optimized': round(gc_content(optimized), 2),
                'note': opt.get('notes', '')
            })
    else:
        raise SystemExit('Unsupported input format. Use FASTA/JSON/JSONL.')

    with open(args.output, 'w') as f:
        for rec in results:
            f.write(json.dumps(rec) + '\n')

    print(f'Optimized {len(results)} records → {args.output}')


if __name__ == '__main__':
    main()


