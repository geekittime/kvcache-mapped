"""Describe complete Hotpot outputs without changing predictions or scoring."""
import json
import re
from pathlib import Path
from core import atomic_json, digest
from evaluate import read_jsonl, norm_text, answer_body
from protocol_guards import matching_pair, generation_parity

HERE = Path(__file__).resolve().parent

def describe(rows):
    n = len(rows)
    assert n
    return dict(n=n, **{k: sum(r[k] for r in rows) / n
        for k in ['em', 'f1', 'raw_em', 'raw_f1', 'generated_tokens']},
        length_stops=sum(r['stop_reason'] == 'length' for r in rows),
        multiline=sum('\n' in r['text'].strip() for r in rows),
        explicit_answer_label=sum(bool(re.search(
            r'(?:^|\n)\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)',
            answer_body(r['text']), re.I)) for r in rows),
        extraction_changes_normalized_text=sum(
            norm_text(r['pred']) != norm_text(r['text']) for r in rows))

def main():
    result = dict(passed=True, scoring_unchanged=True, pairs={},
        interpretation='Descriptive output-format and reference-defined subgroup analysis. '
        'No new correctness metric, semantic judge, filtering or parser adjustment. '
        'Subgroups are exploratory; raw and extracted scores must both be reported.')
    lines = ['# HotpotQA complete-output diagnostics', '',
        'Both conditions use the same frozen prompts and prediction-only extraction. '
        'These descriptive subgroups do not alter the primary scores.', '',
        '| Direction | Group | n | EM weighted/readout (%) | F1 weighted/readout (%) | Raw F1 weighted/readout (%) | Mean tokens weighted/readout |',
        '|---|---|---:|---|---|---|---|']
    for pair in ['06_17', '17_06']:
        records, protocols, receipts = {}, {}, {}
        for mode in ['weighted', 'readout']:
            base = HERE / 'runs' / pair / mode / 'hotpot_off'
            out = base / 'full'
            stage = json.loads((base / 'stage.json').read_text())
            audit = json.loads((out / 'audit.json').read_text())
            metrics = json.loads((out / 'results.json').read_text())
            assert stage['passed'] and audit['passed'] and audit['complete_dataset']
            assert metrics['complete'] and metrics['n'] == 7405
            assert digest(out / 'results.json') == stage['full_results_sha256'] == audit['results_sha256']
            assert digest(out / 'predictions.jsonl') == metrics['predictions_sha256'] == audit['predictions_sha256']
            rows = read_jsonl(out / 'predictions.jsonl')
            assert len(rows) == len({r['index'] for r in rows}) == 7405
            assert generation_parity(read_jsonl(base / 'smoke/predictions.jsonl'), rows) == stage['generation_parity']
            records[mode] = sorted(rows, key=lambda r: r['index'])
            protocols[mode] = metrics['protocol']
            receipts[mode] = dict(results_sha256=digest(out / 'results.json'),
                                  predictions_sha256=digest(out / 'predictions.jsonl'))
        matching_pair(protocols['weighted'], protocols['readout'])
        a, b = records['weighted'], records['readout']
        for x, y in zip(a, b):
            assert all(x[k] == y[k] for k in ['index', 'gold', 'prompt_ids_sha256', 'type', 'level'])
        selectors = {'all': lambda r: True,
            'yes_no_reference': lambda r: norm_text(r['gold']) in ('yes', 'no'),
            'span_reference': lambda r: norm_text(r['gold']) not in ('yes', 'no')}
        for kind in sorted({r['type'] for r in a}):
            selectors['type_' + kind] = lambda r, kind=kind: r['type'] == kind
        groups = {}
        for name, select in selectors.items():
            x, y = [r for r in a if select(r)], [r for r in b if select(r)]
            assert [r['index'] for r in x] == [r['index'] for r in y]
            da, db = describe(x), describe(y)
            groups[name] = dict(weighted=da, readout=db,
                em_gains=sum(not p['em'] and q['em'] for p, q in zip(x, y)),
                em_losses=sum(p['em'] and not q['em'] for p, q in zip(x, y)))
            lines.append(f"| {pair} | {name} | {len(x)} | {da['em']*100:.3f}/{db['em']*100:.3f} | {da['f1']*100:.3f}/{db['f1']*100:.3f} | {da['raw_f1']*100:.3f}/{db['raw_f1']*100:.3f} | {da['generated_tokens']:.2f}/{db['generated_tokens']:.2f} |")
        result['pairs'][pair] = dict(receipts=receipts, groups=groups)
    lines += ['',
        'Extraction normalizes explicit answer labels, boxed answers and leading yes/no. '
        'Raw scores compare the entire generated response with the reference. '
        'Long explanations can lower raw F1 even when they contain a correct answer. '
        'Conversely, extraction can miss unlabelled correct spans or select an earlier explicit answer in a contradictory response. '
        'Thus neither score alone establishes a causal improvement in reasoning.', '',
        'The frozen prompt requests a short answer. It is not changed after observing these outputs. '
        'No native-model control was run in this comparison; the absolute gap from native performance cannot be inferred.', '']
    result['script_sha256'] = digest(__file__)
    atomic_json(HERE / 'checks/hotpot_complete_diagnostics.json', result)
    (HERE / 'HOTPOT_DIAGNOSTICS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(passed=True, directions=2, complete_cells=4, samples_per_cell=7405)))

if __name__ == '__main__':
    main()
