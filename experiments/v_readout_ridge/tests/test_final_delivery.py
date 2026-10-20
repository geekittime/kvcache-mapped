"""Final delivery must reject incomplete, duplicated or inconsistent evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finalize_delivery import COUNTS, PAIRS, package, validate_result


def complete_result():
    comparisons = []
    for pair in PAIRS:
        for case, n in COUNTS.items():
            keys = (['em', 'f1', 'raw_em', 'raw_f1'] if case.startswith('hotpot')
                    else ['recall'] if case.startswith('ruler')
                    else ['macro_accuracy'] if case.startswith('mmlu') else ['accuracy'])
            comparisons.append(dict(pair=pair, case=case, status='accepted', n=n,
                complete_conditions={m:dict(n=n, **{k:0.5 for k in keys}) for m in ['weighted','readout']},
                metrics={k:dict(weighted=0.5, readout=0.5, delta=0., paired_bootstrap_95=[0.,0.]) for k in keys},
                generation_parity={m:dict(n=4, first_token_exact=4, full_token_sequence_exact=4, parsed_answer_exact=4)
                                   for m in ['weighted','readout']}))
    return dict(complete=True, accepted_cells=52, expected_cells=52, comparisons=comparisons,
                upstream={p:dict(passed=True) for p in PAIRS})


def test_accept_complete_grid():
    validate_result(complete_result())


@pytest.mark.parametrize('fault', ['partial', 'duplicate', 'missing_condition', 'wrong_count',
                                  'score_mismatch', 'delta_mismatch', 'parity', 'nan_ci', 'missing_metric', 'missing_fit'])
def test_reject_invalid_grid(fault):
    r = complete_result(); c = r['comparisons'][0]
    if fault == 'partial': r['complete'] = False; r['accepted_cells'] = 29
    elif fault == 'duplicate': r['comparisons'][-1] = copy.deepcopy(c)
    elif fault == 'missing_condition': del c['complete_conditions']['readout']
    elif fault == 'wrong_count': c['n'] -= 1
    elif fault == 'score_mismatch': c['complete_conditions']['readout']['accuracy'] = 0.9
    elif fault == 'delta_mismatch': c['metrics']['accuracy']['delta'] = 0.1
    elif fault == 'parity': c['generation_parity']['weighted']['full_token_sequence_exact'] = 3
    elif fault == 'nan_ci': c['metrics']['accuracy']['paired_bootstrap_95'][0] = float('nan')
    elif fault == 'missing_metric': c['metrics'].clear()
    elif fault == 'missing_fit': del r['upstream']['06_17']
    with pytest.raises(AssertionError): validate_result(r)


def test_package_requires_execution_gate(tmp_path):
    (tmp_path/'checks').mkdir()
    (tmp_path/'RESULTS.json').write_text(json.dumps(complete_result()))
    with pytest.raises(FileNotFoundError): package(tmp_path)
    assert not (tmp_path/'FINAL_REPORT_ZH.md').exists()
    assert not (tmp_path/'checks/final_delivery.json').exists()


def test_package_rejects_guard_failure(tmp_path):
    (tmp_path/'checks').mkdir()
    (tmp_path/'checks/guard_failure.json').write_text('{}')
    with pytest.raises(AssertionError): package(tmp_path)
    assert not (tmp_path/'checks/final_delivery.json').exists()


def test_actual_interim_report_is_rejected():
    path = Path(__file__).resolve().parents[1]/'RESULTS.json'
    r = json.loads(path.read_text())
    if r['complete']:
        validate_result(r)
    else:
        with pytest.raises(AssertionError): validate_result(r)
