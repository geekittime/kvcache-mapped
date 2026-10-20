"""Package final audited results; never launch inference or accept partial cells."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
COUNTS = {'arc_off': 1172, 'arc_on': 1172, 'mmlu_off': 1400, 'mmlu_on': 1400,
          'hotpot_off': 7405, **{f'ruler_{k}k_{mode}': 400
                                for k in [4, 8, 16, 32] for mode in ['off', 'on']}}
PAIRS = {'06_17': 'Qwen3-0.6B → 1.7B', '17_06': 'Qwen3-1.7B → 0.6B'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.delivery.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def validate_result(result):
    assert result['complete'] is True
    assert result['accepted_cells'] == result['expected_cells'] == 52
    assert set(result['upstream']) == set(PAIRS)
    assert all(proof['passed'] for proof in result['upstream'].values())
    expected = {(p, c) for p in PAIRS for c in COUNTS}
    comparisons = result['comparisons']
    assert len(comparisons) == len(expected) == 26
    assert {(c['pair'], c['case']) for c in comparisons} == expected
    for c in comparisons:
        assert c['status'] == 'accepted' and c['n'] == COUNTS[c['case']]
        assert set(c['complete_conditions']) == {'weighted', 'readout'}
        for mode in ['weighted', 'readout']:
            assert c['complete_conditions'][mode]['n'] == c['n']
            parity = c['generation_parity'][mode]
            assert parity['n'] > 0
            assert parity['n'] == parity['first_token_exact'] == parity['full_token_sequence_exact'] == parity['parsed_answer_exact']
        expected_metrics = ({'em', 'f1', 'raw_em', 'raw_f1'} if c['case'].startswith('hotpot')
                            else {'recall'} if c['case'].startswith('ruler')
                            else {'macro_accuracy'} if c['case'].startswith('mmlu') else {'accuracy'})
        assert set(c['metrics']) == expected_metrics
        for name, metric in c['metrics'].items():
            assert all(math.isfinite(metric[k]) for k in ['weighted', 'readout', 'delta'])
            assert math.isclose(metric['delta'], metric['readout'] - metric['weighted'], abs_tol=1e-12)
            for mode in ['weighted', 'readout']:
                assert metric[mode] == c['complete_conditions'][mode][name]
            lo, hi = metric['paired_bootstrap_95']
            assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi


def render(result):
    validate_result(result)
    lines = ['# V Readout Ridge 对比实验最终报告', '',
             '52/52 个正式评测单元已通过独立审计。两个方向各完成 400 条训练样本拟合、64 条独立验证样本检查。', '',
             'Weighted 为 KV-Lingo Stage1 attention-weighted fitting 加 λ=0.01 ridge；Readout 仅修改 V 拟合目标。'
             '同方向 K 权重逐位相同。两组均无 bias，无 Stage2，未测 TTFT。', '',
             '| 模型方向 | 任务/思考模式 | 样本数 | 指标 | Weighted (%) | Readout (%) | 差值 (百分点) | 配对 95% CI |',
             '|---|---|---:|---|---:|---:|---:|---|']
    for c in result['comparisons']:
        for name, m in c['metrics'].items():
            lo, hi = m['paired_bootstrap_95']
            lines.append(f"| {PAIRS[c['pair']]} | {c['case']} | {c['n']} | {name} | {m['weighted']*100:.3f} | {m['readout']*100:.3f} | {m['delta']*100:+.3f} | [{lo*100:+.3f}, {hi*100:+.3f}] |")
    lines += ['', 'on/off 分别为开启/关闭思考。MMLU-Pro 为 14 科各 100 题的宏平均准确率；ARC-C 为完整测试集准确率。'
              'RULER 仅为 niah_single_2 子任务的 recall，不是完整 RULER 套件综合分。'
              'HotpotQA 为完整 distractor 验证集的答案提取 EM/F1，同时保留整段输出 raw EM/F1。', '',
              '## 长度与解析检查', '',
              '所有截断和未解析题都保留在分母中，下表顺序均为 Weighted / Readout。', '',
              '| 方向 | 任务/模式 | 达到输出长度上限 | 未解析 | 平均生成 token 数 |',
              '|---|---|---|---|---|']
    for c in result['comparisons']:
        stops = c['length_stops']; unparsed = c['unparsed']; lengths = c['mean_generated_tokens']
        lines.append(f"| {c['pair']} | {c['case']} | {stops['weighted']}/{stops['readout']} | {unparsed['weighted']}/{unparsed['readout']} | {lengths['weighted']:.2f}/{lengths['readout']:.2f} |")
    lines += ['', '## 独立验证与结论边界', '',
              '| 方向 | 条件 | 样本平均 KL | token 平均 KL | token 平均 CE | PPL | 相对 V readout 平方误差 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for pair, proof in result['upstream'].items():
        for mode, m in proof['validation_summary'].items():
            lines.append(f"| {pair} | {mode} | {m['kl_sample_mean']:.6f} | {m['kl_token_mean']:.6f} | {m['mapped_ce']:.6f} | {m['mapped_ppl']:.6f} | {m['readout_v_relative']:.6f} |")
    lines += ['', 'Readout 的正则化训练目标及独立验证 continuation KL/CE 更低，但独立验证中的 teacher-attention V readout 重建误差更高。'
              '训练目标下降不能代替泛化提升；训练时固定的原生 attention 与运行时 mapped K 产生的 attention 不同，当前观察不能证明因果机制。', '',
              'HotpotQA 的答案提取指标与整段输出 raw 指标可能方向相反；0.6B→1.7B 的 raw EM/F1 下降。'
              '两方向 yes/no 子集均退步，详见 [HOTPOT_DIAGNOSTICS.md](HOTPOT_DIAGNOSTICS.md)。不能概括为所有指标一致提升。', '',
              '训练数据来源及长度混合遵循论文描述，但作者未公开精确样本 ID 和 packing 实现。'
              '本地训练 prefix 总量为 3,809,293 tokens，论文约 2.8M；不能声称逐样本复原作者隐藏划分。'
              '使用本地公开说明的 prompt/评分实现；MC 统一采用 parser v2，保留原始预测和 v1 评分。', '',
              '这是固定训练集、固定随机种子的两条件配对实验。区间由 2,000 次配对 bootstrap 得到，'
              'MMLU-Pro 按科目分层；未校正多重比较，不覆盖训练种子不确定性。未测原生模型性能基线。'
              '没有根据测试分数调整超参数。', '',
              '## 可追溯文件', '',
              '- [RESULTS.json](RESULTS.json)：完整机器可读结果和验收记录。',
              '- [checks/final_delivery.json](checks/final_delivery.json)：最终报告、各单元证据和预测文件的 SHA-256 清单。',
              '- [README.md](README.md)：冻结科学协议及已知复现差异。',
              '- [PROCESS.md](PROCESS.md)：阶段执行、问题修复与验收过程。',
              '- [RUNBOOK.md](RUNBOOK.md)：复核及恢复命令。', '',
              f"正式结果生成时间（UTC）：{result['generated_at_utc']}。未 push。", '']
    return '\n'.join(lines)


def package(root):
    checks = root / 'checks'
    assert not list(checks.glob('failure_*.json'))
    assert not (checks/'evaluation_resume_failure.json').exists()
    assert not (checks/'guard_failure.json').exists()
    guard = read(checks/'guard_complete.json')
    assert guard['passed'] and guard['accepted_cells'] == 52
    assert guard['coordinator'] == read(checks/'resume_evaluation_process.json')
    result = read(root/'RESULTS.json')
    validate_result(result)
    assert result['protocol_sha256'] == digest(root/'README.md')
    evidence = {}
    def record(path):
        evidence[str(path.relative_to(root))] = digest(path)
    for c in result['comparisons']:
        for mode in ['weighted', 'readout']:
            base = root/'runs'/c['pair']/mode/c['case']
            stage = read(base/'stage.json')
            assert stage['passed'] and stage['n'] == c['n']
            assert stage['generation_parity'] == c['generation_parity'][mode]
            assert stage['full_results_sha256'] == digest(base/'full/results.json')
            record(base/'stage.json')
            for part in ['smoke', 'full']:
                folder = base/part; audit = read(folder/'audit.json')
                assert audit['passed'] and audit['complete_dataset'] == (part == 'full')
                assert audit['predictions_sha256'] == digest(folder/'predictions.jsonl')
                assert audit['results_sha256'] == digest(folder/'results.json')
                for name in ['audit.json', 'protocol.json', 'runtime.json', 'results.json', 'predictions.jsonl']:
                    record(folder/name)
            if c['case'].startswith(('arc', 'mmlu')):
                record(base/'full/mc_scoring_v2.json')
    for pair, proof in result['upstream'].items():
        fit = root/'artifacts'/pair/'full_v1/run.json'
        val = checks/f'validation_{pair}_full/results.json'
        assert digest(fit) == proof['fit_run_sha256']
        assert digest(val) == proof['validation_results_sha256']
        record(fit); record(val)
    for path in [root/'RESULTS.json', root/'RESULTS.md', root/'README.md', checks/'guard_complete.json']:
        record(path)
    report = root/'FINAL_REPORT_ZH.md'
    report.write_text(render(result)); record(report)
    write_json(checks/'final_delivery.json', dict(passed=True, complete=True, accepted_cells=52,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), sha256=evidence,
        interpretation='Packaging after full independent report audit; no inference, scoring or protocol changes.'))
    marker = '## Final delivery packaged'
    process = root/'PROCESS.md'
    if marker not in process.read_text():
        with process.open('a') as f:
            f.write('\n'+marker+'\n\n52/52 cells passed the final independent report gate and execution guard. '
                    'finalize_delivery.py verified the 26 paired comparisons, source evidence and prediction hashes; '
                    'FINAL_REPORT_ZH.md and checks/final_delivery.json contain the final delivery. '
                    'Negative metrics and reproduction limitations are preserved. No push.\n')


def main(wait):
    with (HERE/'checks/final_delivery.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (HERE/'checks/guard_complete.json').exists():
            assert not (HERE/'checks/guard_failure.json').exists(), 'Execution guard failed; delivery refused'
            if not wait:
                raise RuntimeError('Final execution gate is still pending; no final delivery written')
            write_json(HERE/'checks/delivery_status.json', dict(complete=False, phase='waiting_for_final_execution_gate',
                time_utc=datetime.now(timezone.utc).isoformat()))
            time.sleep(30)
        package(HERE)
        write_json(HERE/'checks/delivery_status.json', dict(complete=True, phase='packaged',
            time_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    try:
        main(parser.parse_args().wait)
    except Exception:
        write_json(HERE/'checks/delivery_failure.json', dict(complete=False, traceback=traceback.format_exc()))
        raise
