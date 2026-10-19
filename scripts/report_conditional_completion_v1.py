"""Document the complete pilot, its failed gate, and acquisition tradeoffs."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from certified_frfs.conditional_completion import ConditionalCompletion1NN

DEST = ROOT / 'results/conditional-completion-v1'


def synthetic_stress():
    rng = np.random.default_rng(20261012)
    d, n = 64, 80
    y = np.repeat([0, 1], n // 2)
    Q = rng.uniform(0, 1, (100, d))
    truth = np.tile([0, 1], 50)
    Q[:, 0] = np.where(truth == 0, .1, .9)
    rows = {}
    for scenario in ['common_reference_nuisance', 'independent_reference_nuisance']:
        R = np.full((n, d), .5) if scenario.startswith('common') else rng.uniform(0, 1, (n, d))
        R[:, 0] = y
        result = {}
        for method, bound, adaptive in [('shared_mi', 'shared', False),
                                        ('shared_adaptive', 'shared', True),
                                        ('independent_mi', 'independent', False)]:
            model = ConditionalCompletion1NN(R, y, list(range(d)), bound=bound, adaptive=adaptive)
            trace = [model.predict_one(lambda a, q=q: q[a]) for q in Q]
            prediction = [t['prediction'] for t in trace]
            result[method] = {'mean_saved_fraction': 1 - np.mean([t['count'] for t in trace]) / d,
                              'accuracy': float(np.mean(prediction == truth)),
                              'teacher_agreement': float(np.mean(prediction == [model.predict_full(q) for q in Q]))}
        rows[scenario] = result
    return {'seed': 20261012, 'd': d, 'reference_rows': n, 'queries': len(Q),
            'scope': 'descriptive stress, not used to pass the preregistered development gate', 'scenarios': rows}


def main():
    threadpool_limits(1)
    report = DEST / 'report.md'
    if report.exists():
        raise RuntimeError('report preserved; use a new version')
    s = json.loads((DEST / 'summary.json').read_text(encoding='utf-8'))
    audit = json.loads((DEST / 'independent-audit.json').read_text(encoding='utf-8'))
    assert audit['passed']
    names = list(s['dataset_means'])
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(DEST.glob('*-s9191-f*.json'))]
    thresholds, slowdowns = {}, {}
    for method in ['shared_mi', 'shared_adaptive', 'independent_mi']:
        thresholds[method] = []
        slowdowns[method] = []
        for r in rows:
            full, conditional = r['methods']['full'], r['methods'][method]
            saved = r['dimension'] * (1 - conditional['mean_acquired_fraction'])
            thresholds[method].append((conditional['online_seconds'] - full['online_seconds'])
                                      / len(r['test_labels']) / saved * 1e6)
            slowdowns[method].append(conditional['online_seconds'] / full['online_seconds'])
    stress = synthetic_stress()
    (DEST / 'synthetic-stress.json').write_text(json.dumps(stress, indent=2), encoding='utf-8')
    a = s['aggregate']
    text = [
        '# 核心机制重设计 v1：按样本认证获取属性', '',
        '**结论：新的机制已实现并验证了逐样本预测保留与属性获取减少；尚未建立分类优势或计算加速，不能据此认定达到 FSS 投稿水平。**', '',
        '旧方法从训练集鲁棒边际中选择一个全局属性子集。新方法固定完整 1NN 模型，'
        '对每个待分类样本逐步获取属性；只有证明未获取属性在整个单位区间内任取值都不会改变分类时才停止。'
        '因此认证直接决定能否少读属性，而不是一个与实际分类松散相关的训练奖励。', '',
        '这是研究目标的实质改变：保护完整模型的逐点预测，优化每个样本的获取量。'
        '它也会保留完整模型的错误，因此不能提高同一完整模型的准确率。', '',
        '## 完整试验', '',
        '6 个已暴露的开发数据集 × 3 个外层划分，种子 9191；18 个划分全部完成，未根据外层结果调参或筛选划分。'
        '静态 MI 对照的子集大小由训练内 3 折选择。以下先按划分求均值，再对数据集等权平均。', '',
        '| 方法 | 平衡准确率 | 平均少获取属性 | 与完整模型预测一致 |',
        '|---|---:|---:|---:|'
    ]
    for method, label in [('full', '完整 1NN'), ('shared_mi', '共享界 + 固定 MI 顺序'),
                          ('shared_adaptive', '共享界 + 瓶颈自适应获取'), ('independent_mi', '独立界 + 固定 MI 顺序'),
                          ('static_mi', '训练内选大小的静态 MI')]:
        agreement = '100%' if method != 'static_mi' else '不要求一致'
        text.append(f'| {label} | {a[method]["BA"]*100:.3f}% | {(1-a[method]["mean_acquired_fraction"])*100:.3f}% | {agreement} |')
    text += ['', '共享界和独立界在同一 MI 顺序下多节省 '
             f'{(a["independent_mi"]["mean_acquired_fraction"]-a["shared_mi"]["mean_acquired_fraction"])*100:.3f} 个百分点。'
             '这支持“共同未读变量的影响可以抵消”的机制，未证明算法相对近年先进方法的总体优势。', '',
             '| 数据集 | 共享固定顺序节省 | 共享自适应节省 | 独立界节省 |', '|---|---:|---:|---:|']
    for n in names:
        m = s['dataset_means'][n]
        text.append(f'| {n} | {(1-m["shared_mi"]["mean_acquired_fraction"])*100:.2f}% | '
                    f'{(1-m["shared_adaptive"]["mean_acquired_fraction"])*100:.2f}% | '
                    f'{(1-m["independent_mi"]["mean_acquired_fraction"])*100:.2f}% |')
    text += ['', '![属性获取对照](acquisition-comparison.png)', '',
             '## 成本结论与失败门槛', '',
             f'固定 MI 主方案节省 {(1-a["shared_mi"]["mean_acquired_fraction"])*100:.2f}%，'
             '未达到预先写下的 20% 扩展门槛，因此按协议不扩大种子和数据集。'
             '自适应方案的较高节省属于预设诊断对照，不能事后替换主门槛并宣布成功。', '',
             f'自适应原型的在线计算时间为完整 1NN 的中位数 {np.median(slowdowns["shared_adaptive"]):.1f} 倍。'
             '属性读得更少，不等于 CPU 运算更少。每一步的认证与策略选择带来了额外计算。', '',
             '在线时间比较采用同一标量查询接口；完整模型重复五次取中位数，条件模型重复三次取中位数。'
             '训练、参考缓存构建及预处理时间单独记录，静态 MI 的批量时间不与标量时间比较。'
             '获取量是已有完整数据上的回调模拟，当前时间不含真实传感器、检测费用或物理采集延迟。', '',
             '若每个属性具有相同的实际获取延迟 t，粗略的平均收支平衡条件为：', '',
             '`t > (条件预测 CPU 时间 − 完整预测 CPU 时间) / 平均少获取属性个数`', '',
             f'自适应方案按 18 个划分计算，门槛中位数 {np.median(thresholds["shared_adaptive"]):.1f} 微秒/属性，'
             f'范围 {min(thresholds["shared_adaptive"]):.1f}–{max(thresholds["shared_adaptive"]):.1f} 微秒。'
             '这是根据测得 CPU 开销推算的场景条件，不是真实采集实验，也不包括训练摊销。'
             '已经全部存入内存的廉价特征不适合这个原型。', '',
             '固定顺序共享缓存为 O(d n²)，本试验最高约 '
             f'{max(r["methods"]["shared_mi"]["cache_bytes"] for r in rows)/1024**2:.1f} MiB。'
             '大样本高维部署仍需重新设计存储与检查策略。自适应方案没有该缓存，但计算更慢。', '',
             '## 理论与审计', '',
             '已写明精确的单敌人/单朋友完成界、充分的多类别停止条件、固定见证下的界单调性，'
             '以及相同顺序下共享界对独立界的包含关系。多朋友的 min/max 交换不精确，'
             '不能将充分条件包装成充要条件。单对界是已有共享三元组定理的特例。', '',
             f'5 项机制测试通过；独立重放 {audit["conditional_query_trajectories"]:,} 条获取轨迹、'
             f'{audit["early_stop_certificates_replayed"]:,} 次提前停止及 {audit["metric_values_checked"]} 个指标值，'
             '全部通过。审计不导入新选择器，手工重建预处理并重新计算停止界和预测。', '',
             '第一次轨迹重放在 Heart 的一个自适应瓶颈选择上出现差异：先除以 d 和先乘以 1/d 的'
             '浮点舍入改变了近似并列的参考对顺序。审计现在对路径重放统一乘法顺序，并用独立直接求和'
             '重新验证每个接受的共享停止界；原始结果与核心算法未改动。不是正式区间算术证明。', '',
             '## 机制压力测试', '',
             '64 维合成任务的第一维为类别信号。其余维度在所有参考对象中相同时，'
             '共享界消掉共同距离影响，只读第一维；独立界仍需读更多维。'
             '当参考对象的其余维度独立随机时，噪声会改变完整 1NN 的几何，条件模型会保留这些错误。'
             '具体全部结果保存在 synthetic-stress.json，不用于替换真实数据的失败门槛。', '',
             '## 后续决定', '',
             '当前成果是可用的机制原型及一个明确的适用条件，不是高水平论文的已成立主贡献。'
             '它面向属性采集昂贵且完整模型值得保留的场景；如果继续坚持普通廉价表格数据上的分类优势，'
             '应停止这条路线，而不能用节省获取量替代未实现的准确率目标。', '',
             '继续这条路线必须先核查自适应属性获取与不完整信息粗糙集的已有工作，再解决认证开销，'
             '引入同等预测保留要求的先进获取策略，并开展真实或有独立依据的获取成本实验。'
             'RFRR、FD+、FCE+ 的正式复现欠缺没有因这个原型而解决。', '',
             '本轮未修改或编译 LaTeX，历史 8 种子数据未覆盖。']
    report.write_text('\n'.join(text) + '\n', encoding='utf-8')
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    x = np.arange(len(names))
    for shift, method, label, color in [(-.25, 'shared_mi', 'Shared bound, MI order', '#3568a8'),
                                       (0, 'shared_adaptive', 'Shared bound, adaptive', '#25856b'),
                                       (.25, 'independent_mi', 'Independent bound, MI order', '#b68038')]:
        saved = [(1-s['dataset_means'][n][method]['mean_acquired_fraction'])*100 for n in names]
        ax.bar(x + shift, saved, .24, label=label, color=color)
    ax.set_xticks(x, ['Glass', 'Heart', 'Ionosphere', 'Sonar', 'WDBC', 'Wine'])
    ax.set_ylabel('Attributes not acquired (%)')
    ax.set_ylim(0, 41)
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Identical full-model predictions; 6 development datasets, 3 folds')
    fig.savefig(DEST / 'acquisition-comparison.png', dpi=180)
    fig.savefig(DEST / 'acquisition-comparison.pdf')
    plt.close(fig)
    inputs = [ROOT / 'research/conditional-completion-v1-protocol.md',
              ROOT / 'research/conditional-completion-v1-theory.md',
              ROOT / 'src/certified_frfs/conditional_completion.py',
              ROOT / 'src/certified_frfs/baselines.py', ROOT / 'src/certified_frfs/datasets.py',
              ROOT / 'src/certified_frfs/__init__.py', ROOT / 'tests/test_conditional_completion.py',
              ROOT / 'scripts/run_conditional_completion_v1.py', Path(__file__),
              ROOT / 'scripts/audit_conditional_completion_v1.py', ROOT / 'requirements-lock.txt',
              ROOT / 'data/manifest.json']
    inputs += [ROOT / 'data/processed' / f'{name}.npz' for name in names]
    inputs += sorted(p for p in DEST.iterdir() if p.is_file())
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest = DEST / 'evidence-sha256.json'
    manifest.write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    archive = ROOT / 'output/conditional-completion-v1-evidence.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for p in inputs + [manifest]:
            bundle.write(p, str(p.relative_to(ROOT)))
    print(json.dumps({'report': str(report), 'archive': str(archive),
                      'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'adaptive_median_slowdown': float(np.median(slowdowns['shared_adaptive'])),
                      'adaptive_break_even_us': float(np.median(thresholds['shared_adaptive'])),
                      'stress': stress}, indent=2))


if __name__ == '__main__':
    main()
