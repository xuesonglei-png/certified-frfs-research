"""Write a report and portable archive only after successful complete replay audit."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'output/iteration5-reproduction-v1'


def main():
    a=json.loads((BUNDLE/'runs/rerun-001/audit.json').read_text())
    assert a['passed'] and a['completed_splits']==42
    assert all(v is True for v in a['original_workspace_checks'].values())
    means=a['means'];stats=a['statistics']
    rows=[]
    for name,r in means.items():
        rows.append(f"| {name} | {100*r['balanced_accuracy']:.6f}% | {100*r['removed_fraction']:.6f}% | {100*r['balanced_training_coverage']:.6f}% |")
    report=f'''# 第 5 轮独立复现核验报告

结论：完整复现通过，原始结果未被覆盖。

- 14 个数据集 × 3 个外层折 = 42 个划分；全部重新训练和内层调参。
- 8 种方法，共 336 条方法/划分结果；不是读取历史选中属性后只重算预测。
- 核验选中属性、最终配置、所有内层候选评分、干净和扰动分类指标、完整认证记录。
- 比较 {stats['numeric_comparisons']:,} 个浮点值，非完全相同的值为 {stats['nonexact_float_values']} 个，
  最大绝对差为 {stats['max_absolute_difference']}；判定容差为 1e-12。
- 原工作区 {len(a['original_workspace_checks'])} 个结果文件的 SHA-256 全部保持一致。
- 隔离代码相关测试：29 项通过。
- 本次 3 个进程完整复跑墙钟时间约 {a['run_wall_seconds']/60:.2f} 分钟；
  该数字仅供复现预算参考，不是方法效率比较实验。

| 方法 | 平均平衡准确率 | 平均删除属性比例 | 平衡训练认证覆盖 |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

这些平均值先按数据集平均三个外层折，再对 14 个数据集等权平均。
完整逐数据集指标在 `runs/rerun-001/audit.json`，新逐折结果在同目录。

## 代码来源与变动核对

未恢复历史整体 SHA-256 对应的完整源码快照。本包明确采用当前可用实现的冻结副本。
与已有较晚开发 ZIP 比较，179 个 Python 文件中，37 个一致、141 个不在该 ZIP 内、
1 个 `validated_pool.py` 有变动。本轮执行路径涉及的核心模块和两个原运行脚本均一致；
有变动的模块不在该路径上。14 个处理后数据校验值及六个核心依赖版本与该归档一致。
详见 `source-audit.json`、`execution-path-audit.json`、`data-and-environment-audit.json`。

独立运行器只提取原第 5 轮调度逻辑，限定八个实验组；使用保存的外层索引，
将按原种子重建的内层索引显式冻结；新输出目录拒绝覆盖。预处理、选属性、评分、
认证函数来自包内冻结代码。外层分数仅在重算完成后用于审计，不用于修改方法或调参。

## 可以据此说什么

可以确认该历史开发结果可重复：平均删除约 30.61% 属性，BA 为 78.16%，
全属性为 77.99%。原先关于这些数值的描述有可检查的代码、参数、数据和逐折结果支撑。

不能据此升级为独立确认、显著预测优势或统计非劣。此次使用相同历史数据和划分，
目的是复现；没有新增独立泛化证据。完整原始源码身份未恢复与行为完全复现是两件事。
删除属性比例也不等于运行时间节省比例。

运行方式见 `README.md`。后续实验应从此包复制新版本，并使用新的输出目录。
'''
    with (BUNDLE/'verification-report.md').open('x',encoding='utf-8') as f:f.write(report)
    paths=sorted(p for p in BUNDLE.rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
    checks={p.relative_to(BUNDLE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with (BUNDLE/'package-sha256.json').open('x',encoding='utf-8') as f:json.dump(checks,f,indent=2)
    paths.append(BUNDLE/'package-sha256.json')
    dest=ROOT/'output/iteration5-reproduction-v1.zip'
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,arcname='iteration5-reproduction-v1/'+p.relative_to(BUNDLE).as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for name,h in checks.items():
            assert hashlib.sha256(z.read('iteration5-reproduction-v1/'+name)).hexdigest()==h
    checksum=hashlib.sha256(dest.read_bytes()).hexdigest()
    with dest.with_suffix('.zip.sha256').open('x',encoding='utf-8') as f:f.write(checksum+'  '+dest.name+'\n')
    print(json.dumps({'archive':str(dest),'bytes':dest.stat().st_size,'sha256':checksum,
                      'files':len(paths),'zip_verified':True},ensure_ascii=False))


if __name__=='__main__':main()
