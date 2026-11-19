# v1.5 可重复执行入口

实际数据、代码快照、配置绝对路径和每次Slurm的`worker.py`／`run.sh`／stdout／stderr均在私人PScratch。原始和逐源结果不复制到本地。公开的本轮模板方法配置仍为`configs/precision-v03.toml`、`configs/spatial-v08.toml`和`configs/continuous-v12.toml`；实际路径通过未提交的个人配置提供。

以下命令**仅在已获得Slurm计算节点的终端**执行。`DPM_*`变量由私人配置填写；不要重用HOME变量。分配账号、QOS与机器资源复用已经核验的本机配置，不在公开模板里猜值。

```bash
export PYTHONPATH="$PWD/src:$DPM_RUNTIME"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
: "${SLURM_JOB_ID:?必须在Slurm计算节点}"

# 极小完整曝光路线；同一目录绑定源码commit、原输入和物理参考。
python -m deep_pm.nsc_exposure_complete \
  "$DPM_PROOF" "$DPM_PARENT_INPUT" "$DPM_SHARED" "$DPM_GEOMETRY" "$DPM_REGION" \
  --workers 2

# 无r对象切割取数；只对位于同一父core内的proof使用--exposure-audit。
python -m deep_pm.nsc_allband_inputs \
  "$DPM_INPUT" "$DPM_PARENT_INPUT" "$DPM_GEOMETRY" "$DPM_REGION" \
  --workers 1 --exposure-audit "$DPM_PROOF"

# 科研盲目录：原r／物理r／全NSC对照，所有源先冻结再查光谱。
python -m deep_pm.nsc_science_catalogue \
  "$DPM_FIT" "$DPM_INPUT" "$DPM_SHARED" "$DPM_GEOMETRY" "$DPM_XIAO" "$DPM_REGION"

# 固定旧光谱成员的信息实验，不用于筛选有利目标。
python -m deep_pm.nsc_information qso "$DPM_ABC" "$DPM_FROZEN_QSO_TRIAL"
python -m deep_pm.nsc_band_information "$DPM_BAND" "$DPM_FROZEN_QSO_TRIAL"

# 全年在关联前隔离；B固定2017、北一固定2018，C代表区元数据选2017。
python -m deep_pm.allband_blind_year \
  "$DPM_YEAR" "$DPM_INPUT" "$DPM_SHARED" "$DPM_GEOMETRY" "$DPM_REGION" \
  --held-year "$DPM_FROZEN_YEAR"
python -m deep_pm.allband_member_audit "$DPM_MEMBERS" "$DPM_YEAR"
python -m deep_pm.nsc_discovery_replay "$DPM_REPLAY" "$DPM_OLD_YEAR" "$DPM_YEAR"

# 全连续计划在逐core完成后汇总；读取私人逐源向量，输出只含汇总。
python -m deep_pm.nsc_science_summary \
  "$DPM_RUN" "$DPM_GEOMETRY" "$DPM_RUN/continuous_science_summary.json"
```

`nsc_allband_inputs`保存完整下载缓存和每分块COUNT核验，原SDSS r位置从已签名标准化表保留，不从缺失的旧原光度文件补造。每个run目录只能对应同一源码commit／配置／输入；更改方法时新建版本目录，可复用只读旧输入，不覆盖旧回执。

全C实际执行冻结原91个core及其半开归属，并在`private_frozen_plan.json`记录原sources／fits／参考图、Xiao及几何SHA。普通完整输入／科研先导成功后四节点运行，每节点两路，外部服务并发至多四路。科学控制使用原全C共同物理组C0，所有核心引用相同NPZ；无支持真实组保留库存，不为增加band或源数独立重标定。

`nsc_science_summary`按实际个体QSO向量汇总、严格共同源及观测组敏感性；不平均core的P68。名义面积、几何时间机会和目标实际SNR／关联／零点认证分开。图由公开去标识汇总生成：

```bash
python -m deep_pm.nsc_information_figures results/summary figures/v15_information_dcr_error_summary.png
python -m deep_pm.nsc_dark_figures results/summary/v15_complete_dark_validation_summary.json figures/v15_complete_dark_source_summary.png
```

本地只用这些公开汇总作图。同步前对每份JSON执行`deep_pm.public_summary.check_aggregate`，SCP使用明确的`*_summary.json`允许列表，不复制整个远端目录；通过审核后与源码、PLAN、WORKLOG、PROJECT_CONTEXT一并提交。

连续区全部科学分片完成后，`DPM_RUN`应指向经过五核心科学重放的合并索引，而不是原失败驱动目录。汇总代码版本与拟合代码版本分别记录；不得把新汇总源码SHA冒充原拟合SHA。原v1.2 QSO档案缺坐标／成员／ICRS误差时，历史控制仅能做明确标注的唯一光度+曝光集合标量输出QA，歧义保留，不按PM相近配对，也不据此产生精度真值。

```bash
# 本地只读取公开汇总；不需要逐源数据。
PYTHONPATH=src python -m deep_pm.nsc_continuous_figures \
  results/summary/v15_continuous_science_summary.json figures

# 本地完整测试；线程上限同时避免本机原生sklearn并行运行故障。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python -m pytest -q
```

科研目录分片和完整ICRS协方差的具体定义见[读取说明](catalogue_layout_v15.md)。未知夜相关与空间零点质量位必须随结果保留；不能把研究解数量转写成经过纯度认证的恒星数。

## 公共历元归属和连续控制消融

原`DPM_ANCHOR_RUN/fits.json`是来源任务索引；配置还含未入该计划的候选／子区，不能把全部fields当owner。若原连续统计在保存汇总后报告归属不符，先核查，不吞掉其他失败。以下是本轮所有原记录只读审计的公开模板，实际私人`worker.py`／日志亦留在运行目录。`DPM_OWNER_AUDIT`仍为个人PScratch，含逐源失败记录，不可同步。

```python
import gzip, json, os
from pathlib import Path
from deep_pm.__main__ import tomllib
from deep_pm.continuous_reference import verify_directory
from deep_pm.exposure_catalogues import inside
from deep_pm.io import digest, save_json
from deep_pm.nsc_ownership_overlay import STANDARD, CONTROL, active_specs
from deep_pm.public_summary import check_aggregate

assert os.environ.get('SLURM_JOB_ID')
state = Path(os.environ['DPM_ANCHOR_RUN'])
out = Path(os.environ['DPM_OWNER_AUDIT'])
geometry = Path(os.environ['DPM_GEOMETRY'])
fits = json.loads((state / 'fits.json').read_text())
specs = active_specs(tomllib.loads(geometry.read_text()), fits)
out.mkdir(parents=True, exist_ok=True)
bad = []
for core, folder in fits.items():
    folder = Path(folder)
    verify_directory(folder, (STANDARD, CONTROL))
    for name in (STANDARD, CONTROL):
        with gzip.open(folder / name, 'rt') as stream:
            records = json.load(stream)
        for slot, row in enumerate(records):
            if not inside(*row['position_deg'], specs[core]):
                owners = [k for k, spec in specs.items()
                          if inside(*row['position_deg'], spec)]
                bad.append(dict(core=core, file=name, slot=slot, record=row,
                                common_epoch_owners=owners))
proof = dict(source_manifest_sha256=digest(state / 'fits.json'),
             outside_by_type={name: sum(r['file'] == name for r in bad)
                              for name in (STANDARD, CONTROL)},
             affected_cores=sorted({r['core'] for r in bad}),
             outside_full_region=sum(not r['common_epoch_owners'] for r in bad),
             ambiguous_owners=sum(len(r['common_epoch_owners']) > 1 for r in bad),
             slurm_job=os.environ['SLURM_JOB_ID'])
save_json(out / 'private_ownership_failures.json', bad)
check_aggregate(proof)
save_json(out / 'ownership_failure_summary.json', proof)
```

只有内部唯一归属的旧记录才能重分配；外部／多owner、改变成员／状态、原已有竞争近邻均拒绝，保留诊断另查。当前5全band+2r全部是内部转移，未删除任何科研解：

```bash
python -m deep_pm.nsc_ownership_overlay \
  "$DPM_RUN" "$DPM_ANCHOR_RUN" "$DPM_BASE_STATS" \
  "$DPM_OWNER_AUDIT" "$DPM_GEOMETRY"

# 从已完成科学控制冻结真实成员，运动和误差须逐值重现。
python -m deep_pm.nsc_science_ablation \
  "$DPM_FIXED_C" "$DPM_SCIENCE_CORE" "$DPM_GEOMETRY" "$DPM_REGION"
python -m deep_pm.nsc_information qso "$DPM_TIME_C" "$DPM_FIXED_C"
python -m deep_pm.nsc_band_information "$DPM_BANDS_C" "$DPM_FIXED_C"
```

C普通／重复控制是原已读数据的解释，不称新盲验证；每band消融只比较相同覆盖源，不用新PM挑有利样本。当前图入口在同目录存在`v15_C_information_summary.json`时另生成实际C夜／band图。

覆盖率及尾部的描述性Wilson95用已有`deep_pm.validation.ratio(k,n)`，其中`k=round(fraction*n)`且核验回算比例精确一致。它是对象独立近似，不能替代公共观测组敏感性；原始比例／分母和输入汇总SHA均保留在`v15_continuous_error_coverage_summary.json`。快照checkout完成或加锁后再启动计算，不能只因目录存在就并发使用尚未填充的工作树。


固定v1.2地理分层的读出与单／多run对照不能混为一谈。没有新拟合，须先复现全区N／P68；当前原普通21–22相对Xiao的组权区间仍跨零：

```bash
python -m deep_pm.nsc_geographic_readout   "$DPM_GEOGRAPHIC" "$DPM_ANCHOR_RUN"   "$DPM_RUN/continuous_science_summary.json" "$DPM_GEOMETRY"
```


用于在线审阅的精简统计原样复制已有数值并绑定完整汇总SHA；不再拟合或改变样本：

```bash
PYTHONPATH=src python -m deep_pm.nsc_public_overview results/summary   results/summary/v15_science_overview_summary.json
```
