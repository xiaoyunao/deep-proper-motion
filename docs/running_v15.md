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
