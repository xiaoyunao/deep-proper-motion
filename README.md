# deep-proper-motion

SDSS+NSC DR2 多历元自行可行性实验 A。**实验性研究代码，尚未达到正式自行星表的验证与发布标准。**

- [项目主线与当前证据](PROJECT_CONTEXT.md)
- [实验 A v1.2 连续区最终报告](reports/experiment_A_v1.2_continuous.md) · [复现命令](docs/continuous_c0_v12.md) · [配对精度图](figures/v12_continuous_paired_precision_summary.png)
- [实验 A v0.8 中文报告](reports/experiment_A_v0.8_spatial.md) · [空间验证协议](docs/spatial_validation_v08.md)
- [实验 A v0.7 中文报告](reports/experiment_A_v0.7_regional.md) · [实验 A v0.6 中文报告](reports/experiment_A_v0.6_astrometry.md) · [实验 A v0.1 中文报告](reports/experiment_A_v0.1.md)
- [科学问题](docs/science_case.md) · [数据与访问](docs/data_access.md) · [单位与选择](docs/selection_and_units.md) · [方法](docs/methods.md) · [验收协议](docs/validation_protocol.md)
- [可公开的汇总](results/summary) · [统计图](figures) · [研究记录](WORKLOG.md)

实验 A v0.2 的独立验证、注入和同源精度对照见 [中文报告](reports/experiment_A_v0.2.md)、[八步计划](docs/experiment_a_v02_plan.md) 和 [v0.2 流程](docs/experiment_a_v02.md)。以下入口保留实验 A 的基本操作。请用明确的阶段版本配置和独立 Scratch 根。

使用 Python 3.8+，依赖见 `pyproject.toml`。在安装后的环境中用 `deep-pm`，也可设 `PYTHONPATH=src` 并调用 `python -m deep_pm`。NERSC 已实测 Python 3.8、NumPy 1.24、SciPy 1.10、Astropy 5.2、scikit-learn 1.3，Python 3.8 需要额外的 tomli。

从项目根目录运行。登录节点只做环境探查、数据管理及提交。先用 `deep-pm --work "$DEEP_PM_WORK" discover` 和 `sacctmgr` 确认实际 HOME、Scratch 和 CPU 资源账户，再提交。参数 `DEEP_PM_WORK` 指向已验证的个人 Scratch 项目目录，`DEEP_PM_PYTHON` 与 `DEEP_PM_EXTRA_PYTHONPATH` 随实际环境设置。区域执行时使用同一冻结配置和代码快照：

```bash
export DEEP_PM_WORK="${PSCRATCH:?}/deep-proper-motion"
export DEEP_PM_PYTHON=python
export PYTHONPATH="$PWD/src"
mkdir -p "$DEEP_PM_WORK/logs"
sbatch -A "${DEEP_PM_ACCOUNT:?set a verified CPU account}" -q "${DEEP_PM_QOS:?}" \
  -C cpu -N 1 -n 1 -c 2 -t 00:25:00 -o "$DEEP_PM_WORK/logs/%j.log" \
  --wrap='set -e; for r in sdss object gaia; do "$DEEP_PM_PYTHON" -u -m deep_pm --route "$r" fetch; done; bash scripts/experiment.sh smoke'
```

`scripts/experiment.sh` 依次执行 `normalize → inventory → associate → fit → validate → report`。先确认已有 `sdss.csv`、`nsc_object.csv` 和分离的 `validation/gaia.csv`。NSC CCD 路线可用 `fetch --route exposure` 执行。八个曝光的路线审计可在已有 object 路线数据的计算作业里用 `python -m deep_pm.audit "$DEEP_PM_WORK/smoke" --ra 30 --dec 0 --radius 0.037` 重跑。其他研究入口为 `deep_pm.counterfactual`、`deep_pm.diagnostics`、`deep_pm.injection`、`deep_pm.precision`、`deep_pm.exhaustive`、`deep_pm.solver_ablation`、`deep_pm.audit_plots`。

测试用 `PYTHONPATH=src python -m unittest discover -s tests -v`。公开同步用本地 `bash scripts/sync-summary.sh`，先设置 `DEEP_PM_SSH_TARGET` 和 `DEEP_PM_REMOTE_SCRATCH`。该入口只接收预设名称的统计和图片。数据保留与持久化见 [数据文档](docs/data_access.md)。

实验 A v0.3 从 `b92912c` 单列普通源精度实验。用 `bash scripts/precision-v03.sh test_equator` 在 Slurm 上从冻结目录复现基线并检验独立星系零点。配置和验收口径见 [v0.3 计划](docs/experiment_a_v03_plan.md)，实测结果见 [v0.3 中文报告](reports/experiment_A_v0.3_precision.md)。

用版本化 Slurm 工作目录运行 `deep_pm.precision_inputs` 获取精确曝光数据及检查出版 FITS 首部，再用 `deep_pm.chromatic_control`、`deep_pm.dark_precision` 运行色项消融和整年留出。审计产物用 `scripts/sync-v02-summary.sh` 逐阶段校验回执后同步。


实验 A v0.4 用 `bash scripts/nsc-v04.sh bright` 严格复核冻结亮星，`inventory` 从 DESI DR1 光谱表与 SDSS、NSC 位置预查覆盖，`qso` 按冻结 M3 校验选择后的区域。三个模式均需给出区域参数，在 Slurm 计算节点执行。验证样本与版本记录见 [项目主线](PROJECT_CONTEXT.md) 和 [v0.4 中文报告](reports/experiment_A_v0.4_nsc.md)。


v0.5 区域流程依次调用 `deep_pm.qso_fields inventory`、`deep_pm.pilot_inputs`、`deep_pm.pilot_production`。用 `--config configs/production-v05.toml` 传给 v0.4 的预查入口，并可用 `--sdss-sas` 指定验证过的 SDSS DR17 SAS 镜像。给试生产阶段独立版本化的 Scratch 目录。用 `deep_pm.pilot_cohort --pilot … --stripe …` 核对本轮独立观测块。正式发布范围和验收规则见 [v0.5 计划](docs/plan_v05.md)。

v0.6 在 Slurm 计算节点复用一个已标准化的区域缓存。输入目录须含 `detections.csv`、`external_labels.csv`、`spectroscopic_qso.csv` 和独立的 `validation/gaia.csv`，同时传入 SDSS DR17 的 `window_flist.fits`。

```bash
python -m deep_pm.astrometry_audit "$DEEP_PM_RUN" g09 "$DEEP_PM_EXISTING_INPUT" "$DEEP_PM_SDSS_WINDOW"
```

公开同步使用 `bash scripts/sync-v02-summary.sh astrometry-v06 g09` 校验回执。用 `python -m deep_pm.astrometry_plots results/summary/v06_*_astrometry_v06_*_summary.json` 重绘公开图。 [v0.6 计划](PLAN.md) · [误差与时间审计](docs/astrometry_v06.md)

v0.7 先在 CPU Slurm 节点用 `deep_pm.regional_preflight` 核验覆盖，再运行 `deep_pm.regional_inputs` 与 `deep_pm.regional_production`。已有核对过的 NSC/SDSS 数据或下载缓存可从原版本目录复用。独立使用 `deep_pm.dense_archive --checkpoint` 生成首个 COSMOS 分片，再从同一版本化目录运行续取。预先锁定的区域与备选见 [v0.7 范围](docs/regional_v07_scope.md)。输入和输出均置于 NERSC Scratch。

在已验证的 CPU compute job 内，从版本化代码根目录运行以下示例。每个并发区域需要独立留出 CPU、内存和档案服务时间。

```bash
export DEEP_PM_INPUTS="<versioned-scratch-input-root>"
export DEEP_PM_OUTPUTS="<versioned-scratch-production-root>"
export DEEP_PM_SAS="<verified-dr17-eboss-sas>"
export DEEP_PM_QSO="<verified-desi-dr1-qso-fits>"
bash scripts/regional-v07.sh north_a single_e
```

`deep_pm.regional_cohort` 通过多个 `--region` 读取带回执的完成目录，`--previous` 接收历史区域目录以核查共享观测。公开汇总经 `scripts/sync-v02-summary.sh regional-pool-v07 regional_pool` 验证，并由 `deep_pm.regional_plots` 生成图。

### v0.8 分节点扩区

候选与入选范围分别见 [空间配置](configs/spatial-v08.toml) 和 [冻结矩阵](configs/spatial-v08-fields.toml)。在 NERSC CPU Slurm 分配内使用 scripts/spatial-v08.sh 的 acquire 或 reduce 模式，将每个 task 配置在独立节点。DEEP_PM_NODE_TASKS 为当前节点数，DEEP_PM_WORKERS 为各节点内的独立区域进程数，其他输入输出路径通过同一脚本的环境变量设置。

取数重试沿用原提交、配置、目录和缓存。逐检测、逐源、自行配对及留出数据保留在 NERSC Scratch。允许列表脚本可导出阶段摘要，deep_pm.spatial_figures 从摘要复现图。

v0.9 将 M3 与曝光／探测器级位置零点放到同一批检测上比较，先运行连续区域内的两个独立小试。冻结区域、试点阈值与可恢复的 NERSC 取数和拟合入口见 [v0.9 复现说明](docs/exposure_v09.md)。

v1.0 在同一框架内运行 `deep_pm.exposure_pilot --validation-audit`，应用范围、冻结区域及亮 Gaia 协议见 [计划](PLAN.md)。在相同输入回执下可添加 `--sensitivity` 复核细组星系数量和父组约束。新增区域以 `configs/validation-v10.toml` 取数和合并，通过 `--sdss-sas` 检查相邻 field 并保存用于观测块统计的私有回执。`deep_pm.validation_shared` 对北部两个核心合并检测后拟合一次参考图。`deep_pm.validation_plots` 从本地公开汇总生成图。

### 实验 A v1.1 复现

[v1.1 计划](docs/v11_completed_plan.md)与 [重叠模型及运行说明](docs/overlap_validation_v11.md)记录输入回执、物理观测块、扩区规则和可重放命令。exposure_pilot 在 overlap-v11.toml 配置下额外运行 C1b。overlap_blocks 在 NERSC 合并 QSO 验证块，overlap_plots 从公开汇总绘图。
