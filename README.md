# deep-proper-motion

SDSS+NSC DR2 多历元自行可行性实验 A。**实验性研究代码，尚未达到正式自行星表的验证与发布标准。**

- [项目主线与当前证据](PROJECT_CONTEXT.md)
- [实验 A v0.1 中文报告](reports/experiment_A_v0.1.md)
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


v0.5 区域流程依次调用 `deep_pm.qso_fields inventory`、`deep_pm.pilot_inputs`、`deep_pm.pilot_production`。用 `--config configs/production-v05.toml` 传给 v0.4 的预查入口，并可用 `--sdss-sas` 指定验证过的 SDSS DR17 SAS 镜像。给试生产阶段独立版本化的 Scratch 目录。用 `deep_pm.pilot_cohort --pilot … --stripe …` 核对本轮独立观测块。正式发布范围和验收规则见 [v0.5 计划](PLAN.md)。
