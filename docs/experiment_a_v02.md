# 实验 A v0.2 运行与验证

v0.1 基线是 `5de4372`。v0.2 使用独立分支和 Scratch 快照。缓存四区是回归样本。候选开发区和新留出区的定义在 `configs/experiment-a-v02.toml`。高自行挑战按 Gaia 质量和四个速度区间先划分不同天空域，再用固定散列排序取样。近邻挑战锥保持间距。位置只用于确定取数锥的范围，目标标签单独保存到 `validation`。

## A1

- 时间桶服务于空间搜索。每条轨迹用自身的独立夜、观测组、跨度和形式自行误差评价时间信息。背景密度在单次曝光的 5″ 邻域估计，加入一个伪计数。局部曝光足迹边缘和深度变化仍会影响这项近似。
- 评分累加归一化位置残差相对于局部均匀背景的证据，每条轨迹扣一次复杂度代价。跨曝光误差相关仍需标定。固定代价和置信阈值是待验证的规则。
- 近重复要求运动参数相容和探测集合高度重叠。共享曝光若存在两条不同检测则保留竞争。可以合并时重新拟合并保留映射。时间限额下通过整数性和占用约束检查的解单列为暂定。每个分量记录大小、状态、耗时和最优性差距。 [SciPy MILP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
- 恢复要求无矛盾对象标签、每巡天至少 80% 的合格检测、最早 SDSS 夜和每个关键时间组。报告候选、最终选中、高置信、歧义及计算不可决。对整体污染用已知和未知轨迹给出辨识界，已核验轨迹的 Wilson 区间只反映该样本下的描述性不确定度。规模化统计按区域重采样。
- 整年留出在关联前移除该年。按 NSC-only/跨巡天、Gaia/无 Gaia、NSC r 波段深度分别统计。无 Gaia 的预测一致需同时满足位置误差、同波段光度、单一竞争轨迹和低偶然率。它仍属于未独立证实的预测。
- 近距真值采用 Gaia 距离、有值时的径向速度及地球年视差，在独立验证流程中生成。A1 关联继续采用线性模型，近距轨迹可能因视差和未知加速度漏检。单独报告这类样本。

## 注入

NSC 使用发布的芯片顶点，SDSS 使用 DR17 扫描坐标范围及 TAI。注入先统计几何可观察，再按巡天的显式抽样概率、经验位置误差和时间覆盖进入回收。几何掩膜未包含芯片内部缺陷、饱和、混叠和真实检出曲线。负对照在共同裁剪范围内做多个方向和幅度的平移。应按参考历元与调查端点计算最大位移、视差和边缘余量。高速背景需要按相同边界重新取得。 [SDSS 坐标变换](https://www.sdss4.org/dr16/algorithms/astrometry/) · [NSC 芯片字段](https://datalab.noirlab.edu/data-explorer?showTable=nsc_dr2.chip)

## A2

锁定 Gaia 确认关联。只在 NSC 对象编号一对一且发布自行有效的同一批源比较四种方案，另外保留较宽的三路控制。逐 SDSS run 移除观测，计算位置和 PM 改变量，位置按 run/camcol 聚合。共享方差敏感度用 SDSS run 与 NSC 夜为共享组，分别比较 25、50、100 mas 的独立误差底限和 25、50 mas 的共享偏移尺度。这些是敏感度参数，未经实测标定。回归四区的端到端结果由独立 A1 作业生成。

## 运行

- 已缓存四区用 `bash scripts/experiment-v02.sh smoke wrap north pal5`。先设 `DEEP_PM_WORK` 和 `DEEP_PM_FROZEN_V01` 到不同 Scratch 目录。
- A2 在单独的 Scratch 根用 `bash scripts/precision-v02.sh pal5 smoke wrap north`。输出 `paired_control_summary.json` 的统计可离开 NERSC，Gaia 标签和 Data Lab 缓存保留在远端。
- 回归完成后才运行 `bash scripts/independent-v02.sh dev_repeat test_north test_equator test_crowding`。高自行挑战先用 `python -m deep_pm.challenge_sample "$DEEP_PM_WORK"` 固化分档目标，再对其生成的区域清单分别运行；`DEEP_PM_CONFIG` 指向冻结后的 Scratch TOML。
- 无 Gaia 留出与平移使用 `python -m deep_pm.counterfactual <region-workdir> --region <region>`。真实几何注入用 `python -m deep_pm.injection <region-workdir> --region <region> --sdss-window <window_flist.fits> --trials 30`。
- 阶段程序须在 NERSC CPU 作业中执行。程序检查运行提交、配置和输出校验值；目录不复用跨提交的同名产物。逐检测、逐源、轨迹及 Gaia 文件留在 Scratch。绘图可用脱敏聚合统计离线生成。

当前阶段状态与待验证项见 [进度报告](../reports/experiment_A_v0.2.md)。

## 核验与同步

在本地为 `scripts/sync-v02-summary.sh` 设置 `DEEP_PM_SSH_TARGET`、`DEEP_PM_REMOTE_PROJECT`、`DEEP_PM_REMOTE_RUN` 和 `DEEP_PM_EXPORT_REVISION`。最后一项是已部署的汇总导出代码提交，可以独立于实际计算提交。用 `bash scripts/sync-v02-summary.sh association smoke` 或 `bash scripts/sync-v02-summary.sh precision pal5` 拉取已结束阶段的聚合统计。导出核对配置与产物校验值，检查阶段之间的输入一致性，并把汇总的代码版本、配置校验和、产物校验和与 Slurm 作业号写入回执。旧缓存回归的角色从冻结配置补齐。图表可从这些聚合数据重绘。

高自行挑战只按开发、留出和自行档导出。每个锥对应一个冻结目标。指向主挑战或补充挑战的 Scratch 根，运行 `bash scripts/sync-v02-summary.sh challenge-results challenge`。导出核对冻结配置和目标表、每个已完成场实际验证的目标、以及分析版本。汇总记录冻结数、有完整报告数及合格数，只有每个冻结目标都完成报告才保存到本地。区分合格样本的关联完整度与整个冻结清单的实际产出。
