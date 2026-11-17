# v0.9 冻结范围与分层比较

三个矩形只依据档案和几何选择。C（RA 345°–355°、Dec −5°–+5°）的均匀抽样双巡天覆盖为 100.0%，且 92.85% 同时具有至少 0.60 年 NSC r 跨度。北侧时间支持有明显缺口，保留遮罩。选择 C 还利用了已读回的 Xiao2025 地基位置和 DR16Q/DESI 光谱样本；未读取本轮自行。目标区域球面面积 99.873 平方度。

- 重复丰富小区 RA 353°–355°、Dec −1°–+1°。抽样位置均有至少两年 SDSS 重复跨度及达标的 NSC r 跨度，SDSS 扫描数中位数 18。选择五个充分重复格里估计测量数最低者，约 470 万条 NSC 归档测量。
- 稀疏小区 RA 345°–347°、Dec −5°–−3°。抽样位置无两年以上 SDSS 重复，扫描数中位数为 1，95.25% 有双巡天及有效 NSC r 跨度。选取与重复区在 RA、Dec 均分离且时间支持较好的经济小区。
- 两处都保留 0.06° 缓冲。HEALPix NSIDE=128 RING 用于组织和诊断。C0/C1 在每处试点共用一张切平面和物理观测组。试点间分别估计观测图与零点，全区阶段需共同求解后再分块应用。

C0 沿用 v0.8 M3 的位置组、星系参考、W0 权重与 J2015.5 位置报告历元。C1 在相同训练对象上联合解二维平移，最少 15 个不同参考星系支持细组，子组向同一物理父组施加固定 50 mas 差值约束，弱组使用同一解中的父组。按训练星系共享重采样，误差传到自行，留出星系和全部独立光谱 QSO 仅在验证中使用。两个模型的严格对照使用相同检测与 NSC r 时间支持；盲对照使用共同固定位置关联及各模型通过质量筛选的解。

两个试点均需显示留出星系的跨巡天位置径向 P68 至少改善 5%，对象内配对区间支持改善。严格同源光谱 QSO 至少 20 个交叉支持对象，C1 相对 C0 的径向 P68 配对区间须排除超过 10% 的劣化，并复查两轴零点、尾部、可识别的 detector 或空间边界变化。对象重采样区间条件于本场观测序列，跨场推广还取决于独立物理组重复。正式高精度目标仍为每轴 ±0.5 mas/yr。当 C1 在共同支撑集上未有稳定独立收益时保留 C0，报告试点结果及测量限制；C2–C4 只增加有独立残差依据的项。已有的色差和 DCR 负结果见 [v0.3](../reports/experiment_A_v0.3_precision.md)。

## 重放入口

在对应 git 版本和安装了项目依赖的环境中运行。目录通过本机环境变量传入。`DEEP_PM_M3_WORKERS` 控制每个节点内的星系重采样并行度。取数按五个 `parts` 分配，分别在独立目录续跑。合并与拟合须等该试点全部分片通过回执校验。每个作业应限制 BLAS 线程，以免与分片或重采样并行竞争。

```bash
cfg=configs/exposure-v09-fields.toml
PYTHONPATH=src python -m deep_pm.exposure_partition "$cfg"

# Within the compute allocations, with v09_root, v09_catalogues, and v09_sas set locally.
python -m deep_pm.exposure_inputs "$v09_root/parts" "$part" "$v09_catalogues" "$v09_sas" "$cfg" --workers 6
python -m deep_pm.exposure_merge "$v09_root/inputs" "$pilot" "$v09_root/parts" "$cfg"
python -m deep_pm.exposure_pilot "$v09_root/output" "$pilot" "$v09_root/inputs" "$v09_catalogues/xiao_c.csv" "$cfg"
python -m deep_pm.exposure_coverage "$v09_root/inputs"

# Locally, after sync-v02-summary.sh has exported the public pilot receipts.
PYTHONPATH=src MPLBACKEND=Agg python -m deep_pm.exposure_plots
```
