# v1.1 重叠 field 零点模型

C0 连结 SDSS run/camcol 与 NSC exposure。C1 通过背景星系共同解出 SDSS field 及 NSC CCD 的细组零点，采用 15 个训练参考源的最小支撑与 50 mas 的细组–父组收缩。C1b 保留同一星系位置参数、C1 网络和参考系，再给 C1 支持的相邻 SDSS field 加入相对位置约束。

从 SDSS DR17 SAS 的逐场 photoObj 读取相邻 field，保留 RUN_PRIMARY 过滤前的重复记录。对同一个 run 和 camcol 的相邻 field，以星系形态、无子源、坐标取自 r、距离 ≤500 mas 及 300 mas 的竞争间隔筛选。训练、留出沿用 C0/C1 参考星系划分，同一 NSC 参考源及其所有 SDSS 重复位置一起划分。各段优先读取候选多的至多 160 个邻场。训练额外过滤多对一竞争及无有效位置误差的记录。 [SDSS DR17 Resolve](https://www.sdss4.org/dr17/algorithms/resolve/)

约束的观测量是 SDSS 相邻 field 对同一星系的东、北位置差，约束同两组零点之差。误差使用两位置误差的平方和及 20 mas 地板。一个星系在同一物理扫描的边总权重至多为 1。拟合稳健降权，48 次重采样中星系主位置与差分共用同一星系权重。相邻 field 可共享像素，并复用进入主拟合的 SDSS 位置。地板、权重上限和共享重采样近似处理相关性，保留留出检验和物理扫描级区间。新增差分只进入星系标定，恒星自行继续用去重后的独立观测。逐像素协方差未估计。

拟合在各场使用局部切平面和共用 C0 NSC 父组对齐，保留 J2015.5。HEALPix NSIDE 128 用于输出和检查。跨巡天位置比较按留出星系分组，在同一批有效检测上求 SDSS 和 NSC 的中位位置差。暗端 QSO 自行检验在 22≤r<23 优先。三种标定固定相同观测，另对同时有官方 NSC 和 Xiao2025 地基自行的对象做五路比较。留出预测按整年重建训练轨迹，测试年星系可给测试曝光定标。

按共享 SDSS 物理 run、NSC exposure 联合成块。保守版本再按接近 UTC 正午切换的 NSC 观测日联结，避免同夜曝光被当成独立长期证据。独立块太少时报告描述统计。场内对象重采样区间用于诊断，扩区用物理块。边界用独立 SDSS run 重采样。每场严格亮 Gaia 只按 NSC r<21 和已冻结的高质量规则取样。暗端采用光谱 QSO 和恒星独立年份位置留出。

## 命令入口

在 NERSC 已分配的 Slurm CPU 节点运行。用实际路径设置 IN（含各场原始输入回执）、OUT、LEGACY（Xiao 裁剪 CSV）、SAS（SDSS SAS 镜像）、AUDIT（私有诊断输出）。REGION 选冻结的验证核心。建议每场使用一个 CPU 节点，将线性代数库限制为单线程，按参考星系重采样并行。

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEEP_PM_M3_WORKERS=12
python -m deep_pm.exposure_pilot "$OUT" "$REGION" "$IN" "$LEGACY" configs/overlap-v11.toml \
    --validation-audit --sdss-sas "$SAS"
python -m deep_pm.overlap_edge_audit "$AUDIT" "$IN/$REGION" "$SAS" configs/overlap-v11.toml "$REGION"
```

额外普通场以 deep_pm.overlap_inputs 从缓存圆形父区、已核验的 QSO 母表和本地 FITS 建立冻结矩形裁剪。四个 v1.0 场用原获取回执，旧场 Gaia 观测期字段复用 v1.0 的服务缓存。C0/C1 基线的原版本可用 deep_pm.validation_existing 重放。

底层的逐源、逐组和留出明细留在 NERSC。对每场用 bash scripts/sync-v02-summary.sh 校验回执、白名单及脱敏属性。合并用 python -m deep_pm.overlap_blocks --help，输出仅有场名、样本量、共享组计数和统计值。

```bash
PYTHONPATH=src python -m deep_pm.overlap_plots results/summary figures
```
