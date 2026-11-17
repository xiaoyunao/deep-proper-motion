# 连续 C 区的 C0 试运行

C 区为 RA 345°–355°、Dec −5°–+5°。几何核心使用半开区间，坐标范围和物理观测身份在配置中冻结。既有 14 个 1° 格可由四个缓存足迹覆盖，东南东半的两个 1° 输入保留各自回执。其余核心分批取数。所有核心采用 0.06° 输入缓冲、J2015.5、HEALPix NSIDE 128 RING 和 W0 r-only 线性稳健运动。

科学试运行统计实际取数、参考图支持、观察边界和数据缺口。正式认证仍沿用历史零点与接缝证据。C1/C1b 的留出边界对照作为独立试验。

## 收数和公共标定

初始探针比较官方 TAP 同步、Data Lab 对象 ID 和 Data Lab 原生 SQL JOIN。原生 JOIN 根据服务器归档对象坐标切分，每块单独核对服务端计数、内容和检测 ID；存储成功分块的行数和 SHA-256。只有正常完成版本回执的输入进入参考系。NSC 对象标签用于索引、背景星系定义及官方 NSC 同源验证，盲 W0 关联从单曝光位置生成。

原生 SQL 可使用 `native_count_window = true` 试行同一响应中的服务端计数。每个返回行携带 LIMIT 之前的总数。只有逐行总数一致、实际行数相等且符合上限时才写入缓存。超过上限继续递归分片，空响应、截断或查询失败回退到独立 COUNT 路径。分片最终的检测 ID 核验沿用原流程。该选项使用独立的配置及版本回执。

每核心独立提取符合冻结条件的参考星系检测，位置均投影到同一个以 RA 350°、Dec 0° 为中心的切平面。重复检测按 detection_id 去重，跨分片身份冲突行剔除，同物理曝光多重参考记录保留歧义统计。采用冻结星系划分和 C0 共享参数（SDSS run/camcol、NSC exposure），参考星系以 48 次重采样传播共同校准误差。基准的 NSC 中位位置常量统一定锚每个连通分量。缺少双巡天连接的组、支撑不够的组和未连接分量分别记录。共享物理组只有一个偏移估计。完整区域处理选择各核心主要支持分量，不将不同分量的检测混进同一个运动拟合。逐源文件中的运动分量以同一切平面为坐标，参与验证的 Gaia/发布自行也投影到同一平面。

## 科学统计

连续区扫描分层按 v0.9 的 2° 预查冻结。Dec −1° 至 +1° 为长重复带，绝对 Dec ≥3° 为普通扫描对照，其余为过渡区。场内以实测物理 run 核验这一几何分类。普通源盲样本和用于固定对象与检测的验证样本分开。发布比较固定相同对象，Xiao2025 限制 PM_SOURCE=DESI。所有 NSC r≥21 的 Gaia 自行退出精度和评级，保留 21–22、22–23 等光谱 QSO 静止控制与实测整年遮挡后的暗源位置预测。NSC 已发布位置受到 Gaia DR2 定标。缺少独立暗星自行真值时，零点与位置重复性有各自的适用范围。

保守的连通块按真实 SDSS run、NSC exposure 和观测夜计算。另有多成员权重分析，分别以 SDSS run、NSC exposure 或观测夜为共享单位，在每个单位抽独立指数权重，再平均到源。输出用于比较各类共享观测的敏感性，独立信息量是权重负荷指数。实际删组从源的拟合移除整个 run、曝光或夜并重估运动，计入失去时间支持的情况。删组保留原星系校准，测量的是自行对观测取舍的敏感性。相关参数仍不足以单独认证全区空间零点。

三路严格同源样本报告 C0 相对官方 NSC 和 Xiao 地基的径向 P68 比例变化。每种物理观测组的 800 次抽样对三路使用同一权重，共享组少于三或合格对象少于五时仅报告可用数量。扫描分层同时保留冻结的地理带和按每个 QSO 实际 SDSS run 数分组的结果。权重区间反映共享观测的敏感性，保守连通块仍单列。

仪器分层从固定同源拟合使用的实际 NSC 曝光连接芯片目录的仪器信息。整区自行拟合回执与芯片回执均核对校验和。按星等和扫描带报告单一仪器、多仪器组合以及缺失元数据的样本数与同源精度。仪器分层比较刻画样本覆盖，仪器效应还需要控制位置和时基。

## 命令

在 NERSC Slurm CPU 节点执行，所有输入目录和回执映射均保留在 Scratch。共用下面的配置。逐源目录不用于公开同步。

```bash
export PYTHONPATH="$PWD/src"
export DEEP_PM_M3_WORKERS=12 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
config=configs/continuous-v12.toml
python -m deep_pm.continuous_reference extract "$REFERENCE_SHARD" "$config" "$CORE" "$INPUT_SHARD"
python -m deep_pm.continuous_reference solve "$SHARED" "$config" whole_c "$REFERENCE_PLAN_JSON" --separate-components
python -m deep_pm.continuous_fit "$CORE_OUTPUT" "$CORE" "$INPUT_SHARD" "$SHARED" "$XIAO_CSV" "$config" --annual
python -m deep_pm.continuous_summary "$PUBLIC_SUMMARY" "$CORE_PLAN_JSON" "$SHARED_PLAN_JSON" "$config"
python -m deep_pm.continuous_instruments "$INSTRUMENT_SUMMARY" --plan "$CORE_PLAN_JSON" --config "$config"
MPLBACKEND=Agg python -m deep_pm.continuous_plots "$PUBLIC_SUMMARY" figures
MPLBACKEND=Agg python -m deep_pm.continuous_plots "$PUBLIC_SUMMARY" figures --detailed
```

详细图使用同一色标对照东西与南北零点。暗 QSO 的 ±1σ 覆盖率以 68.27% 为色标中心。每幅图只渲染传入汇总中的已核验核心，未核验区留灰。QSO 面板标出各核心样本数。

物理边界另行读取相邻 SDSS field 的 `photoObj` 原始重复记录，以及 NSC 同一曝光内不同 CCD 对同一留出星系的测量。SDSS 按候选数排序，每核心至多检查 160 个邻 field，并报告全部可选 field 数、读取数及匹配数。两类物理重复对都共享一个 C0 零点，位置差可测量这一尺度的重复性；对的数量决定证据范围。几何边界继续使用共享图的 HEALPix 与处理核心留出统计。

整区分块边界以实际共边线段为单位，比较两侧各 2′ 内的留出星系跨巡天位置残差。仅使用解算时绑定的参考回执；同一星系取有双巡天支持、检测最多的参考图分量。每个分量每侧至少五颗才报告差值，样本不足的边界仍保留计数。位置差可受两侧样本与观测条件影响。

```bash
python -m deep_pm.continuous_detectors "$DETECTOR_OUTPUT" "$config" "$INPUT_SHARD" "$REFERENCE_SHARD" "$SHARED" "$SDSS_SAS" "$CORE"
python -m deep_pm.continuous_seams "$SEAM_SUMMARY" "$SHARED" "$REFERENCE_PLAN_JSON" "$config" --plan
```

一次试运行以实际配套回执为边界。所有未取全的块以缺区记录，不能将已有的几何覆盖估算视为已处理数据。 [NSC 数据入口](https://datalab.noirlab.edu/data/nsc) · [SDSS DR17 Resolve](https://www.sdss4.org/dr17/algorithms/resolve/) · [NERSC CPU 队列](https://docs.nersc.gov/jobs/policy/)

覆盖回执分别统计已有取数、参考提取、拟合回执和有可用参考分量的格数。逐核心汇总提供最新成功取数的读入行数、标准化记录大小、耗时、峰值内存和参考星系支持。读入行数含缓冲区及合并输入的重复读取。失败和重试的额外时长由 Slurm 回执记录。

合格 W0 解按 NSC r 星等和实际 SDSS run 数汇总。用于拟合的 SDSS 跨度、NSC 跨度、联合跨度、独立组数、形式误差与最大观测组的斜率权重份额分别给出第 16、50、84 百分位及有效个数。重复比例的分母是各星等档已通过 W0 质量门槛的源。逐源记录留在 NERSC。
