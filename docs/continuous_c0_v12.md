# 连续 C 区的 C0 试运行

C 区为 RA 345°–355°、Dec −5°–+5°。几何核心使用半开区间，坐标范围和物理观测身份在配置中冻结。既有 14 个 1° 格可由四个缓存足迹覆盖，东南东半的两个 1° 输入保留各自回执。其余核心分批取数。所有核心采用 0.06° 输入缓冲、J2015.5、HEALPix NSIDE 128 RING 和 W0 r-only 线性稳健运动。

科学试运行统计实际取数、参考图支持、观察边界和数据缺口。正式认证仍沿用历史零点与接缝证据。C1/C1b 的留出边界对照作为独立试验。

## 收数和公共标定

初始探针比较官方 TAP 同步、Data Lab 对象 ID 和 Data Lab 原生 SQL JOIN。原生 JOIN 根据服务器归档对象坐标切分，每块单独核对服务端计数、内容和检测 ID；存储成功分块的行数和 SHA-256。只有正常完成版本回执的输入进入参考系。NSC 对象标签用于索引、背景星系定义及官方 NSC 同源验证，盲 W0 关联从单曝光位置生成。

每核心独立提取符合冻结条件的参考星系检测，位置均投影到同一个以 RA 350°、Dec 0° 为中心的切平面。重复检测按 detection_id 去重，跨分片身份冲突行剔除，同物理曝光多重参考记录保留歧义统计。采用冻结星系划分和 C0 共享参数（SDSS run/camcol、NSC exposure），参考星系以 48 次重采样传播共同校准误差。基准的 NSC 中位位置常量统一定锚每个连通分量。缺少双巡天连接的组、支撑不够的组和未连接分量分别记录。共享物理组只有一个偏移估计。完整区域处理选择各核心主要支持分量，不将不同分量的检测混进同一个运动拟合。逐源文件中的运动分量以同一切平面为坐标，参与验证的 Gaia/发布自行也投影到同一平面。

## 科学统计

普通源盲样本和用于固定对象与检测的验证样本分开。发布比较固定相同对象，Xiao2025 限制 PM_SOURCE=DESI。所有 NSC r≥21 的 Gaia 自行退出精度和评级，保留 21–22、22–23 等光谱 QSO 静止控制与实测整年遮挡后的暗源位置预测。NSC 已发布位置受到 Gaia DR2 定标。缺少独立暗星自行真值时，零点与位置重复性有各自的适用范围。

保守的连通块按真实 SDSS run、NSC exposure 和观测夜计算。另有多成员权重分析，分别以 SDSS run、NSC exposure 或观测夜为共享单位，在每个单位抽独立指数权重，再平均到源。输出用于比较各类共享观测的敏感性，独立信息量是权重负荷指数。实际删组从源的拟合移除整个 run、曝光或夜并重估运动，计入失去时间支持的情况。删组保留原星系校准，测量的是自行对观测取舍的敏感性。相关参数仍不足以单独认证全区空间零点。

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
MPLBACKEND=Agg python -m deep_pm.continuous_plots "$PUBLIC_SUMMARY" figures
```

一次试运行以实际配套回执为边界。所有未取全的块以缺区记录，不能将已有的几何覆盖估算视为已处理数据。 [NSC 数据入口](https://datalab.noirlab.edu/data/nsc) · [SDSS DR17 Resolve](https://www.sdss4.org/dr17/algorithms/resolve/) · [NERSC CPU 队列](https://docs.nersc.gov/jobs/policy/)
