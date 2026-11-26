# v2.1 运行、签名、资源与完整更新预算

2026-11-26；冻结6faabde，版本分支experiment-a-v2.1-coordinate-consistency。原下载、星系、逐检测、逐源及Gaia补充表均在个人PScratch；本地/公开仓库只含代码和经检查的聚合。未把数据写入Home，未要求HPSS登录。

## 实际计算版本

| 阶段 | 运行代码快照 | 证据 |
|---|---|---|
| 元数据登记 |113b7b9|三真实图头、按元数据选3先导及16核心 |
| 固定成员B1/B2 |6879862|70120源、1060旧盲目录光谱身份；H和旧公共cov复现 |
| 2D质量与球面重关联 |c6e3051|三核后同缓存扩16，原0.8形态 |
| 有界参考初试/时序对照 |e5d6647 / cdc7adc|同边界/成员，48配对draw，后者含参考权重单独点响应及计时 |
| 实际历元亮星 |e5d6647 / 92602d9|6471固定身份五参数补取；后者含独立身份K预测 |
| 旧几何唯一候选全集竞争 |76fd2b7|90255原训练+7254新增、实际CCD/depth；首次LS形态解析错误不影响竞争/CCD |
| LS修复/公平尺度/数值读出 |8772e9a|p025=25度回归；形态重读、相同训练中心误差评价、16核4×4与检测唯一性 |
| 原训练形态哈希对照/参考时序 |cdc7adc|9019原控制、7254新增、三参考域完整元数据 |
| 原生r−i与g−i |85a161d|6470合法颜色；不拟合新生产系数 |
| 全历元/训练成员连续性 |d690bba|7479全历元新增映射到实际留年分母 |
| 实际median-NSC gauge/质量门槛 |4abdeae|单零组探针不是实际gauge；固定源误差/残差退出分列 |

每核ticket包含source/fit回执、参考NPZ、相关实现的SHA。旧C0的48个draw是同图同步的图平面量；不得再次前向投影。最终公开文件SHA和源代码文件SHA在发布回执中登记。代码快照先完整解包再运行，原生BLAS/OMP均单线程；没有同时启动超过两个共享分配。

## 复现入口

必须在已分配Slurm计算节点、源码快照根目录中执行。实际个人绝对路径和解释器在不提交的配置中。下面变量示意应指向原冻结缓存，输出使用独立新版本目录；不要覆盖已经完成的v2.1结果。

```bash
export PYTHONPATH="$PWD/src:${DPM_RUNTIME:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DPM_NEW="${DPM_ROOT:?}/v2.1-reproduce"
python -m deep_pm.coordinate_inventory "$DPM_ROOT/v1.9/batch/science" "$DPM_NEW/registration"
python -m deep_pm.coordinate_replay "$DPM_NEW/registration/private_coordinate_plan.json" "$DPM_NEW/fixed" --workers 3 --pilot
python -m deep_pm.coordinate_replay "$DPM_NEW/registration/private_coordinate_plan.json" "$DPM_NEW/fixed" --workers 3
python -m deep_pm.coordinate_associate "$DPM_NEW/registration/private_coordinate_plan.json" "$DPM_NEW/association" --workers 3 --pilot
python -m deep_pm.coordinate_associate "$DPM_NEW/registration/private_coordinate_plan.json" "$DPM_NEW/association" --workers 3
python -m deep_pm.coordinate_reference "$DPM_NEW/registration/private_coordinate_plan.json" "$DPM_NEW/reference" --fixed "$DPM_NEW/fixed"
python -m deep_pm.coordinate_stars "$DPM_NEW/registration/private_coordinate_plan.json" "$DPM_ROOT/v2.0/ngc_motion/private_cluster_diagnostics.json.gz" "$DPM_NEW/stars" "$DPM_ROOT/v2.1/cache"
python -m deep_pm.coordinate_ngc_trace "$DPM_NEW/registration/private_coordinate_plan.json" "$DPM_ROOT/v2.0/ngc_photometry/private_photometric_parent.csv.gz" "$DPM_NEW/association" "$DPM_NEW/ngc415" "$DPM_ROOT/v2.1/cache-ngc"
python -m deep_pm.coordinate_readout "$DPM_NEW/registration/private_coordinate_plan.json" "$DPM_NEW/fixed" "$DPM_NEW/association" "$DPM_NEW/readout"
python -m deep_pm.coordinate_stellar_readout "$DPM_NEW/stars/private_stellar_positions.json.gz" "$DPM_NEW/readout/v21_stellar_group_summary.json"
```

旧几何候选实际复用v20成员。可在Slurm中运行：

```bash
python -m deep_pm.coordinate_candidate "$DPM_ROOT/v2.0/private_preregistration.json" "$DPM_ROOT/v2.0/selection-candidate" "$DPM_NEW/candidate" "$DPM_ROOT/v2.1/cache-candidate" "${LS_DR9_ROOT:?}" --workers 2
```

`coordinate_candidate`主CLI执行全集竞争/CCD机会，`repair_morphology`只补读固定新增的外部形态，`identity_control`哈希预选原训练对照，`member_continuity`连接全历元与训练成员；均使用独立输出。实际Slurm driver、命令与源版本保存在PScratch，输入实路径不公开。可按以下函数入口重放，而不是重新搜索形态阈值：

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
from deep_pm.blind_catalogue_quality import read
from deep_pm.coordinate_candidate import repair_morphology, identity_control, member_continuity
from deep_pm.coordinate_support import gauge_audit
from deep_pm.coordinate_readout import quality_gates
# 从已设置的个人配置读取路径；输出为独立新版本。
root = Path(os.environ['DPM_ROOT'])
out = Path(os.environ['DPM_NEW'])
ls_dr9_root = Path(os.environ['LS_DR9_ROOT'])
plan = read(root/'v2.0/private_preregistration.json')
with ProcessPoolExecutor(max_workers=2) as pool:
    morph = list(pool.map(repair_morphology, [
        (r, str(out/'candidate'), str(out/'morphology'), str(ls_dr9_root)) for r in plan['selected']]))
with ProcessPoolExecutor(max_workers=2) as pool:
    identity = list(pool.map(identity_control, [
        (r, str(out/'candidate'), str(out/'morphology'), str(out/'identity'), str(ls_dr9_root))
        for r in plan['selected']]))
member_continuity(root/'v2.0/private_preregistration.json', root/'v2.0/selection-candidate',
                  out/'candidate', out/'readout/v21_candidate_continuity_summary.json')
gauge_audit(out/'registration/private_coordinate_plan.json', out/'gauge')
quality_gates(out/'registration/private_coordinate_plan.json', out/'fixed', out/'association',
              out/'readout/v21_quality_gate_summary.json')
```

`coordinate_scale_audit`读取冻结v20模型及盲目录，不重训；`color_coordinates`读取已保存星位诊断；本地仅对公开统计运行 `python -m deep_pm.coordinate_figures .`。合成数学验收可本地执行：

```bash
PYTHONPATH=src python -m deep_pm.coordinate_checks results/summary/v21_mathematical_acceptance_summary.json
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m pytest -q
```

## 本轮实耗与失败

14个已分配任务含1失败，共 **0.02773872 CPU节点小时**（预算5）；唯一候选连同全部共享后处理任务上界 **0.01081814**（目标1）。版本目录末次快照 **1.480GiB / 6009文件**；队列为空。实际Slurm作业与step MaxRSS均写入[资源总账](../results/summary/v21_resource_usage_summary.json)。用顶层已分配Elapsed×billing TRES/256/3600折算CPU节点小时，含62015775导出失败与科学上未采用的试验；不是Iris最终项目账务认证。短step的0/空MaxRSS表示未测得，不解释为零内存。共享分配的墙钟时间不能直接当完整节点小时。

shared_interactive拒绝sbatch及第三个并发提交，均未分配计算资源；随后复用已有salloc工作流。外部LS首读出错误是科学实现失败，完成作业仍计入资源并保留；只补读缺项，不重跑成功的竞争/足迹阶段。首次全项目git archive包含公开旧图约304MB，后续改为显式代码/配置/测试允许列表，约2.2MB，不复制逐源数据。

空间快照是版本目录的实际文件字节/文件数，不是监测到的峰值，也不等于全用户配额。形态阶段触及大sweep内存页，step RSS靠近12–16GiB请求；若重跑双worker外部形态，建议20GiB留余量。三worker源重放/关联使用12–16GiB；完整参考图另计。

## 已有全域一次更新的实测外推

所有下载和签名规范输入可复用。登记范围共2875核心、15,625,958科研解，含halo重复加载输入1,750,398,729行。三个参考图均存在图心远端，不能按某个近中心核心的表现把整个图标为无需更新。

| 阶段 | 本轮测量 | 外推依据 | 规划CPU节点小时 |
|---|---|---|---:|
| 固定成员B1+B2诊断 |16核合计317.63 worker秒、70120源、5020798含halo输入 | 分别按源数222.85倍、输入行数348.63倍，原8 billing/3worker |0.205–0.320；若全域保留此对照，另加安全系数 |
| 2D质量+球面关联及对照 |16核合计2513.96 worker秒 | 同两比例，10 billing/3worker |2.026–3.170；2倍工作/调度余量为约4.1–6.4 |
| 已有两完整参考图 |历史61913025/61913029 | 原同类方法全图真实1.803/5.375节点小时 |7.178原模型实耗 |
| 第三参考图 |119.55百万参考测量，对比第二图106.40百万 | 用第二图密度比例估计，非实测 |约6.04旧模型代理 |
| 三完整二维参考重解 |原始参考唯一测量39.43/106.40/119.55百万 | 旧全图合计代理13.22，规划1.5–3倍全局耦合/内存/迭代余量 |约20–40，条件规划 |

有界48draw旧对角/完整2D耗时近2.484/2.506、中1.698/1.551、远2.074/2.196秒，未显示局部显著代价。但小边界图无法代表一亿行全图的条件数、数据结构与内存，不能只乘这一近1的比值。

原最大参考RSS约349GiB；显式二维稀疏矩阵若近翻倍会达到约698GiB，**禁止直接将有界试验设计放大到全域**。完整求解需保持原数学目标，采用紧凑数值/内存映射和分块或矩阵自由消元，先核实实际节点内存及真实连通分量迭代性能。可并行独立参考域、同域共享统一解；不能以独立小区零点替代共同参考图。旧12图平面mas选择与原生角单位支持定义亦须分别登记。

下一次完整生产更新可先规划约22–46节点小时、48封顶余量（若全域另保留B1诊断需在余量内另计）。这是**有条件的资源估计，不是当前实现保证的费用上限，也不是本轮5小时预算内的已执行工作**。三完整参考分量尚未重解；本轮按要求交付边界实验、科学影响和预算，不擅自消耗下一轮额度。

## 公开文件有效性

`v21_candidate_completion_summary`的竞争/CCD结果有效，其external_morphology字段因旧文件名解析失败无效；采用`v21_candidate_morphology_summary`和`v21_candidate_identity_summary`。单零组gauge启发式探针已废弃，采用`v21_gauge_definition_summary`。对象条件区间、SDSS run敏感度、UTC夜敏感度和旧图同步参考不确定度分开，不相互替代。


| Slurm | 用途 | 状态 | 分配秒 | billing | 最大报告step RSS(GiB) |
|---|---|---|---:|---:|---:|
|62014996|registration|COMPLETED|40|6|0.877|
|62015306|fixed_covariance_pilot|COMPLETED|74|10|1.220|
|62015531|fixed_covariance16|COMPLETED|112|8|4.330|
|62015775|failed_schema_export|FAILED|134|8|0.101|
|62016076|angular_pilot|COMPLETED|242|8|1.764|
|62016244|reference_stars|COMPLETED|166|6|1.672|
|62016271|angular16|COMPLETED|772|10|5.011|
|62016379|candidate_full_competition|COMPLETED|685|10|15.955|
|62016945|stellar_K_NGC_QSO|COMPLETED|234|6|1.994|
|62017155|morph_repair_symmetric_QSO|COMPLETED|237|8|11.955|
|62017532|identity_reference_metadata|COMPLETED|141|8|11.955|
|62017910|native_colours|COMPLETED|26|4|未采样|
|62018123|continuity_old_gauge_probe|COMPLETED|24|4|未采样|
|62018626|actual_gauge_quality|COMPLETED|81|6|1.619|
