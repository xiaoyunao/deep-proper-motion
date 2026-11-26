# v2.2 已有覆盖更新与恢复

全部科学执行在NERSC Slurm计算节点。`WORK`为个人PScratch下的独立v2.2工作根，`FROZEN`为冻结v1.9科学元数据根（含continuation manifest）。旧输入、参考、点值与失败回执不覆盖；本机只取公开允许的聚合。

全部2875核心、三个完整图/各48draw及科学读出已经完成。以下仅为必要时按签名恢复的入口，不应重启已成功阶段：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m deep_pm.full_coordinate_batch "$WORK" \
  --component 0 --component 1 --component 2 \
  --workers 32 --memory-gib 160 --hours 3.5
```

这里component仅决定调度域；每个核心在拟合前读取**所有实际目标物理组**，依其真实分量派发，不用本核心有没有参考星系代替目标支持。不同图的共同模式不混为相关。worker受实际CPU/内存约束，不能把上述32/160直接用于小分配。当前共享QOS实测单分配1节点/4小时、每用户2个提交；同时对独立域作业并行。`salloc`与`srun`都必须显式`-n 1`，避免Slurm按默认任务数重复启动驱动器。

恢复批次可显式选`--memory-model receipt-validated`，默认仍是原参考入口RSS上界。新资源预留上界为原估计与`2+8*CSV_GiB+128*旧科学gzip_GiB`的较小者，必须先有至少50个相同科学实现/实际分派回执，且所有实测源峰值≤预留80%才启用；任一后续核心越过80%则停止采用新预留并对剩余/运行任务恢复旧计数。此项只改调度，不改源算法或随机数。原规范输入和物理组映射SHA未变时，复用已有“全部目标组”分派回执，避免恢复时重复扫描巨大CSV；未知定义不能直接复用。实际速度/RSS由恢复批次记录，不据公式宣称加速倍数。

该恢复模式先用至多8个常驻进程验证**已经完成**的源回执、全科学缓存键及所有产物SHA，绝不在缓存池内拟合缺失源。12个按文件体量选出的真实大核心核验阶段约2秒、单进程峰值292MiB，原数值/回执不变；不将其当完整拟合速度。新拟合仍每核心独立进程，完成即释放全部分配。

Linux生产可显式加`--worker-start fork`，先载入模块再为每个新核心建立独立进程，减少共享文件系统的重复解释器启动。它要求本根目录的真实三分量数值验收与实现签名完全一致：14197解的完整状态/协方差/成员/权重、两条r控制、原固定与新盲光谱产物的解压内容全部相同。未通过时不能采用。BLAS/OMP/NUMEXPR均限制1线程；`subprocess`仍为默认复现入口。验收见[真实等价](../results/summary/v22_fork_worker_equivalence_summary.json)与[只读缓存](../results/summary/v22_warm_cache_validation_summary.json)。

全域点值完成后：

```bash
python -m deep_pm.full_coordinate_catalogue index "$WORK" --workers 8
python -m deep_pm.full_coordinate_catalogue close "$WORK"
python -m deep_pm.full_coordinate_production_readout "$WORK"
python -m deep_pm.full_coordinate_continuity "$WORK"
python -m deep_pm.full_coordinate_quality "$WORK" --workers 8 --frozen "$FROZEN"
python -m deep_pm.full_coordinate_ngc "$WORK"
```

源索引与光谱源池逐条解码已签名JSON，仅保留必要字段/光谱slots；3真实核心的科学Parquet逐字节相同，282固定/282盲光谱身份全部数值一致。原4worker/16GB物化索引OOM保留，不能据此称源拟合失败。最新源索引还显式验证输入点值SHA、源摘要版本、参考版本和J2015.5归属。

- `index`先验证每核心签名；若主源版本改变，停止受影响索引，不默默复用。小工程测试可显式指定`--frozen`，不能误用工程根的相对历史路径。
- `close`将一致的精确成员副本归为单一代表，其余共享检测保留冲突；原始点值始终可追溯。完整2875核心时断言原冻结15,625,958分母复现。新R控制的origin计数与全球accepted计数分别报告，不交叉相减。
- `production_readout`只读同版本回执和最终索引，汇总选择漏斗、逐分量检测应用、源级误差来源和实际worker成本；它不重做拟合。三项改动共同实施后的净数不能单凭漏斗拆成独立因果增益；对照原v2.1固定实验、B1协方差和本轮固定身份读出分别解释。
- 不能把原/新规范输入混进旧缓存。有效键包含数据/参考/实现/坐标及支持定义；失败步骤的未完成输出没有有效回执，使用独立恢复版本或保存原尝试后重算。
- 留出年读出从测试年前移除全部目标band；`full_coordinate_year_readout`只重读归档验证记录，按物理源整体重采样，不重新关联或拟合。
- 最终公开源/质量图统一NSIDE64；组织与原生机会地图继续NSIDE128。NSIDE64源统计直接从接受源重聚合，不能平均细像素中位数。QSO不足5的已占据像素显示灰色，不当作零改善。

资源账本用`full_coordinate_resources`读取实际父分配AllocTRES billing和Elapsed，失败/取消/OOM全部包含；step只诊断，不重复计费。额外后处理分配用`--job JOBID`登记。48 CPU节点小时总上限包含g支线≤2小时；原始外推不代替实测成本。私有输出位置及重建指针见[目录布局](catalogue_layout_v22.md)。

## 本轮终态与最小恢复

最终62052901已完成2875个派发/索引/参考SHA闭合及八格历史×run读出；没有剩余本轮计算。源科学实现SHA为`f7916c4cbec7f8219035791ef52a66a1224d058147628a7e0aa9aaeee9b40513`。最终派发快照2d6e6a9，主要只读收口8183d6d；NGC数组类型恢复6b0301b；r星等只读分母bd71f7d；交叉分层2fcc265，均未改变主源实现。

```bash
# 只在需要恢复缺失的八格汇总时使用，已有回执将直接验证复用
python -m deep_pm.full_coordinate_conditions "$WORK"
# 本地仅用已经审核的去标识汇总重画实际图
PYTHONPATH=src python -m deep_pm.full_coordinate_figures "$REPOSITORY"
```

`cluster_motion.mean`把缓存JSON列表恢复为float64数组，再合成独立与同步参考项。旧主驱动62049478的QSO成功后调用旧NGC而失败，原FAILED状态保留；62049478.6完成修复NGC、最终SHA绑定核验，勿据此重跑全域QSO。原完整QSO聚合约117MB，发布为无损gzip与空间精简主JSON，所有全域/条件数值完整保留；不能误传逐源池或未压缩大聚合。发布布局和原SHA写在publication_layout，完整gzip可解压复核。

最终账本从所有28个父分配终态sacct读取billing/256×elapsed，约4.541417节点小时，包含失败；B保守上界0.448485包含在总额。step列为诊断，不能重复加计或把worker墙时/extern TotalCPU当费用。存储快照163.49GiB/79866文件仅是新工作根当前占用，后续只读`du --inodes -s`有界统计为93496 inode（含目录、不跟随符号链接目标）；快照不代表峰值、历史缓存或inode配额。
