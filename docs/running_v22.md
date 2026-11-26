# v2.2 已有覆盖更新与恢复

全部科学执行在NERSC Slurm计算节点。`WORK`为个人PScratch下的独立v2.2工作根，`FROZEN`为冻结v1.9科学元数据根（含continuation manifest）。旧输入、参考、点值与失败回执不覆盖；本机只取公开允许的聚合。

三个完整图、各48draw及首分量验收已经完成。其余核心可按完成回执恢复：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m deep_pm.full_coordinate_batch "$WORK" \
  --component 0 --component 1 --component 2 \
  --workers 32 --memory-gib 160 --hours 3.5
```

这里component仅决定调度域；每个核心在拟合前读取**所有实际目标物理组**，依其真实分量派发，不用本核心有没有参考星系代替目标支持。不同图的共同模式不混为相关。worker受实际CPU/内存约束，不能把上述32/160直接用于小分配。当前共享QOS实测单分配1节点/4小时、每用户2个提交；同时对独立域作业并行。`salloc`与`srun`都必须显式`-n 1`，避免Slurm按默认任务数重复启动驱动器。

恢复批次可显式选`--memory-model receipt-validated`，默认仍是原参考入口RSS上界。新资源预留上界为原估计与`2+8*CSV_GiB+128*旧科学gzip_GiB`的较小者，必须先有至少50个相同科学实现/实际分派回执，且所有实测源峰值≤预留80%才启用；任一后续核心越过80%则停止采用新预留并对剩余/运行任务恢复旧计数。此项只改调度，不改源算法或随机数。原规范输入和物理组映射SHA未变时，复用已有“全部目标组”分派回执，避免恢复时重复扫描巨大CSV；未知定义不能直接复用。实际速度/RSS由恢复批次记录，不据公式宣称加速倍数。

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
