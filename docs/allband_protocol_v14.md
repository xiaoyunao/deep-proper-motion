# 全波段阶段的科学控制与执行约束

基准为5fba219，原始C0/W0 r-only／50mas底噪／J2015.5／四参数稳健解冻结。原North Two已经打开，本轮不包装成新的未知真值。实际流程按PLAN八步骤，所有真实计算仅Slurm、输出版本目录独立。

## 首要参照实验

N1/N3全部原参考分片先以RA/Dec重投影至(188,33)，物理参考ID、检测ID、训练/留出、原生边际误差和行序保持；原两个图只读。使用相同C0物理组、支持门槛、48次参考bootstrap和12mas组SE门槛，共同解一次。先报告数学上共享组一致性，再用留出星系和独立QSO检查位置、运动、尾部、形式误差与零点；公共项不除以源数平方根，不同旧图抽样不假装同步。双巡天不连接的分量和弱组保留退出状态，不强行当一个规范。

A只重放原全部参考图及相同参数，保存：原训练支持、剥离后支持／连通分量、进入求解的组、SE门槛前后偏移与公共抽样、逐核心退出原因。先逐组核对保留偏移与原v1.3一致，再解释12核心缺支持。未验证前不认定弱桥、图算法或测量误差为根因。

## 候选和公平比较

所有实际NSC滤光片都统计；u/g/r/i/z/Y/VR是官方目录标签清单，不作为选择白名单。SNR仅一项条件。原生位置误差、时间、饱和／截断／溢出、混叠提示、CCD足迹与WCS／参考支持必须追溯。被去混叠的检测不等于必然错误，需保留身份／邻居证据和严格控制样本。缺测标志或误差不视作零。

主对照保持原r对象和原r成员，B仅追加合格NSC检测；若增加物理质量条件会删掉原r成员，另设同样删选的r控制，把质量删选和加band的贡献分开。C与B检测完全相同，只改经过验证的误差模型。SDSS其他band的offset、逐源TAI、逐band质心与photoField WCS误差未验证前不进入D；不把一条RAERR复制为五次独立误差。

目录索引的ndetr及rmag<24等历史获取条件会影响新锚点候选；已有缓存上的产出只声称该检索集合，补充少量曝光/CCD路线检查遗漏与objectid选择效应。objectid不得作新关联正确答案。

## 残差、训练与协方差

每曝光自由二维零点与一个额外全局band常数代数退化。纯重参数化不算收益；校正必须有明确定锚和可辨识的残差特征，例如可核验几何的颜色项。原galaxy训练/留出身份保留；新增误差学习可在留出物理参考中预先二分，但该部分应明确改作参数训练，另一半才验收。光谱QSO不参与系数训练。亮端可靠Gaia星用途独立标记，r≥21目标Gaia PM不评分。

逐band和整年删组记录实际时间/斜率信息，源成员固定后与再次盲关联两项分开。位置独立噪声、曝光／扫描／观测夜随机项、星系零点公共bootstrap分列，避免已含共模的发布误差与新公共项重复相加。没有足够训练参考或独立重复对时记录未校准，不自动沿用旧W2参数。

## 采用与扩大

比较相同QSO和检测的二维P68、两轴中位/NMAD、尾部、标准化覆盖及实际源数；分21–22、22–23、单/多run、NSC跨度和真实仪器组合。对象区间、物理组/夜敏感性与独立区域重复证据分开，少量块不当广域认证。

只有普通区精度或经过审核的有效源数收益可复现，新增源零点／身份／误差及有效机会有支持，才冻结采用方案并启动共同图约10×10度科学目录和独立迁移。没有收益则保存负结果和具体停止原因，不以处理更多面积代替验证，也不预设全波段必然更准。

新增源分r已有、追加band才可用、非r新锚点；测试年所有目标检测先移除再生成训练关联，真实合格曝光机会与所有档案芯片分开，一对一匹配和位移负对照不取消。未知身份不能算正确，匹配门限内残差不能代表无条件误差。

## 本轮复现入口

所有真实表操作在Slurm分配内，从对应源码commit根目录运行；每个输出使用独立版本目录。参数路径来自个人不提交配置，不在此填机器绝对路径。

```bash
python -m deep_pm.allband_inventory "$WORK" "$SOURCE" "$SHARED" "$GEOMETRY" "$REGION"
python -m deep_pm.allband_qso "$WORK" "$SOURCE" "$FROZEN_FIT" "$FROZEN_C0" "$COMMON_C0" "$GEOMETRY" "$REGION" --multiband
python -m deep_pm.allband_diagnostics "$WORK" "$FIXED_QSO_TRIAL"
python -m deep_pm.multiband_error_trial "$WORK" "$FIXED_QSO_TRIAL" "$REFERENCE_SHARD"
python -m deep_pm.allband_blind_year "$WORK" "$SOURCE" "$SHARED" "$GEOMETRY" "$REGION"
python -m deep_pm.exposure_index_audit "$WORK" "$SOURCE" "$SHARED" "$GEOMETRY" "$REGION"
python -m deep_pm.legacy_band_attribution "$WORK" "$FROZEN_LEGACY_EXTRA_TRIAL"
```

共同图、组筛除、SDSS位置探针和支持敏感性的CLI见相应模块的`--help`。`reference_support_trial`仅复制全部pre-SE偏移／同步抽样，不重新定锚、不删除公共位置协方差，不是默认C0。原额外源须按旧W2 scale/floor及逐检测顺序重现；线性归因固定稳健影响系数，均值可加而中位数不可加，不当作因果或纠正系数。

后续真实运行入口（相应模型均是研究对照）：

```bash
python -m deep_pm.sdss_position_check "$WORK" "$SDSS_PROBE" "$SDSS_SAS"
python -m deep_pm.fixed_year_multiband "$WORK" "$FROZEN_BLIND_YEAR"
python -m deep_pm.allband_member_audit "$WORK" "$FROZEN_BLIND_YEAR"
python -m deep_pm.galaxy_chromatic_probe "$WORK" "$FIXED_QSO_TRIAL" "$REFERENCE_SHARD"
python -m deep_pm.chromatic_reference prepare "$MODEL_WORK" "$FROZEN_BATCH" "$JOINT_REFERENCE" --workers 4
python -m deep_pm.chromatic_reference solve "$MODEL_WORK" "$FROZEN_BATCH" "$JOINT_REFERENCE"
python -m deep_pm.chromatic_validation qso "$WORK" "$FIXED_QSO_TRIAL" "$MODEL_WORK"
python -m deep_pm.chromatic_validation year "$WORK" "$FROZEN_BLIND_YEAR" "$MODEL_WORK"
python -m deep_pm.chromatic_validation blind-zero "$WORK" "$FROZEN_BLIND_YEAR" "$MODEL_WORK"
```

颜色模型另存`private_chromatic_reference.npz`，其科学资格化和固定对照见[单色项协议](chromatic_joint_trial_v14.md)。年度和盲零点的投影中心必须数值一致，JSON列表／NPZ元组表示不同不构成科学差异；真实不同投影仍拒绝。所有参数、缓存、回执只在独立版本工作目录；不得用新源码覆盖旧失败阶段。

公开图只读汇总即可重现（不需要逐源数据）：

```bash
PYTHONPATH=src python -m deep_pm.allband_evidence_figures results/summary figures/v14_independent_evidence_summary.png
```
