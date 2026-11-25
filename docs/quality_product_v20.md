# v2.0 附加科研质量产品

这是一份独立于原生产状态的质量证据层，不改自行点估计或原4×4协方差。原每源 `quality_flags`、`kind`、真实物理组、观测信息、参考图SHA及独立／共同协方差继续保存。

## 连接与口径

- 由每源J2015.5标准ICRS位置计算HEALPix RING NSIDE32，连接 `v20_ancillary_quality_summary.json` 的cells。
- QSO证据按原光谱 `reference_r` 的21–22及22–23分别提供。恒星应保留原实际NSC r或SDSS r来源，不能把其颜色／选择差异忽略。
- 原星表NSIDE128组织不变。附表是NSIDE32统计证据，不能当作统一零点校准参数，不能把全像素面积当作已认证面积。
- `median_masyr` 是实测原始QSO两轴中位数；`raw_1sigma`／`raw_2sigma` 使用原报告误差；`centered_1sigma` 只诊断去中位数后的覆盖。
- `conditional_object_zero_interval95` 是对象抽样描述区间，不包含全部公共观测组／空间模式。`independent_group_support` 只检验现有组数门槛，不能解释为这些组完全独立。
- `stellar_transfer_certified=False`，`unmeasured_shared_systematic_error=True`。16种训练QSO经验尺度均未稳定通过身份确认及八新窗迁移，原协方差不覆盖、尺度不采用。
- 没有合格QSO样本的单元保持缺测，不能填零。没有可测的普遍暗恒星认证面积，用null表示证据未建立。

## 采样用途层

`r_anchor_well_sampled` 沿用原季节与最大物理组杠杆标志；`r_anchor_limited_sampling` 是少于三NSC季节或单组≥80%杠杆；`non_r_anchor` 另列。分类互斥但没有穷尽源身份、颜色差、空间误差。它们是选择研究用途的入口，不是正式纯度等级。

每类的源数与保留率是准确计数；`occupied_core_envelope_sqdeg` 是出现这类源的处理核心面积上包络，不同类别重叠，不能相加。几何机会面积仍用原冻结扫描／CCD／时基回执，认证面积另列。`all_magnitude_source_count` 包括所有星等，不称作暗星数。

## 停止规则与复现

本轮不追加新的误差参数以追求某个QSO覆盖目标。源成员先冻结，验收QSO不估计C0零点。NGC局部QSO平移仅作失败诊断；其平移均值不含训练零点不确定度，不可拿来替换正式误差。

NERSC入口：`python -m deep_pm.usability_quality REGISTRATION OUTPUT`，随后 `python -m deep_pm.usability_ancillary CURRENT_SCIENCE QUALITY_SUMMARY CLUSTER_DIAGNOSTICS OUTPUT`。路径从不提交的个人配置读取，科学程序须在Slurm计算节点运行。

完整结果见[科学报告](../reports/experiment_A_v2.0_science_usability.md)、[QSO训练与转移](../results/summary/v20_quality_calibration_summary.json)、[附加质量层](../results/summary/v20_ancillary_quality_summary.json)。
