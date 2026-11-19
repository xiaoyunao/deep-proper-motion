# v1.5 连续区科研目录的读取与误差定义

这是分片保存的完整科研试验目录，不是正式发布星表。私人`DPM_RUN/fits.json`是91个核心的逻辑索引，指向已经签名的实际输出；其中5个核心引用NaN光谱星等修复后的续跑，其他核心引用原完成目录。`validated_completion_summary.json`绑定科学重放与统一参考SHA。最终逻辑目录另有`private_owner_overlay.json`：按J2015.5把5个全band／2个r控制原记录转给唯一实际核心；原fits索引的目录名是锚点任务来源，不能直接冒充最终输出owner。状态／协方差／成员的canonical JSON hash不变，原科学文件和回执不覆盖。只有已签名实际91核心参与归属，配置中其他试验／子区别名不参与。`sources.json`及每片`source_receipt.json`追溯实际输入，原失败目录和日志保留，不伪造一次全部成功。

## 每个分片

- `private_standard_motion.json.gz`：全NSC合格band的科研盲发现结果；先写这些源，再匹配光谱QSO。每行含ICRS位置、自行、完整4×4协方差、参考历元、真实检测ID、物理组、band、仪器、夜、时间杠杆、稳健权重、来源类别与质量标志。
- `private_r_control.json.gz`：同输入／同共同图下原r质量的控制结果；新物理质量r控制的数量、同源QSO结果另在汇总保存。旧r质量控制与物理质量r不可互相冒充。
- `private_known_qso_blocks.json`：已知光谱位置对照，含显式本地验证索引和ICRS状态；这不是盲目录全部对象的身份真值。
- `private_blind_qso.json`：源目录冻结后的光谱位置唯一匹配。`C0`在这里是**同源已知位置的全NSC拟合**，不是r-only；r控制必须读明确命名的对照。
- `nsc_science_catalogue_summary.json`与`run_manifest.json`：选择漏斗、科研源数、QSO指标、运行参数与输入／输出校验。源级信息不离开PScratch，公开仅去标识汇总。

## 协方差

`reference_epoch=2015.5`；角位置为度，自行为标准`mu_alpha*`／`mu_delta`（mas/Julian年）。`covariance`顺序为：

1. 局部东向位置扰动（mas）；
2. `mu_alpha*`（mas/yr）；
3. 局部北向位置扰动（mas）；
4. `mu_delta`（mas/yr）。

位置协方差是局部切向角扰动，不是`RA*cos(Dec)`这个坐标函数的协方差。`state_jacobian`把切平面状态及完整协方差传播至每源ICRS；对角线第1／3个零起始索引的平方根等于`pm_error_masyr`。独立／参考切平面分量分别保存在`independent_tangent_covariance`和`reference_tangent_covariance`，同一个Jacobian可转换各分量。

原`detection_arrays`仍将发布east/north误差加底噪后作为切平面两轴对角权重的近似，没有重建逐检测完整WCS／投影相关项。输出状态及其条件4×4使用完整ICRS Jacobian传播；这不等于补齐了原生测量协方差。本轮为保持C0/W0公平对照不改变该近似。

参考分量来自原共同C0的同步bootstrap，公共组不能按源数开平方缩小。原生误差和50mas底噪是当前对角W0；未知同夜跨band误差、实际空间残余和绝对frame tie未全部进入该4×4，不能视其为已校准总误差。跨源公共参考协方差需要保留原组、参考NPZ和同步draw，不可从各源对角误差单独重建。

## 科研质量与源类

`bands`列的是NSC采用band；所有跨巡天解另含SDSS r，完整逐检测band可按ID回连原输入。`non_r_anchor`只表示没有合格NSC r锚点，不表示没有SDSS r位置或r测光。

`r_existing`、`extra_r_anchor`、`non_r_anchor`按完整全历元源定义；与年度验证只看训练年的三类分别报告。`r_existing`由实际成员与旧r合格对象的重叠识别，不由自行相近或NSC object ID识别。非r源的比较星等须来自真实SDSS r等明确测光，缺测单列。

每个源仍带`night_correlation_unqualified`、`spatial_zero_uncertified`及`identity_unconfirmed_outside_spectral_audit`。这表示未知误差／身份范围，不是已经测出全部源有误。额外形态未知或少于三个NSC季节的标志保留。当前数量只称“科研合格解”，不称纯度认证的暗星数。

## 计算节点上的最小读取

在个人配置填好`DPM_RUN`，获得Slurm节点后：

```python
import os
from pathlib import Path

assert os.environ.get('SLURM_JOB_ID')
run = Path(os.environ['DPM_RUN'])
core = 'repeat_sparse'
from deep_pm.nsc_ownership_overlay import owned_records
sources = [row for origin, slot, row in owned_records(run, core)]
print(core, len(sources))
```

不要把这些逐源行写入公开日志，也不要同步整个目录到本地。图和报告仅使用已审核的`*_summary.json`。
