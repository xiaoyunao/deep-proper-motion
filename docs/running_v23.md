# v2.3 实际方法、回执和恢复

本轮是有限真实数据试验，主目录0320f59及g分支9a7d7db保持冻结。科学结果及限制见[中文报告](../reports/experiment_A_v2.3_faint_star.md)，逐条条件见[预登记](faint_star_protocol_v23.md)。不要将条件信息标签当成已认证暗星。

## 方法与入口

- `faint_preflight_v23`：绑定v22输入/最终回执，读取三场HACKS与13场SDSS实际schema。`faint_assets_v23`只补PSF/gain/sky/行时刻元数据。缓存文件留PScratch。
- `hst_validation_v23 prepare`：先冻结外部质量、HST原生光度、真实参考epoch、邻源及双向唯一位置身份；`evaluate`之后才读取地面PM。HST相对/发表绝对零点及公共协方差分开，同星Xiao仅DESI。当前主样本0，不能给暗端精度。`faint_hst_opportunity_v23`只追加机会计数，不改规则。
- `faint_covariance_v23`/`faint_budget_v23`：恢复保存H及权重，完整二维SDSS+NSC+48同步参考项闭合保存PM协方差；每核索引/组名缓存，首源H逐值对照原实现。八预选核心2128源。亮Gaia候选仅位置归因，非暗端验收。
- `early_pixel_data_v23`：实际空间PSF、官方掩膜与DN/sky/gain/dark噪声；发布16.5<r<20隔离点源、物理身份哈希80/20分开接框架，常量二维平移，跨场同校准对象bootstrap同步。1/13通过，其余样本不足。
- `early_pixel_targets_v23`：冻结40光度身份；晚NSC按实际质量、曝光唯一/竞争与参考支持选取一次，不用object ID真值；32条件轨迹/8退出。每run几何最佳field，真实row-time；无任何校准对象的场明确退出。
- `early_pixel_model_v23`：自由4状态，逐run正通量/局部背景剖面+50mas位置nuisance；7初值，局部Hessian/每轴profile/公共扰动响应。早期-only完全移除晚位置。有限PSF核没有覆盖时模板严格为0，避免把多项式外推到未观测的探测器位置；不新增运动先验。
- `early_pixel_validation_v23`：40真实、300空背景、150孤立/150邻源移动PSF控制；位置/掩膜/时基/相位/深度先登记。实际背景不重复加噪声，注入源Poisson；有效PSF负翼作约1–2%通量的物理非负近似，保留对精密偏差验收的限制。晚期为真实采样的条件模拟，非NSC足迹完备性模型。整早期run留出、9次公共扰动实际重拟合。每案例原子回执，公开不含位置/成员/切图。
- `faint_closeout_v23`：从实际私有产物在Slurm汇总，四张主图；不再拟合生产源。主图小幅布局QA不改变科学统计。

完整目标4×4协方差为条件像素/晚期测量项、C0同步参考项、独立接框架bootstrap响应的和，再传播到每源标准ICRS；不将不同来源同编号draw强行配成同一噪声。它仍不包含未辨识的颜色/夜/绝对零点误差。局部线性响应通过有限非线性重拟合诊断，不等于全样本严格误差覆盖。

## 真实 Slurm 阶段

|父分配|代码/阶段|实际状态与恢复|
|---|---|---|
|62083030|preflight|HST下载成功后header对象JSON失败，原缓存保留。|
|62083282|e92fdbb assets|恢复schema并补齐13个PSF/time/noise，完成。|
|62083592|6f0c188 HST|质量/身份冻结成功，历史目录错误路径使evaluate退出；冻结不重做。|
|62083895|d5e7594 diagnostics|26个官方C PSF对照及HST N=0评价完成。|
|62083976|b8d0742 budget|2核心完成；低效重复索引/组名构造被定位后取消，保留成本及完成回执。|
|62084179|e61d81b pixelprep|13场接框架完成，1场通过；flag只读审计完成。|
|62085121|b5673d9 pixel pilot|冻结晚轨迹成功；无校准场占位图心导致退出，改为缺机会。|
|62085263|d5e1a80 pixel pilot|3个先导实际计算完成，公开摘要cases键被拒绝；个体结果留私有。|
|62085587|e017361 validation|516个真实/控制案例成功保存，弱profile越出PSF有限支持而退出。|
|62085663|37527ef budget recovery|复用原2核心并完成余6核心、HST机会计数；H逐值等价和协方差闭合。|
|62085943|960f685 validation recovery|复用签名兼容的516结果，完成640案例及汇总；未重抽背景/改门槛。|
|62086200 / 62086412|d2ef6bd / 7bac300 closeout|保留初版；完整回执纠正1/13场支持及停止理由，四图与实际统计完成。|
|62086712 / 62086979 布局QA|3c1dbd5 + 最终布局文件SHA|只移动图例、补覆盖文字并修正小共同项文字的对比度；科学统计逐项相同。|

实际失败/取消的终态保留，不能将失败父作业因某些step成功而改称完成。资源按所有父分配计费，step仅用于RSS等诊断，不能再次加计。[终态资源](../results/summary/v23_resource_final_summary.json)

## 执行环境与命令

真实路径/主机置于不提交的配置。以下变量均为用户实际已有环境；不创建Home数据目录、不要求HPSS。已有Home环境只读使用。

```bash
# NERSC：先在个人PScratch的独立代码快照中执行。
export PYTHONPATH="$DEEP_PM_BASE/v2.2/runtime:$PWD/src:$DEEP_PM_COMPAT_RUNTIME"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# 在已核实account/QOS的Slurm内：-N1 -n1，worker不超过分配CPU。
"$DEEP_PM_PYTHON" -m deep_pm.hst_validation_v23 evaluate "$DEEP_PM_BASE" "$DEEP_PM_V23"
"$DEEP_PM_PYTHON" -m deep_pm.faint_budget_v23 "$DEEP_PM_BASE" "$DEEP_PM_V23"
"$DEEP_PM_PYTHON" -m deep_pm.early_pixel_validation_v23 "$DEEP_PM_BASE" "$DEEP_PM_V23" --workers 8
"$DEEP_PM_PYTHON" -m deep_pm.faint_closeout_v23 "$DEEP_PM_BASE" "$DEEP_PM_V23"
```

这些是恢复入口，不应顺手重跑已完成阶段。HST prepare、40晚期输入与背景登记已冻结。旧案例兼容仅允许明确登记的旧模型/实现/晚期SHA：修复只把完全无PSF核覆盖的模板返回精确0，旧成功计算在这些点已是零插值；不能泛化接受任意旧结果。不同输入/规则/模型应另建版本目录。

最后一个分配退出后，资源终态命令是**管理性 sacct 读出**，不进行科学拟合，使用同一完整PYTHONPATH，避免Python3.8遗漏tomli兼容层：

```bash
"$DEEP_PM_PYTHON" -m deep_pm.faint_resources_v23 "$DEEP_PM_V23" --final
# 本地：仅18个具名公开文件，校验后原子保存；不复制远端目录。
python scripts/sync-v23-summary.py --host "$DEEP_PM_SSH_TARGET" --remote-root "$DEEP_PM_V23"
```

同步拒绝非允许路径、链接、逐源键、真实个人绝对路径及非有限JSON；取消操作者UID仅在公开副本中删除，私有原账本保留。同步回执绑定公开18文件SHA，后续重画必须更新它。[实际公开同步](../results/summary/v23_public_sync_summary.json)

本地验证：`PYTHONPATH=<compat>:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q`，274通过；新增自由4状态/共同平移、弱单历元、空模板、有限PSF范围与二维协方差回归。真实科学验证与本地数学测试分别报告。

## 当前结论与后续边界

真实40的12个信息通过不等于可采用新星。1/13框架支持、暗端偏差区间和邻源覆盖尚不满足全套要求，415新增拟合0。HST中央场未提供暗端真值。优先解决实际早期位置/邻源及恒星绝对转移证据，不自动重启g、形态、全域PSF或扩区。
