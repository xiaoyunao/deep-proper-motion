# SDSS + NSC 多历元自行

项目研究较早的 SDSS 单历元位置、独立的 NSC DR2 单曝光位置与联合天测校准，能否改善普通恒星和缺少 Gaia 自行的暗星测量。Xiao et al. (2025) 用 SDSS 与 Legacy Surveys 的两端位置估计自行；本项目检验多历元采样和公共位置误差的独立贡献。 [Xiao et al. (2025)](https://doi.org/10.3847/1538-3881/adb72c) · [官方 NSC](https://datalab.noirlab.edu/data/nsc)

**当前阶段 v1.5（2026-11-19）**：按用户新决定，SDSS每个真实物理run只使用发布r位置，保留多run；其他band仅颜色／光度／DCR诊断，停止SDSS五band方法D。NSC全部实际合格band进入候选。基于b23f8ef，先固定r／SDSS身份重放、完整小区exposure/CCD补齐，分新夜与同夜信息，再做一个几何DCR及颜色误差检验；共同参考和新增源零点仍需独立评价。用户已授权约10×10度连续**科研试运行**，与正式生产验收分开，完成小区核验后执行。[当前八步计划](PLAN.md)。

本轮已核验B完整对象输入757/757曝光小区检测，新增55725条NSC而SDSS r保持；冻结ABC显示新夜／时基增加不保证最暗绝对自行改善。唯一g几何DCR及星系学习误差均未获一致独立验证，不采用。4节点61565637执行连续C普通先导及99.873平方度科研试运行，使用原统一C0/W0全NSC，正式采用与空间零点尚未认证。[本轮报告](reports/experiment_A_v1.5_nsc_information.md) · [实测图](figures/v15_information_dcr_error_summary.png)。

**历史v1.4实测判断（2026-11-19）**：全波段有科学信息收益，但未冻结采用模型。184核心1.093亿检测核查覆盖七种NSC band；N1/N3共同物理组可复现一致参考，A高SE退出主要公共模式仍未采用。固定B暗星21–22／22–23留出位置P68改善15.7%／19.0%；辅助B区83颗21–22 QSO自行P68 5.345→4.396（对象区间改善4.9–28.2%），其余区域不统一。独立星系支持的单一g颜色项已实际共同求解并传播同步公共协方差，改善北一／北三若干年度位置预测，但新增北一24颗QSO北向−2.167→−2.189mas/yr，零点未解决。SDSS r发布误差实测对应质心项，场级／跨band公共误差需另建；非r索引确实遗漏测量。**保留C0/W0 r-only；不启动约100平方度完整multi目录或全覆盖。** 下一优先是参考颜色／真实DCR几何及SDSS逐band共享误差与曝光路线非r取数，不按待验收QSO调系数。[中文科学报告](reports/experiment_A_v1.4_allband.md) · [证据图](figures/v14_independent_evidence_summary.png) · [八步验收状态](PLAN.md)。

- **实验 A** 用公开目录位置，评估检测关联、自行精度和系统误差。v0.2 的定向高自行留出恢复 19/24，纯位置基准为 0/24 和 3/24；普通代表场的运动法受大型冲突影响。 [v0.2 报告](reports/experiment_A_v0.2.md) · [冻结提交](https://github.com/xiaoyunao/deep-proper-motion/commit/b92912c220252349a6050ef25cdab6e711d37ded)
- **实验 B** 比较同曝光的发布位置与自由 PSF 质心，并将新图像历元的贡献单独测量。目录级剩余误差能否稳定归因到质心条件，决定是否开展小规模图像对照。
- **v0.3** 四区 214 颗亮星中，最早一次 SDSS 降低四场的北向 NMAD。固定赤道 47 颗星与检测，M3 将北向偏差从 −4.66 降到 −1.33 mas/yr。额外的颜色和 DCR 项在独立验证中未稳定改善。赤道 64 条 (r=21)–23 轨迹的留出位置没有稳定绝对误差收益；6 个光谱 QSO 的自行对照仍太小。 [v0.3 报告](reports/experiment_A_v0.3_precision.md) · [冻结提交](https://github.com/xiaoyunao/deep-proper-motion/commit/9eacbec17f7c31041ff55be05debe277fd10dd88)
- **v1.2 冻结结果** v1.2 完成连续 C 区 99.873 平方度，91 核心共同参考图和 384,232 个 W0 解均通过回执。2,267 个最暗三路同源 QSO 的 NSC／Xiao 地基／C0 P68 为 14.04／13.61／10.79 mas/yr。相对 Xiao 的改善为全区 20.7%、重复带 44.6%、普通区 4.2%，普通区区间跨零，21–22 等出现 4.2% 恶化。单连通参考图及所有核心已完成，空间零点、边界和独立暗恒星真值仍限制正式发布。下一轮优先在独立天区复制条件性收益及定位局部残差。 [最终报告](reports/experiment_A_v1.2_continuous.md) · [配对图](figures/v12_continuous_paired_precision_summary.png) · [整区天测图](figures/v12_continuous_science_summary.png) · [边界图](figures/v12_continuous_boundaries_summary.png) · [完整统计](results/summary/v12_full_continuous_summary.json.gz) · [完成回执](results/summary/v12_full_completion_summary.json) · [几何审计](results/summary/v12_full_ownership_summary.json) · [计划](PLAN.md)

- **v1.4方法状态** 全NSC band候选已实际用于固定源、盲新增和非r锚点研究，不再限制为旧g/i/z；全方差／公共夜模型、经验颜色项、A支持恢复均保留为研究分支，未替代冻结C0。SDSS其他band恢复出真实质心、时刻与原生二维误差，但独立坏标志／跨band共享误差未资格化，D未进入采用方案。原结果与失败回执不覆盖，不将旧north_two或预先读过场区称新盲区。[实际启用审计](docs/method_status_v14.md)

- **v1.3 冻结八阶段研究** 冻结91核心10,438 QSO及7071条原控制解零差；普通区长基线收益显著、重复SDSS增量随观测支持变化。两旧区及三新核心的误差／多波段模型均未获采用。新增预登记226.963平方度256核心全部完成并按冻结规则解封58.720平方度保留窗，共792,656个W0、严格共同QSO7717／2288。单SDSS run相对Xiao两档约差6.8%／4.9%，多run改善20.4%／26.9%；保留区21–22单run约差11.7%，多run改善9.9%。真正跨独立图的1332个相同检测源显示局部1–2mas/yr差异，A12核心11.970平方度无保留参考支持，非零源核心214.993平方度不等于图像完整覆盖。整年预测及实际合格曝光足迹已审核，N2低重现主要不能用全部几何覆盖作分母。全256核心归属、重复均通过，但空间零点／跨图质量未认证。**决定不启动全覆盖正式生产，保留C0研究产品；先联合跨区物理参考组并追溯支持稳定性，不自动进入大规模PSF或高自行。** [最终中文报告](reports/experiment_A_v1.3_conditions.md) · [最终三目录图](figures/v13_final_independent_comparison_summary.png) · [跨图差分](figures/v13_cross_window_summary.png) · [完整统计](results/summary/v13_final_independent_summary.json) · [计划](PLAN.md)

- **v0.4 亮星同源复核** 183 颗有共同 M3 支持。二维绝对误差第 68 百分位相对 NSC 在北天改善 87%，拥挤区改善 32%，两区的对象配对区间排除零。赤道改善 8%，区间跨零，北向点估计仍较差。Pal 5 的 20 颗用 M1 可改善 66%。 [v0.4 报告](reports/experiment_A_v0.4_nsc.md) · [亮星图](figures/v04_bright_nsc_summary.png)
- **v0.4 暗端 QSO** 三个新天区共有 162 个严格共同源，97 个在 (21\le r<23)。M3 把二维绝对误差第 68 百分位从官方 NSC 的 8.23 降到 3.53 mas/yr，改善 57%，对象配对区间 43–64%。两处 Stripe 82 共用 17 个 SDSS run；考虑观测块后改善仍为正。暗端北向中位差为 −0.67 mas/yr，22–23 等只覆盖一个观测块。扩大 A 的独立场和暗恒星验证，暂缓全巡天与 B。 [v0.4 报告](reports/experiment_A_v0.4_nsc.md) · [暗端汇总](results/summary/v04_qso_pool_nsc_pool_v04_297ee541e3c9_3daf6332d85a_summary.json) · [汇总程序提交](https://github.com/xiaoyunao/deep-proper-motion/commit/297ee541e3c97694cc358b485c26a467f8abdfe3)
- **v0.5 试生产** 五场 6.04 平方度，输出 22,802 个盲关联自行解，NSC r=22–23 有 2,016/3,529 个候选通过（57%）。89 个与 Stripe 82 无共享扫描的 22–23 等 QSO，已知位置同源 P68 从 16.15 降到 9.22 mas/yr（43%，观测块区间 16–69%）。最暗端盲目录仅 25 个同源 QSO，东／北零点仍有超标，暂无独立非 Gaia 暗恒星完整真值表。继续定向验证，暂缓 100–300 平方度。COSMOS 全场取数超出试验吞吐，按预先登记的 G09 替代。 [报告](reports/experiment_A_v0.5_pilot.md) · [试生产汇总](results/summary) · [关联筛选图](figures/v05_production_funnel_summary.png)

- **v0.6 权重与物理模型** r 波段 89 颗最暗 QSO 的 W1 相对 W0 配对区间在三个普通场均跨零。W0 继续作为普通源四参数基准，输出质量位、有效历元和删组敏感度。五个普通场近星只有 2/17 支持自由视差试验，暗源档案没有独立证据可推广 L5。 [v0.6 中文报告](reports/experiment_A_v0.6_astrometry.md) · [计算提交](https://github.com/xiaoyunao/deep-proper-motion/commit/7c7935a751ed364b7bcf06f600fce16d8eec6d22)
- **v0.6 覆盖与下一步** 6.04 平方度内，35,858 个普通 NSC r 锚点中 17,083 个有重复 SDSS。固定 W1 后多 NSC 波段把 22–23 等合格试验解从 2,002 增至 2,532。各场 QSO 精度和零点表现仍有差异。针对新增多波段暗源及场零点做独立验证，扩大生产和实验 B 暂缓。 [候选源数图](figures/v06_multiband_yield_summary.png) · [SDSS 扫描统计](figures/v06_sdss_scans_summary.png)

- **v0.7 分区验证** W0 输出 132,045 个试验解。140 个最暗共同盲 QSO 在三个独立观测块上，官方 NSC → W0 的径向 P68 为 18.63→6.68 mas/yr，改善 64.2%，块区间 59–83%。旧 v0.5 最暗的 89 个 QSO 中，62 个在点源形态筛选退出、2 个在拟合质量退出，25 个保留。 [v0.7 报告](reports/experiment_A_v0.7_regional.md) · [空间零点图](figures/v07_spatial_zero_summary.png)
- **v0.7 下一步** 22–23 等固定 W1 加 NSC g/i/z 后有效源 13,631→16,189。整年留出的 W1 原来无合格解的 7,058 个多波段训练候选中有 5,363 个唯一 r 位置重现。零个区域满足全部冻结门槛。优先复核空间零点、独立暗恒星和新增多波段源，100 平方度及图像级暂缓。COSMOS 的 0.12 平方度分片续取通过，取得 211 万条测量。 [多波段和留出图](figures/v07_multiband_holdout_summary.png) · [下一步计划](PLAN.md)

- **v0.8 旧区审计** 507 颗严格的 r=22–23 QSO 中，140 颗保留，360 颗止于形态筛选，7 颗止于拟合质量。放宽门槛增加已知点源恢复数，极暗 Gaia 验证及星系污染证据仍有限。north_c 与 east_d 的偏差在场内显著，局部空间梯度对独立 QSO 无稳定改进。 [v0.8 报告](reports/experiment_A_v0.8_spatial.md) · [验证协议](docs/spatial_validation_v08.md)

- **v0.8 第一批扩区** 新增 21.295 平方度和 84,101 个 W0 解。83 颗最暗共同 QSO 的 NSC→W0 P68 为 25.48→7.60 mas/yr，六个物理块区间 59%–85%。正式零点区间仍宽于阈值。 [v0.8 报告](reports/experiment_A_v0.8_spatial.md)

- **v0.8 累计结论** 502,422 个 W0 解，共同最暗 QSO 473 颗，官方 NSC→W0 二维 P68 23.91→7.32 mas/yr，五个物理块的改善区间 65%–86%。暗端 W1 多波段产出 +37.9%，整年留出重现 24,600/31,506。60 个核心的正式高精度验收为 0，仍缺少验证暗恒星的独立星历。继续目录级 A，先定位天区与恒星颜色的零点差异。 [v0.8 中文报告](reports/experiment_A_v0.8_spatial.md) · [空间零点图](figures/v08_spatial_field_zero_summary.png)

- **v0.9 曝光级试点** 两区 7.990 平方度，C1 在留出星系的跨巡天位置 P68 改善 5.2% 与 7.5%。重复丰富区 164 个 r=22–23 严格四方共同 QSO 的 NSC／Xiao2025 地基／C0／C1 径向 P68 为 9.13／10.63／4.84／4.22 mas/yr。稀疏区非劣效下界仍超出冻结容许范围，暗端零点与探测器边界需要独立验证。保留 C0 为一般基准、C1 为重复丰富区域选项，暂停连续 99.873 平方度扩区。 [v0.9 中文报告](reports/experiment_A_v0.9_exposure.md) · [四方比较图](figures/v09_exposure_motion_summary.png)

- **v1.0 C1 适用范围** 新增 5.985 平方度，两处 413 个严格 21–23 等 QSO 的径向 P68 为 12.88→11.25 mas/yr，按共享扫描和曝光为两个独立块。四处真实 SDSS 相邻 field 的位置差均随 C1 增大。东南相邻两块共用 50 个物理零点组，一次共同拟合可行。C0 保留为一般基准，C1 仍为精度候选。Gaia 暗端退出评分；两个新增区 2017 年留出重现多波段独有最暗候选 502/613 和 227/229。相邻 field 约束、正式零点和独立暗星证据是连续区投产前的关键问题。 [v1.0 中文报告](reports/experiment_A_v1.0_validation.md) · [边界图](figures/v10_boundary_summary.png)



当前主workspace遵循用户确认的个人PScratch，后续新计算代码和数据不迁入Home；精确路径保留在未提交配置。正式源级／区域级误差与系统质量字段的现状和缺项见[质量元数据](docs/quality_metadata_v13.md)，不能把源级形式误差等同于空间零点认证。


## v1.4 回执与数据边界

主要计算回执及正负结果见本轮报告末节和WORKLOG。61557533两年度接口失败已在61557949新版本续验成功，原参考解未重做；61556295的A控制失败已续验且严格固定r对象比较完成。61558069确认颜色模型没有解决新增源零点。原始、逐检测、逐源、参考星系和私人对照均保留在个人PScratch，公开仓库仅代码、文档、图与去标识统计；无新增Home数据或HPSS前置。
