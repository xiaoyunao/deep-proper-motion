# SDSS + NSC 自行项目组会图

科学快照固定为 [2e4a04a](https://github.com/xiaoyunao/deep-proper-motion/commit/2e4a04a902151ea97e2dd2c3dba647efa9f92503)。2026 年 11 月 22 日暂停扩区和方法试验。本轮从签名目录、原始拟合成员及历史验证产物只读汇总绘图。每张图提供独立 PDF 和 320 dpi PNG。

## 组会主线

先用覆盖图和四路共同 QSO 精度图回答测到了什么。接着放两张 Aitoff 图并讲出精度收益与北向偏移的空间差别。全源对照、时序例图和条件分层解释目录一致性及斜率由哪些观测约束。最后用误差覆盖与历史留出年图讨论可使用的科学样本。

| 严格共同光谱 QSO | N | NSC DR2 | Xiao2025 | 物理质量 r-only | SDSS+NSC 全波段 |
|---|---:|---:|---:|---:|---:|
| 21 ≤ r < 22 | 79,938 | 15.316 | 8.552 | 7.786 | **6.882** |
| 22 ≤ r < 23 | 35,416 | 22.418 | 12.857 | 11.173 | **9.602** |

上表为同一行对象的二维绝对自行 P68，单位 mas yr⁻¹。与 Xiao2025 比两档分别改善 19.53% 和 25.32%，与 NSC DR2 比改善 55.06% 和 57.17%，与物理质量 r-only 比改善 11.60% 和 14.06%。Xiao2025 只用 PM_SOURCE=DESI 的地基自行。样本以已知光谱位置定义，覆盖度、盲目录身份与其他目录全源一致性均各有分母。 [冻结科学报告](experiment_A_v1.7_2154.md) · [合计统计](../results/summary/v17_2154_science_summary.json)

## 01　覆盖与源密度

J2015.5 的最终归属索引有 11,852 个有源像素和 11,862,169 个全波段解。ICRS Aitoff 全天图用 NSIDE=128 RING 像素的源数，色标取每像素源数的 log₁₀，显示范围 2.30–3.45，低端／高端分别有 989／133 个像素饱和。局部图显示零度接缝、北／西窗口和东部覆盖。名义面积 2,154.184 deg²，具参考及合格时基的几何机会为 2,132.865 deg²。有源像素包括部分覆盖单元，因此保留每像素计数与独立面积统计。 [覆盖及放大 PDF](../figures/meeting_2e4a04a/01_footprint_density.pdf) · [PNG](../figures/meeting_2e4a04a/01_footprint_density.png) · [开场足迹 PDF](../figures/meeting_2e4a04a/01_footprint_density_outline.pdf)

## 02　精度改善 Aitoff

四个面板用各 QSO 保存的 J2015.5 拟合位置在 NSIDE=32 RING 中分箱。颜色取相对改善率 $1-P_{68,\mathrm{all}}/P_{68,\mathrm{ref}}$。例如全波段 P68 为参考值的 80% 时，改善率为 20%。近白表示 0%，改善越大红色越深，恶化越明显蓝色越深。左列相对 Xiao2025，右列相对 NSC DR2。RA 向左增加，中心为 180°，两端是同一个 0° 接缝。灰色区分每像素 1–29 颗 QSO 及有目录源但缺少共同 QSO 的天空，其余留白。

21–22 等有 638 个像素达到 30 颗，共显示 75,904 颗，另 4,034 颗保留在全区统计。22–23 等有 446 个达标像素，显示 31,469 颗，另 3,947 颗保留在全区统计。Xiao2025 采用 −40% 至 +60% 的色标，NSC DR2 采用 −30% 至 +90%。每列上下两档及局部放大图共用各自的标尺。主图中，Xiao2025 两档超出下限／上限的像素数为 4／4 和 0／8，NSC DR2 为 0／3 和 0／3。NSIDE=64 局部图中，Xiao2025 两档为 2／1 和 1／0，NSC DR2 两档均在标尺内。色标三角代表存在超界像素，统计文件保留原值。小格分母是对象数，共享扫描和观测夜仍会带来相关性。 [主图 PDF](../figures/meeting_2e4a04a/02_qso_improvement_aitoff.pdf) · [PNG](../figures/meeting_2e4a04a/02_qso_improvement_aitoff.png) · [QSO 分母 PDF](../figures/meeting_2e4a04a/02_qso_improvement_qso_counts.pdf) · [NSIDE64 局部 PDF](../figures/meeting_2e4a04a/02_qso_improvement_aitoff_local_nside64.pdf)

## 03　随星等的精度、散布和零点

主图用相同对象的四路 P68、两轴 NMAD 和两轴中位。0.5 等细分箱的分母依次为 40,590、39,348、25,552、9,864，合并后逐值复现上表的原始范围。点状形态子图用 NSC r 的中位 CLASS_STAR ≥0.8。22–23 等的 15,691 颗点状形态 QSO 中，P68 从 Xiao2025 的 12.512 降为 8.619 mas yr⁻¹。已保存跨区域的严格共同对象仅覆盖 21–23 等。 [四路性能 PDF](../figures/meeting_2e4a04a/03_qso_precision_and_zero.pdf) · [PNG](../figures/meeting_2e4a04a/03_qso_precision_and_zero.png) · [点状形态 PDF](../figures/meeting_2e4a04a/03_qso_precision_and_zero_pointlike.pdf)

零点图用 02 的同一格网与分母，在 ±3 mas yr⁻¹ 标尺下显示全波段东／北中位，以及全波段中位减物理 r 中位。全区两档全波段东／北中位为 −0.192／−0.889 和 −0.307／−0.522 mas yr⁻¹；北向相对物理 r 分别移动约 −0.61 和 −0.57 mas yr⁻¹。零点图东／北超界像素分别为 20／58 和 27／26；两套中位差图为 1／12 和 7／10。部分北天窗的北向中位约 −2 至 −2.5 mas yr⁻¹，同期仍可在其他窗口看到明显的精度改善。 [零点 PDF](../figures/meeting_2e4a04a/03_qso_precision_and_zero_aitoff.pdf) · [零点变化 PDF](../figures/meeting_2e4a04a/03_qso_precision_and_zero_all_minus_r.pdf)

## 04　全部可唯一配对源的目录一致性

每套外部目录分别用所有通过该目录对应规则的源。横轴是外部目录自行，上排对比全波段自行，下排画全波段减参考目录。实线为差值中位，带状区为对象差值的 16–84 百分位。统一主窗口为单轴自行 ±50、差值 ±25 mas yr⁻¹，尾部图扩大至 ±200 和 ±100 mas yr⁻¹。图面标注每个面板的窗外源数和比例，配对阶段保留这些源。目录间共享原始观测，差值散布包含相关误差。绝对测量精度另见严格共同 QSO。

NSC DR2 的一对一物理成员对应共 **11,803,999** 个，其中两档暗端为 **3,312,780／1,342,362**。57,734 个源的检测横跨多个 NSC 已发布对象，另有 436 个预匹配源竞争同一个外部对象。全样本两轴差值中位约 −0.40／−1.66 mas yr⁻¹。主窗口下排两轴分别有 634,591（5.38%）和 906,318（7.68%）个差值或参考自行超界。 [NSC 主图](../figures/meeting_2e4a04a/04_all_sources_comparison_nsc.pdf) · [21–22 等](../figures/meeting_2e4a04a/04_all_sources_comparison_nsc_r21.pdf) · [22–23 等](../figures/meeting_2e4a04a/04_all_sources_comparison_nsc_r22.pdf) · [宽范围](../figures/meeting_2e4a04a/04_all_sources_comparison_nsc_tail.pdf)

Xiao2025 的地基子表共得到 **5,196,887** 个双向唯一位置对应，两档暗端是 **3,052,707／1,300,461**。现有缓存的 SDSS 编号无法直接对应。本轮核验长期保留归档中 MJD_DESI 的实际存储格式，天区缓冲内 94,079,364 条合格记录均用十进制年份表示 DESI 平均历元，与 SDSS 时刻及归档联合时基重算的差小于 0.02 年。将这些位置用 Xiao2025 保存的自行传播到 J2015.5，计算 1″ 球面位置候选。原 11,862,169 个科研解中，6,665,108 个缺少这一地基子表的位置候选，57 个有多候选，117 个有唯一候选但争用同一外部记录。匹配使用外部自行传播位置，保留由此带来的选择条件，两路自行差值未进入匹配判定。全样本东／北差值中位为 −0.17／+0.08 mas yr⁻¹。主窗口下排有 32,121（0.62%）／30,092（0.58%）个源超界。 [Xiao2025 主图](../figures/meeting_2e4a04a/04_all_sources_comparison_xiao.pdf) · [21–22 等](../figures/meeting_2e4a04a/04_all_sources_comparison_xiao_r21.pdf) · [22–23 等](../figures/meeting_2e4a04a/04_all_sources_comparison_xiao_r22.pdf) · [宽范围](../figures/meeting_2e4a04a/04_all_sources_comparison_xiao_tail.pdf)

## 05　存档时序与残差

6 个匿名例按观测结构预选，选择 20.5≤r<22.5、5–55 条已保存检测的候选，优先选接近 r=21.5、约 25 条检测的实例。依次为单 SDSS run 多季、多 run 长时基、追加 r 锚点、低稳健权重、高杠杆／短 NSC 时基和盲目录中唯一匹配的光谱 QSO。输入单曝光位置用保存的 C0 组零点校正，图中的蓝线和残差来自原拟合直线。纵轴为生产参考图 gnomonic 轴相对 J2015.5 的位移，标题给出标准 ICRS 自行与条件误差。点误差棒对应 W0 两轴位置误差，带状区域由保存的条件协方差传播。淡色空心点的稳健因子小于 0.5，原拟合仍保留这些测量。

有明确旧成员对应时，绿虚线为保存的 **legacy-quality r** 轨迹，物理质量 r-only 的逐源完整直线未保存。单 run 图附校零前位置。匿名例号的逐检测坐标、时序缓存及身份映射留在 NERSC。 [单 run](../figures/meeting_2e4a04a/05_astrometric_time_series_01.pdf) · [多 run](../figures/meeting_2e4a04a/05_astrometric_time_series_02.pdf) · [追加波段](../figures/meeting_2e4a04a/05_astrometric_time_series_03.pdf) · [稳健降权](../figures/meeting_2e4a04a/05_astrometric_time_series_04.pdf) · [高杠杆](../figures/meeting_2e4a04a/05_astrometric_time_series_05.pdf) · [光谱 QSO](../figures/meeting_2e4a04a/05_astrometric_time_series_06.pdf)

## 06　随观测条件的收益

分组图以零增益为参考线，分别展示对 Xiao2025、项目物理 r-only 和 NSC DR2 的二维 P68 降低百分比。每行标注对应严格共同对象。相对 Xiao2025，两档单 run 改善 12.8%／17.9%，多 run 改善 27.5%／35.6%；21–22 等且 NSC 时基不足两年只改善约 0.17%。按联合时基 12 年重新从逐对象值分层，四路分母为 23,865／56,073 和 10,604／24,812。联合时基与 NSC 自身时基用于刻画不同的观测信息。

按 SDSS run 或 NSC UTC 观测夜重加权的 95% 区间描述共享观测结构的敏感度。r-only 对照使用现有对象配对区间。累计相对 Xiao2025 两档对象区间约 19.0%–20.1% 和 24.5%–26.0%，按 SDSS run 重权为 16.7%–22.4% 和 22.9%–28.0%。纳入共享观测夜后，两档各连成一个大分量。条件分层还受实际仪器、天区、形态和样本选择影响。 [相对 Xiao PDF](../figures/meeting_2e4a04a/06_gain_by_observing_conditions.pdf) · [相对 r-only PDF](../figures/meeting_2e4a04a/06_gain_by_observing_conditions_vs_r_only.pdf) · [相对 NSC PDF](../figures/meeting_2e4a04a/06_gain_by_observing_conditions_vs_nsc.pdf) · [仪器分层 PDF](../figures/meeting_2e4a04a/06_gain_by_observing_conditions_instruments.pdf)

## 07　形式误差的覆盖

全波段严格共同 QSO 用原条件误差计算 z=μ/σ。实线以零为中心，虚线先移除该组的自行中位数再除以原误差，比较高斯参考覆盖。累计东／北 1σ 覆盖依次为 62.76%／65.38% 与 62.16%／66.25%，21–22 等联合时基至少 12 年时为 59.8%／63.6%。长时基组的实际散布更小，同时形式误差的覆盖更弱。公共误差、尾部和零点皆可能影响覆盖。 [覆盖 PDF](../figures/meeting_2e4a04a/07_error_coverage.pdf)

## 08　科研产出和观测支持

物理 r-only 有 8,845,816 个解，全波段有 11,862,169 个，净增 3,016,353（34.10%）。全波段分类为 8,831,540 个原有 r、2,943,661 个追加 r 锚点、86,968 个非 r 锚点。原有 r 类按与 **legacy-quality r** 解的至少 60% 物理成员重合定义；三类的构成和物理 r-only 总数差分别报告。保存的 r 星等来自 NSC r 光度的有 11,775,001 个，由 SDSS r 补充的有 87,168 个。

按冻结抽样标志，r 锚点充分／受限分别为 8,723,231 和 3,051,970 个。2,241,451 个解具有单物理组贡献至少 80% 斜率信息的标志，1,125,536 个解少于三个 NSC 季节。空间图把核心内的实际比例投到该核心有源像素，重叠像素按核心在像素内的源数加权。源级光谱支持和独立身份须各自核验，例如累计 22–23 等非 r 锚点只有 69 个盲后唯一光谱匹配。 [产出和星等 PDF](../figures/meeting_2e4a04a/08_yield_and_support.pdf) · [空间支持 PDF](../figures/meeting_2e4a04a/08_yield_and_support_spatial.pdf)

## 09　历史固定检测与独立年份

v1.5 重复 C 的固定对象为 347／181 颗。补入原 r 夜的其他合格波段后，两档 P68 分别由 3.582／5.542 降至 3.390／5.003 mas yr⁻¹。左图重复 C 包含只加每个新夜一次、加入新夜全部检测和加入同夜波段三种增量。右图保留普通 C 仅 13／5 颗的条件性结果。历史测试阐明了新时间信息与同夜位置约束的区别。 [消融 PDF](../figures/meeting_2e4a04a/09_historical_information_tests.pdf) · [v1.5 原研究](experiment_A_v1.5_nsc_information.md)

v1.6 重复 C 先从关联输入隔离 2016 年的所有波段。两档非 r 训练锚点独立年唯一重现 13/17 和 14/18，位置 P68 为 614 与 103 mas。圆点和方点分别是实际位置和第一个位置扰动负对照，区间为 Wilson 95%。这个历史小样本检验位置预测和成员的证据范围。 [留出年 PDF](../figures/meeting_2e4a04a/09_historical_information_tests_held_year.pdf) · [v1.6 原研究](experiment_A_v1.6_error_transfer.md)

## 需要讨论的误差来源

- 当前的 C0 用静止星系约束 SDSS run/camcol 和 NSC exposure 的二维常量零点，保留原发布坐标中的上游定标。颜色项、星等项、几何 DCR 和 CCD/field 多项式均属于历史试验。
- 每次 SDSS 物理 run 使用发布 r 位置，NSC 各波段按单曝光质量及共同参考支持进入拟合。W0 使用发布位置误差和每轴 50 mas 底噪。每源保留 J2015.5 ICRS 状态、稳健降权与 4×4 条件协方差，后者含同步参考星系抽样的传播。
- 输入轴误差沿用投影近似。r 锚点的 ≤10 mas yr⁻¹ 筛选作用于切平面分量，独立非 r 路线作用于标准 ICRS 分量。北向零点、同夜公共误差、非 r 源身份及高杠杆是讨论目录适用范围的重点。 [冻结方法](../docs/method_status_v15.md) · [状态和误差定义](../docs/catalogue_layout_v15.md)

## 复现与文件范围

从仓库根目录运行下列命令可重画去标识统计图。匿名时序图在 NERSC 直接从已保存检测成员、零点、权重和直线参数渲染。逐源和逐检测绘图缓存及例号映射留在个人 PScratch。

```bash
PYTHONPATH=src python -m deep_pm.meeting_figures \
  --summary results/summary/meeting_2e4a04a_plot_summary.json \
  --output figures/meeting_2e4a04a
```

在同一命令加上 `--set gain` 可单独重绘全天和局部的改善率图。

在 NERSC 已分配计算节点上，`meeting_stats` 的 `qso` 阶段重读光谱位置与冻结向量，`shard`／`catalogue` 核对最终归属和 NSC 成员，`xiao-index`／`xiao-shard`／`xiao-summary` 复现传播及位置配对。各分片都传入同一冻结归档及独立绘图工作目录，`--shards` 固定且 `--index` 覆盖全部分片。旧科研输出始终从签名索引读取。实测回执包括 QSO 61736396、目录分片 61736545、NSC 读出 61737669、时序图 61738105 及 Xiao2025 传播与配对 61737971／61738579。 [去标识绘图统计与校验和](../results/summary/meeting_2e4a04a_plot_summary.json)

前例样式参考 Xiao2025 的 Fig. 10 和 Tian2017 的 Fig. 5。图件均使用本项目冻结结果。 [Xiao et al. 2025](https://arxiv.org/html/2405.04016v2) · [Tian et al. 2017](https://arxiv.org/html/1703.06278v5)
