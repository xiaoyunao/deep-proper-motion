# v2.1 实际坐标、权重与门槛审计

基线 `6faabde`，原输出只读；以下修复只用于登记核心的独立诊断版本。

| 层与实际入口 | 量及单位 | 冻结旧实现 | v2.1 对照 |
|---|---|---|---|
| `normalize.sdss` | 发布 r RA/Dec；TAI_r→TT MJD；RAERR/DECERR arcsec→mas | SDSS RAERR 已含 cos(dec)，不再乘；r offset 必须为零；RUN_PRIMARY保留真实独立run | 不变，每run仅r位置 |
| `normalize.nsc` | 单曝光RA/Dec；UTC MJD→TT MJD；raerr/decerr arcsec→mas | 沿用已冻结局部东/北解释；未发布原生相关系数近似0 | 不再乘cos；保留单位局限与全部实际合格band |
| `__main__.detection_arrays` / `quality_detections` | native E/N sigma和每轴50mas底噪 | sigma=sqrt(native²+50²)，然后直接交给全图平面 | 底噪只加一次，`coordinate_metric.projected_covariance` 将完整 Csky 前向投影 |
| `astrometry.project` | gnomonic x/y，mas | 位置已带secθ/sec²θ伸缩 | 不改位置定义；A=d(x,y)/d(E,N)，Cplane=A Csky Aᵀ，保留非对角项 |
| `continuous_reference.extract`→`zero_point.fit_offsets` | 星系xy及物理group偏移，graph mas | 旧图权重使用未经投影sigma | B1/B2源重放固定旧图；参考权重另做固定边界2D/对角对照，不冒充全连通图重解 |
| `read_offsets` | C0 point及48同步bootstrap draw，graph mas | draw已是全图平面量 | B1直接过旧H/J，common数值保持；禁止再乘A。B2过新H/J。独立参考域分开 |
| `coordinate_metric.fit_vector_motion` | 四参数(x0,mux,y0,muy)，mas/mas yr⁻¹ | 旧逐轴对角IRLS | 完整2D GLS，原min(1,6/(4+d²))损失及8次源迭代，d为Mahalanobis。W0底噪不调 |
| `standard_motion.icrs_motion` | 每源ICRS位置、μalpha* / μdelta、4×4局部状态协方差 | 已采用完整逆投影Jacobian | 保留；输出顺序(δα*,μα*,δδ,μδ)，单位位置mas、自行mas/Julian yr；完整位置-运动交叉项 |
| `source_funnel` / `augment_tracks` / `select_catalogue` | 关联1arcsec、QSO0.5arcsec与0.3arcsec竞争余量 | 在大gnomonic中直接用mas圆；r误差门槛与non-r轴不同 | B2_selected先仅统一2D质量/ICRS误差；B3再用3D单位矢量AngularTree实际角距离。原形态0.8保持 |
| `precision_v03.reference_rows` | 星系隔离、Gaia/QSO位置排除、SDSS–NSC参考匹配 | 同一全图平面半径 | B3独立统计球面入口变化；新参考身份不暗中替换冻结C0 |

实际gauge不是某一曝光固定为零：`continuous_reference.solve`先对每个连通分量减去NSC组point的中位数；每组draw先减自身draw均值，再减NSC组的逐draw中位公共模式。此归一化发生在12mas参考组SE筛选之前。序列化后没有单一零曝光是正常结果。有界试验固定外部旧offsets，继承其gauge；全图重解须登记筛选前NSC物理组集合与这一同步归一化。旧12mas参考SE门槛实际作用于图平面，不能称为每处相同的ICRS角误差；本轮不暗中重选旧C0支持组，原生角单位的参考SE诊断另列。

## B1 的两种误差解释

旧成员、点估计、IRLS权重w和影响H固定。估计器降权但噪声仍为C：独立协方差 H C Hᵀ；把稳健降权解释为噪声膨胀：H(C/w)Hᵀ。两者是不同噪声假设，均保存。新C只替换独立测量项，旧公共C0项保持数值原值。不能把两种IRLS解释之差称为几何精度收益。

## 参考试验边界

从三个先导核心的已签名原参考shard选2个支持最多的SDSS组和最多4个NSC组（训练参考≥30），其他旧C0组固定。训练参考必须同时连接自由组与固定边界，最多1200个按身份哈希选取。静止星系位置和二维组增量联合求解；对角/完整2D使用相同成员、相同边界、6次稳健更新+最终求解、48次配对参考对象指数抽样。留出星系不训练。结果误差条件于外部边界，不能与旧全球draw按索引相加或当独立项叠加。

## 时间与恒星位置预测

发布坐标是天球参考方向，J2000不是观测时刻。源拟合使用逐检测TT实际年份与J2015.5四参数线性模型。亮Gaia控制按固定身份补取五参数误差和十个相关系数；逐测量TT转换TCB，与Gaia ref_epoch差进行直线空间运动，使用Astropy builtin地球太阳系质心AU位置计算年周视差。原发布ICRS坐标不重复添加光行差、岁差或大气折射。负测量视差用0的物理近似、缺RV用0均显式标记；RV误差及地表观测者视差未传播。标签只用于验证，未用于关联。

SDSS发布r天测已经包括颜色处理，现有信号应称为剩余色差。[SDSS官方天测说明](https://www.sdss4.org/dr17/algorithms/astrometry/)。没有真实天顶方向，不能仅根据颜色和气团相关直接认定DCR。

## 实际参考星系入口

先在预过滤的检测中，用NSC辅助对象索引取15–23等测量，按NSC已有对象标签计算静态位置中位数及r形态中位数；r形态≤0.85、邻居隔离≥1200mas、Gaia和光谱位置排除≥1500mas。SDSS要求type=3、nchild=0、16–22等、0≤magerr≤0.15，距参考对象≤500mas；同参考/曝光的重复记录排除，必须同时有两巡天。NSC object ID在此为静止参考索引，不是目标盲关联真值。固定身份哈希约30%留出，训练组最低10个独立参考、每参考至少2个物理组，保留SDSS–NSC连通分量。与Xiao2025每HEALPix至少800参考的旧框架不同；这里的HEALPix仍为组织和诊断单位，校准参数按run/camcol和exposure。

## 数学验收与物理模型

前向/逆向Jacobian、RA wrap、高赤纬、相关协方差、GLS退化到旧对角解、参考公共模式及固定边界都有针对性测试。同一个物理状态及协方差重表达须满足1e-3mas/yr与1e-3相对协方差容差。把不同平面内分别拟合直线是不同近似模型；另列曲率，不与坐标表达误差混为一谈。现有普通源30mas/yr模拟在约21yr采样中不同平面差最大约1e-4mas/yr，2000/10000压力样例差可显著增大；本轮不认证高自行。
