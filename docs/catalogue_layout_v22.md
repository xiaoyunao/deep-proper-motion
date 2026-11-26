# v2.2 私有科研产物的可追溯布局

本文件是实际代码/阶段产物说明，不声明全域目录已经完成。个人绝对路径只在不提交的配置中。

- `registration/private_full_coordinate_plan.json`：原2875核心、签名输入、原生产/参考、冻结物理几何与旧图心。新数据不覆盖旧2751.446deg²生产。
- `input/<core>/reference.parquet`：球面参考入口，包含真实检测、参考身份hash拆分、物理组和原生角误差。全域去重、剪枝、连通及冲突回执位于`graph/`；halo测量与唯一物理测量分别统计。
- `reference/component-XXXX/private_c0_unfiltered.npz`：完整分量全部剪枝后组的二维offset、48对象同步扰动、SE前NSC中位gauge集合签名。NPZ内SE不是统一图平面12mas筛除；每条目标实际位置逆Jacobian后两轴≤12mas。
- `sources/<origin_core>/private_standard_motion.json.gz`：源级J2015.5 ICRS位置、自行、完整4×4协方差、独立与参考协方差、稳健权重、物理检测/组/曝光/CCD/field/band/instrument/UTC18夜及TT MJD、局部参考SE、真实时间跨度和完整二维斜率信息。源身份和组支持冻结后才查光谱。原位置/图心/矩阵足以重建同步参考响应；光谱子集另外保存48×4响应。
- 状态协方差顺序为`[alpha_star_position_mas, pmra_masyr, delta_position_mas, pmdec_masyr]`，位置块是该源局部东/北角扰动，RA/Dec中心另以度保存。它不是直接对`ra_deg`的平方度矩阵。交叉项的量纲为mas²/yr等，不能只取两个PM误差替代完整状态。
- 每个源保留`reference_component`和`reference_model_sha256`。48draw只在同一完整分量内同步，不同分量公共误差独立合成。条件协方差不含尚未认证的夜相关、颜色或空间系统项；不宣称总绝对误差。
- `sources/<core>/private_fixed_blind_qso.json.gz`保存所有原盲目录光谱身份及新支持/形式误差/残差退出。`private_blind_qso.json`为新盲目录独立后匹配；两者分母不混用。Gaia暗端自行不参与排名。
- `catalogue/index/<core>/science.parquet`为流式私有索引，10个上三角`cov_00`至`cov_33`恢复4×4，另保存坐标/自行、参考SHA、原始源指针、J2015.5 owner、源类、时间/夜/波段和质量标志。索引不复制全部逐检测数组，完整成员从原指针检索。
- `catalogue/catalogue.duckdb`的`accepted`为最终共同历元保留索引。原计算核心只是计算来源；真实内部边界移动后重新归属，外部边界解另存。精确成员重复只有在位置/自行/协方差/owner/参考一致时保留单个确定代表；其他共享检测竞争源进入`competing`，原始拟合均保存，不根据QSO挑赢家。
- `continuity/`保存全域物理检测交集、双向唯一且各≥60%成员连续性。精确共同、改成员连接、无法连接及竞争分别记账；不能把新成员集直接当新恒星。
- `year/`保存先隔离整年全部目标band再训练/关联的原新共同竞争。真实CCD/depth/支持机会和位移负对照保持独立分母；1arcsec唯一重现是条件位置证据，非恒星正确率。
- `area/`为真实SDSS扫描、NSC CCD、native参考和时间支持的两分辨率等面积积分。源占据HEALPix数、几何机会面积、实际源SNR合格面积及空间零点认证面积不等同。

所有上述逐检测/逐源/验证/参考产品仅留NERSC PScratch。公开仓库只保存审核后的聚合、图、代码与中文判断。源级`spatial_zero_uncertified`、`night_correlation_unqualified`、身份未认证、non-r及高杠杆等标志不因数值计算成功而删除。弱科学质量不阻断其他完整有效区域。
