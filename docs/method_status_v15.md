# v1.5 实际科学方法与采用边界

这是科研试运行说明，不是正式自行星表发布。原v1.2／v1.3／v1.4回执、负结果、源码版本和所有原目录只读保留。

| 环节 | 本轮真实启用 | 尚未采用／证据边界 |
|---|---|---|
| SDSS位置 | 每物理run发布r位置，去同run重处理／相邻field重复，保留不同run；时间须核实为r参考 | 其余band不当独立位置；DR17普通扫描不冒充完整超新星Stripe82档案；缺原光度CSV不删除签名r天测 |
| NSC取数 | 全对象空间索引，无ndetr／rmag／nphotr筛选；分块COUNT和measid核验；7个真实filter有机会 | objectid只索引与外部NSC比较；core+halo及object中心选择仍有边界；完整exposure/CCD小区通过不等于图像完备 |
| 物理质量 | 合法坐标／实际TT时刻、正原生每轴误差≤200mas、SNR≥5、NSC SExtractor致命位252清零、原共同参考支持 | blend/deblend提示及形态未知单列；SDSS质量沿冻结r逻辑，不新增未核验flags解释；星系污染不是已测为零 |
| 参考与零点 | C使用原全区统一物理组C0，SDSS run/camcol与NSC exposure；原参考身份／规范／同步draw不变 | 新全band源不独立按core校零点。N1/N3仍共享已验证共同物理图的两个分量；A公共模式和拒绝组不强行恢复 |
| 误差与拟合 | 每轴发布误差+50mas底噪，TT Julian年，J2015.5，四参数稳健线性；保存ICRS全4×4及独立／参考切平面分量 | 夜共享误差仍未资格化；星系学习误差在独立年／QSO不统一，不采用；不把同夜band当独立长期约束 |
| 颜色与DCR | SDSS其他band仅光度／颜色诊断；一个DECam g真实天顶几何DCR已完整研究 | 同局部D0/D1与原共同C0分开；独立迁移未同时维持精度与零点，停止扩展，不进入连续目录。色误差已传播但SED和颜色测量偏差模型有限 |
| 其他位置项 | 只使用冻结C0物理观测组常数零点；实际仪器／CCD／field和像面位置可追溯用于诊断 | 连续主目录没有新增颜色、星等、CCD／field空间多项式或DCR；C1/C1b和v1.4经验g模型均保留历史正负结果，不默认为已启用 |
| 源关联 | 冻结1角秒常规位置轨迹；r锚点追加NSC合格band；另开无合格NSCr锚点轨迹，测量排他 | 高自行专项不在本轮。r原质量与物理质量控制分开，新增源身份不能由objectid或PM接近定义 |
| 年度验证 | B固定2017、北一固定2018；所有band测试年在盲关联前隔离。C代表区按元数据选2017 | 高重现率是条件位置证据，不认证绝对PM；真实CCD／参考支持／取回曝光／源SNR机会分开；平移对照不代表全目录污染 |
| 绝对比较 | 光谱QSO独立于星系参考训练；相同源比r／all／NSC及Xiao仅PM_SOURCE=DESI；盲输出另报 | NSC r≥21的Gaia PM退出选择与验收；未知暗星无真实PM真值；对象区间不替代公共观测组或空间质量 |
| 标准输出 | 逆gnomonic速度到每源mu_alpha*／mu_delta，位置—运动4×4Jacobian传播，物理组／band／仪器／夜／参考SHA可追溯 | 形式误差不是空间零点上限。科研合格解不是已认证星数，名义core面积不是实际目标SNR或认证面积 |

实际入口：`nsc_allband_inputs`、`nsc_exposure_complete`、`nsc_information`、`nsc_band_information`、`dcr_trial`／`dcr_validation`、`night_error_year`、`nsc_science_catalogue`、`allband_blind_year`、`allband_member_audit`、`nsc_science_summary`、`nsc_opportunity_area`。均通过Slurm执行；公开只同步经过审阅的汇总和科研图。

年度三类源仅由训练年份中的r支持定义，不与完整全历元目录的三类数量互换。取数补齐和使用已取回的其他band分别计数；同一物理成员重放显示原问题源的PM未因取数变多而改变。全部源协方差条件于已选参考规范和发布参考系，不声称已包含未知绝对frame tie、空间残余或全部跨夜系统项。


## 本轮结果冻结

连续C全部91核心／99.873124平方度，481945个全NSC科研解。原r质量控制384234，物理r控制漏斗384219（原锚点归属的逐core漏斗保留），全NSC全区增25.43%。后处理用J2015.5转移5个全band及2个r控制owner；所有状态／协方差／成员原样保留，仅实际冻结核心参与。每core源数／band／时间统计按新owner重算，输入和固定已知QSO统计仍对应原声明footprint。读取需私有签名overlay，见[catalogue_layout_v15](catalogue_layout_v15.md)。

单SDSS run QSO两档P68较r改善17.5%／22.1%，但C的NSC采样不能代表所有普通区。DCR／学习误差继续不采用，尚不开展全覆盖正式生产。实际证据、旧r未恢复、非r样本不足和公共误差限制见[最终报告](../reports/experiment_A_v1.5_nsc_information.md)。
