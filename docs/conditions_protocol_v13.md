# 观测条件与独立天区的预登记

基准 `13d4e29` 的99.873平方度和C0/W0参数冻结。首次新结果读取前提交此协议及配置。

## 同源消融

只用去重DR16Q+DESI DR1的光谱QSO和真实单曝光，NSC r≥21 Gaia运动不参与评分。按联合谱表 reference_r 的20–21、21–22、22–23分档，NSC r用于恒星产出，不混淆星等来源。每个对象固定原500mas唯一匹配与300mas竞争间隔定义的检测，限制相同参考图分量，NSC至少0.6yr、联合至少5yr。全部NSC检测不变，只改变加入的SDSS集合。没有为任一方法单独按其误差重新挑对象。独立列出与NSC、Xiao DESI共同源及全部合格QSO。

比较固定C0下NSC-only／一次最早SDSS／全部SDSS，并以原位置全部检测控制组常量校正作用。报告绝对二维P68、两轴偏差／NMAD／尾部、误差覆盖、独立组、加权有效历元和时间杠杆；有效组=1/sum(组斜率信息份额²)，不把同晚多曝光当作独立长期信息。全部SDSS臂与v1.2逐对象、曝光集合、检测数及自行逐项核对；不一致先修验证。

按冻结地理普通／过渡／重复带及实际单／多SDSS run、NSC时基短／长分层。对象bootstrap作为描述性区间，共享run／NSC曝光／夜的同步权重敏感性另外输出，闭包和样本数保留，不称这些权重区间为全区空间独立验证。

## 两处普通区的后续试验

在读取消融结果前固定 `repeat_sparse` 和 `sparse_north_outer`，均为真实普通扫描区，不按QSO表现挑选。原始测量／C0不变。最小误差模型以巡天／仪器／波段划分，只允许在足够独立参考对的训练证据下拟合比例与底噪；参考组的同步抽样保持，独立与公共误差分列。不得只为使标准化残差等于1而缩误差。

参考源按物理身份固定训练、留出；模型拟合不得使用验证QSO的运动、最终留出年检测或最终新区结果。若原训练参考用于误差学习，必须用未参与任何学习的留出星系验收并报告原组零点拟合相关性。均值修正只在星等、颜色、seeing、airmass、时间或仪器残差具有训练信号且独立验证复现时增加；DCR元数据不足则只诊断。多波段先在原r样本固定对象比较，额外源另行记录训练分母、留出机会、唯一匹配、零点、污染和有效源数。

## 独立区域候选池

`independent-v13-candidates.toml`登记A/B及两处北天候选窗，几何面积合计约366平方度。候选不是已验证覆盖或最终新增面积。先查SDSS真实run／场足迹、NSC芯片多边形、仪器、实际时间跨度、星系支持和光谱QSO位置，排除所有旧试验足迹，按真实可用条件选择约200–300平方度。尚未提取新区自行，不能按表现选区。

A/B已有v0.9纯几何预查可复用，旧试点落在窗内的面积不能算独立。北窗用于检查90Prime／混合仪器，是否满足曝光数和时基由实际查表决定，不保证存在。先完成首批真实取数、参考图、共同QSO／星系与资源审计，之后再启动下一批。

## 科学采用与生产

最小误差或多波段模型须在两处普通区的独立QSO、留出星系和整年预测中取得可复现收益；反向偏差、尾部或额外源污染不得隐藏。失败、样本不足和缺元数据都保存。每轴0.5mas/yr为高精度目标，按能支持区间的空间尺度认证。跨大区要绑定投影中心、标准ICRS与完整协方差、共享物理组及弱支持标志。阶段8依据独立新区证据选择单C0、分层模型或质量分区，并在支持广域产出时开始分批有效覆盖生产。

## 当前核验与复现入口

91核心消融已完成；原7071条21–23等控制轨迹零差。`independent_groups`是不同校准组的数量，`effective_weighted_epochs`是检测Kish数，`effective_groups`是按校准组斜率信息的Kish数，均不等于独立年份；同夜关联另外检查。归档SHA已复核。第五北窗加入后几何候选226.963平方度，实际run复用及星系支持先登记再作新PM。

在Slurm计算节点、相应不可变提交目录中执行，路径由未提交的个人配置提供：

```bash
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m deep_pm.continuous_conditions "$WORK" "$SOURCE" "$SHARED" "$XIAO_CSV" configs/continuous-v12.toml "$CORE"
python -m deep_pm.continuous_condition_pool "$FROZEN" "$ANALYSIS" "$PUBLIC_SUMMARY"
python -m deep_pm.condition_error_trial "$WORK" "$SOURCE" "$REFERENCE" "$SHARED" configs/continuous-v12.toml configs/conditions-v13.toml "$CORE"
python -m deep_pm.condition_residuals "$WORK" "$SOURCE" "$SHARED" configs/continuous-v12.toml "$CORE"
python -m deep_pm.continuous_fit "$WORK" "$CORE" "$SOURCE" "$SHARED" "$XIAO_CSV" configs/adjacent-v13.toml --diagnostic-covariance
python -m deep_pm.adjacent_consistency "$LEFT" "$RIGHT" "$PUBLIC_SUMMARY"
python -m deep_pm.condition_trial_compare "$TRIAL" "$CONDITIONS" "$SOURCE" "$PUBLIC_SUMMARY"
python -m deep_pm.independent_runs "$GEOMETRY_SUMMARY" "$SDSS_WINDOW" "$WORK"
```

各阶段使用独立WORK；新旧配置、提交或拟合选项不能复用写入同一目录。本地重绘只读公开汇总，完整测试限制原生线程与NERSC一致，避免本地sklearn多线程环境崩溃：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m deep_pm.condition_plots results/summary/v13_full_conditions_summary.json figures/v13_full_conditions_summary.png
```

## 整区独立测试区域保留

在查看任何A/B/北窗新自行结果前，保留全部已选 `north_two` 区域作为最终独立测试（空间旧试点已在选区中剔除）。首批可取数、解参考图并写密封结果，但模型选择期间不查看该区QSO／Gaia自行、残差、留出和产出指标。先从其他四窗及既有C区冻结采用方法、条件分层和质量规则，再统一解封，不根据测试结果回调参数。保守共享run清单与实际使用组仍单列，空间分离不等于统计完全独立。

### 首批标准坐标汇总

`python -m deep_pm.independent_summary STATE configs/independent-v13-candidates.toml OUTPUT` 在读取拟合／参考文件之前剔除整个 north_two；不可用普通整区池化代替此保护入口。已保留的切平面运动通过每源逆gnomonic正向Jacobian恢复ICRS；早期私有NSC/Xiao的逆矩阵转换字段存在方向错误，汇总忽略这些字段重建，原冻结科学结果不改变。公开同时保留并明确标记旧切平面控制指标。

最终新区汇总分别给出已知位置控制与W0盲产物。盲源以拟合完成后检测级标签审核映射到源级标准ICRS输出，在同一恢复子样本对照已知位置拟合及两发布目录；未知成员和形态／时间／误差选择不能包装成关联正确率。历史缺失映射标为不可评估。连通观测块少于3时不提供块重采样空间区间；固定参考解和同步参考抽样的不确定度仍分别说明。资源入口以DEEP_PM_BATCH_NODES和各阶段DEEP_PM_*_STREAMS记录并发，不改变科学参数；网络取数避免无界并发。
