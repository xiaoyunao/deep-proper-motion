# 2025 年工作的只读审计

在 NERSC 的个人 Scratch 找到 `proper_motion` 工作树。发现 `final_catalog`、`raw_data_128`、`data_128`、`quasar_128`、`cluster_128`、`distant_star_128` 和 Stripe 82 验证产物目录。限定检查个人 HOME 及旧项目程序目录。程序树只找到环境残留和空的程序、缓存与 notebook 检查点目录，未恢复科学源码或配置。本次未从结果目录批量反推来路，未复制旧代码或 catalog。

| 项目 | 论文可核验定义 | 本次用途 |
| --- | --- | --- |
| SDSS / LS | SDSS DR12 primary 位置与 Legacy Surveys 目录以 1″ 匹配 | 保留 1″ 作为静态基准。数据改为 DR17 独立 run 与 NSC DR2 测量 |
| 星系参考框架 | DESI r 星等 16–21.5、r 星等误差 <0.15、SDSS 类型 3、DESI 非 PSF、DESI Gaia G 标记为零 | 作为静止星系参考选择的设计起点。主样本和 Gaia 验证保留 Gaia 源 |
| 分区和扩张 | NSIDE=128，参考星系目标至少 800，从 0.5° 扩展至最多 2° | NSIDE 沿用为组织尺度，运动 halo 用最大位移和误差决定 |
| 校正 | 论文按区域由参考星系得到位置偏移，并研究颜色、星等和天区依赖 | 多历元残差按巡天和曝光检查。静止星系零点分组需要独立验证 |
| 误差和自行 | 两端位置差除以时间间隔，检验零点和外部误差 | 用实际测量历元、拟合协方差及独立夜/季诊断 |
| 外部检验 | QSO、恒星、星团与 Stripe 82 对照 | 审计目录结构已核实。旧代码的路径、运行参数和绘图脚本仍待恢复 |

当前图使用论文式的衬线科研图、内向刻度及明确单位。其样式为重新构建。 [论文全文](https://arxiv.org/html/2405.04016v2)

NSC 官方源码审阅固定在 `7c0756eea30b7a3afcab0e204310ee39e1c5dabc`。查看 `combine.py` 的季节数据使用、DBSCAN 后备配对、UTC MJD 及 cos(dec) 自行项，并追查 `calibrate.py` 的角距离误差。本项目独立实现 NSC-like 基准。此 commit 是实际查看的 HEAD，DR2 发布版本仍待回溯。 [源码](https://github.com/astro-datalab/nsc/tree/7c0756eea30b7a3afcab0e204310ee39e1c5dabc)

v0.3 在计算节点复查，旧 Scratch 的 `final_catalog`、`quasar_128` 和 `cluster_128` 目前为空。ScienceDB V3 提供 `desipm.fits`。计算节点读取了实际 FITS 首部，HTTP 206 声明文件为 46,955,050,560 字节，表中共有 268,314,513 行。列包含 `objID`、`RA`、`DEC`、`PM_SOURCE`、两分量自行及误差、两历元和 Legacy Surveys 星等。论文说明地基产品用 `PM_SOURCE=DESI`，Gaia DR2 行需剔除。正式比较还需核验断点获取、字段值、唯一同源匹配及真实暗端交集。 [ScienceDB V3](https://doi.org/10.57760/sciencedb.07990) · [论文 Appendix A](https://arxiv.org/html/2405.04016v2)
