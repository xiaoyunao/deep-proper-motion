# 数据与访问

| 数据 | 实际读取状态 | 科学用途 |
| --- | --- | --- |
| NSC DR2 `meas/object/chip/exposure` | Perlmutter 上下载 TAP_SCHEMA，实测 Data Lab SQL 四张表。TAP 数据端点多次超时，官方 Data Lab SQL 同步端点成功 | `meas` 为单次测量，`object.id` 为检索键；CCD/曝光仅用于覆盖和观测条件 |
| SDSS DR17 | Perlmutter 上通过 SkyServer `PhotoObjAll` 取真数据。已读一组逐 field 的 photoObj 和 photoField 及 DR17 镜像目录 | 独立扫描和早历元。主实验用 SQL 查询，FITS 用于单位和字段核对 |
| Stripe 82 专用数据 | 官方 DR7 Stripe82/RUNS 档案说明可访问，CasJobs 登录页响应。未取回专用表数据 | 本轮使用 DR17 在 Stripe 82 内的重复扫描。不声称完整的超新星扫描时间序列 |
| Gaia DR3 | Data Lab 独立下载，已读 NERSC Gaia DR3 分块 FITS schema，实验输入采用查询版 | 评估端。挑战样本和群众性代表样本须独立 |
| Legacy Surveys DR11 | NERSC 官方共享镜像存在，计算节点已读 survey-bricks 元数据 | 尚未做逐曝光 NSC→LS 的匹配；未将 Tractor 位置用于关联或 Gaia 验证 |
| 旧自行目录 | 旧项目路径存在，未同步旧结果文件 | 本轮未作逐源历史对照 |

官方入口与格式 [NSC](https://datalab.noirlab.edu/data/nsc)、[测量 schema](https://datalab.noirlab.edu/data-explorer?showTable=nsc_dr2.meas)、[SDSS DR17 成像](https://www.sdss4.org/dr17/imaging/imaging_access/)、[SDSS photoObj](https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/CAMCOL/photoObj.html)、[SDSS photoField](https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/photoField.html)、[Stripe 82 专用归档](https://classic.sdss.org/legacy/stripe82.php)、[Legacy Surveys DR11](https://www.legacysurvey.org/dr11/files/)。

每个科学 SQL 分块执行独立计数与等长下载，超上限禁止继续，NSC 对象批次可递归分块。缓存键由服务及查询计算，保存行数和 SHA-256，HTTP 错误重试后可从成功分块继续。TAP_SCHEMA 用于单位读取，科学 NSC 查询使用 Data Lab 提供的 PostgreSQL q3c 接口。包含 TOP/LIMIT 的连接测试与科学样本分开。曝光路线先筛 CCD 几何，再按曝光与 CCD 查询。八个曝光的实际路线比较见报告。

服务器路径、环境、账户参数通过提交环境设置。原始数据、逐检测、逐源、认证及原始日志保存在远端。公开归档仅保存代码、脱敏文档、汇总统计图和数据检索方法。Scratch 可被清理。关键可复现摘要和配置保存在本地与 GitHub；历史冻结的小型Home归档保留不动，后续不新增Home数据备份；原始查询响应按需要重新取得；若必须保留难以再获取的文件，使用项目批准的持久区或 HPSS。 [NERSC Scratch](https://docs.nersc.gov/filesystems/perlmutter-scratch/)

| 读取核验 | SHA-256 |
| --- | --- |
| DR17 photoObj-003325-4-0308.fits | `fef78cd8405d0a3af6c607f9d017ad56c2218021c7c594c18205a6dca1e07925` |
| DR17 photoField-003325-4.fits | `bc65febb0ca0e2139c7e298f65a3e8f614a5c5c6b76675992039f3cc51b5c37e` |
| Data Lab 四张 NSC 表的 TAP_SCHEMA 结果 | `d7f369cc9ce86ffe151861d672e289474cace2a63676b53ea4ad0d1d38b2cde9` |

### 实验 A v0.3 发表产品与曝光元数据

NSC 曝光元数据按实际测量中的 exposure ID 查询并核验。SDSS 的独立零运动对照从 `SpecObj` 按光谱分类和红移质量获取。出版 FITS 的首部用 HTTP 206 严格检查返回字节段及长度。原始光谱候选、单曝光行、范围缓存、DESI 切片和逐源结果均只留在 NERSC。

`deep_pm.legacy_subset` 支持在计算节点按 HEALPix RING 试取小区或使用经首部检查的完整 FITS 副本。范围路径验证局部排序、区间两端与返回记录，每个下载块独立缓存；任一所需像素未成功即不写出切片。此路径依赖发布表实际按 RING 排序，目前仅有抽样证据。若已有完整副本，在 Slurm 中运行 `python -m deep_pm.legacy_subset "$DEEP_PM_WORK" --local-fits "$XIAO2025_FITS" --region test_equator`，工作目录需有 `legacy_header_summary.json`。 [ScienceDB V3](https://doi.org/10.57760/sciencedb.07990)

本轮按用户明确的主workspace使用PScratch，原始／标准化／逐源数据不搬入Home。GitHub与本地只留代码、文档和汇总图，旧冻结小归档保持原样。较大结果的长期归档另行核实HPSS初始化及配额，不作为科研运行前置，也不以触碰文件规避Scratch清理；HPSS传输应使用官方xfer队列，htar不在CPU计算节点运行。[HPSS归档](https://docs.nersc.gov/filesystems/archive/) · [htar限制与校验](https://docs.nersc.gov/filesystems/htar/) · [xfer入口](https://docs.nersc.gov/jobs/examples/#xfer-qos)
