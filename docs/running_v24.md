# v2.4 运行与复现

科研计算和缓存读取使用 Slurm CPU 分配。设置私有变量 `DPM_CHECKOUT`、`DPM_PYTHON`、`DPM_RUNTIME`、`DPM_ACCOUNT`、`DPM_SAS`、`DPM_SPECTRA` 和 `DPM_XIAO`。运行代码使用固定 Git checkout。已经成功的私有配置及资源账户以实际回执为准。项目目录 `base=$PSCRATCH/deep-proper-motion`，本轮 `work=$base/v2.4`。公开输出限于明确选定的汇总、科研图与报告。

## 几何和数据入口

下列模块名和参数交给 [Slurm worker](../scripts/v24-worker.sh)，每行代表一次作业。运行时 `--workers` 不超过分配 CPU 数。HLC、曝光、CCD 表可以独立取得。只有原生 CCD 获取完成后才生成其几何分片。

```bash
footprint_assets_v24 hlc "$work/assets/hlc" --workers 4
footprint_assets_v24 exposures "$work/assets/exposures" --workers 4
footprint_chips_v24 "$work/chips" --workers 4
stripe82_parent_v24 "$work/assets/hlc" "$DPM_CHECKOUT/docs/stripe82_protocol_v24.md" "$work/s82" --workers 4
footprint_raster_v24 early "$base" "$work/geometry" --bins 20 --workers 1
footprint_raster_v24 late  "$base" "$work/geometry" --bins 20 --workers 8
footprint_raster_v24 early "$base" "$work/geometry" --bins 40 --workers 1
footprint_raster_v24 late  "$base" "$work/geometry" --bins 40 --workers 6
footprint_manifest_v24 "$base" "$work/registration"
footprint_capacity_v24 "$base"
footprint_sky_v24 "$base"
```

SDSS 输入为 DR17 `window_flist`，NSC 几何来自实际 CCD 顶点和 MJD。1° 所有权矩形去除已登记旧矩形，细网格中心未占据但外包络可能相交的边界留给取数核查。两个分辨率给出球面面积估计。几何/时基机会、本地参考提取、联合图支持及最终科研产出使用各自分母。

## 全波段取数

```bash
footprint_intake_v24 "$base" "$DPM_SPECTRA" "$DPM_SAS" --workers 4
footprint_batch_v24 "$base" "$DPM_SPECTRA" "$DPM_SAS" START END --workers WORKERS
footprint_progress_v24 "$base" "$work/registration/v24_intake_progress_summary.json"
```

四个先导由几何分层和固定哈希选定。其余 15,231 个矩形按预定序号分配，半开 `[START,END)`。先扫 Stripe 82，再按经纬度排序。共享核心锁避免并发重复处理，每核的输入与参考分别签名。空间取数在查询侧归一原生 NSC 经度；曾因负经度触发的回执保留在历史工作目录。

[调度入口](../scripts/dispatch-v24-intake.py) 在登录节点查询 Slurm 和配额，提交与监控作业。它使用实际父作业计费及运行中作业剩余可计费上界估算预算，计算步骤不重复算费。重启调度沿用 `private_allocations.txt` 与 `private_intake_dispatch.json`。 同时接续多个旧取数版本时，由一个调度器按登记顺序填空位。用 `--early-launcher`、`--early-end` 和 `--early-mem` 指定早期批次的原执行版本、结束序号和内存。原批回执只剩少量核心时，按可处理数收缩并行与内存，尾段申请的内存至少覆盖已测单进程最大 RSS 的 1.2 倍。检查 `sacct` 的父作业状态、`ElapsedRaw`、`AllocTRES` 和步骤 MaxRSS；同时保留失败、取消和队列等待记录。先导已观测到零本地参考和高密度长尾，窗口之间动态取任务，同核 BLAS 使用一个线程。

## 共同参考与外部验证

`footprint_meridian_v24` 枚举归档负经度与 ≥360° 的条目。`footprint_longitude_audit_v24` 使用其签名与历史回执核对实际对象、所用 CCD 和跨零旧查询。`footprint_repair_v24` 在独立目录重放确认差异的输入，`resolved` 记录采用版本。

`footprint_reference_v24` 将核验后的旧/新区参考分片绑定到 `$work/joint`。图的前置条件是清单闭合，实际天区必须由完整物理分量共同求解；保留 85° 检查。目标自行与光谱质量验收都以最终冻结目录为准。

```bash
stripe82_external_v24 xiao "$work/s82-xiao-epoch" --xiao "$DPM_XIAO" --workers 1
stripe82_external_v24 nsc "$work/s82-nsc" --workers 4
stripe82_opportunity_v24 "$base" --workers 4
stripe82_comparison_v24 "$base" "$CATALOGUE" "$PLAN" "$OUTPUT" "$VERSION"
```

`CATALOGUE` 指向已闭合的 `catalogue.duckdb`；`PLAN` 必须属于同一版本。对照先在 J2015.5 做全部定位竞争者的双向唯一检查，随后取 HLC 恒星母样本。沿用同一 HLC r。恒星差值保留共享 SDSS 信息，按真实物理 run 重权只作敏感性描述。旧目录比较标记 `frozen-v22-predecessor`，扩域结果另用独立版本目录。

Stripe 82 的分母同时给出冻结归属窗口、实际扫描/CCD 交集和满足联合时基的恒星数。逐星依据每条原生边界及观测时刻传播 HLC 位置；共同源的各目录共用同一分母。

全域源拟合按固定登记序号分配。冻结核心名的 SHA256（前 64 位模 64 为零）及四个几何先导保留同源 physical-r 明细，其他核心只保留选择计数。每一源读取全部实际目标物理组，登记批号只控制计算时机。以下步骤要求最终图的数值回执。

```bash
footprint_reference_admission_v24 "$work/joint"
full_coordinate_batch "$work/joint" --core-range START END --workers WORKERS --memory-gib MEMORY --hours WALL
```

共享物理组唯一性、跨巡天连接、投影范围、NSC 中位基准、每组 48 次同步扰动和两轴协方差均从真实图文件核验。已在冻结旧目录的三个分量上执行同一数值读取，保留独立前代回执。

取数同时可以执行 `footprint_reference_packs_v24 "$base" --workers WORKERS`。一个包装入已闭合的 32 个核心或 4 个先导，保留重复检测供全域统一判定。前代分桶复用绑定输入和 DuckDB 版本，受影响的原核心重新装入。正式参考求解前核对每个包的签名。此步使用与正式程序一致的完整版本检出和 Slurm 环境。

全域目录归属核验后运行以下评价。谱样本的输入是冻结的 DR16Q 与 DESI 光谱并集。程序按物理源交集计算差值，保留每个星等档分母。地图固定 RING NSIDE=32，三目录严格共同 QSO 至少 30 颗显示质量值，低支持区只表示覆盖。改善百分比为 100 × [1 − P68(本目录) / P68(对照)]，Xiao 与 NSC 各有独立色标，同一参考的色标跨星等保持一致。

```bash
footprint_qso_v24 "$work/joint" "$DPM_SPECTRA/spectra_union.csv" "$DPM_XIAO" "$work/joint/qso" --workers WORKERS
footprint_qso_readout_v24 "$work/joint/qso" "$work/joint/qso/v24_spectral_quality_summary.json"
footprint_qso_figures_v24 "$work/joint/qso/v24_spectral_quality_summary.json" "$work/joint/figures"
full_coordinate_ngc "$work/joint" --history-root "$base"
```

亮端沿用已保存的五参数 Gaia 表观位置预测。亮端控制在最终盲目录中按实际检测集合回接，核对参考图、图心、稳健权重和自行。归属后缺失、星等退出和估计器变化均单列。它的空间适用范围限于原有控制天区。

`footprint_public_errors_v24 "$work/joint" "$work/joint/qso" "$work/joint/qso" --workers WORKERS` 按严格共同 QSO 的实际成员重建影响矩阵，先在每个参考分量内对 48 个同步扰动求群体和，再合并独立分量的均值协方差。以实际 SDSS run 同步重权对照三目录 P68。每个核心的私有误差回执可恢复。

目录归属闭合后，用同一实际多边形网格核算成功交付面积。先在两个分辨率分别运行原生参考支持积分，再运行 `footprint_output_area_v24 "$base" "$OUTPUT" --bins 40` 及 `--bins 20` 对照。交付面积使用同一核心两巡天均有签名检测的几何交集。产出面积使用至少有一个最终归属源的核心内、共同参考及实际时基同时合格的面积。源选择完备性仍由独立验证统计量描述。

跨零归档核验可提前执行。给 `footprint_batch_v24` 的完整预定序号范围加 `--archive-boundaries`，只优先访问已有归档几何审计 SHA 所列的新核心。它与正常批次复用同一核心锁、测量查询和成功回执。随后运行 `footprint_longitude_audit_v24`，只对实际有差异的输入做版本化恢复。


`scripts/dispatch-v24-graph.py "$work" "$LAUNCHER" "$DPM_SPECTRA" "$DPM_SAS" "$DPM_ACCOUNT" --watch` 在登录节点只读回执并提交 Slurm 作业。全部签名输入就绪后，调度器再次核对经度异常档案，按实际差异恢复输入，再绑定完整清单并闭合物理参考图。共享计费账本、配额、数值失败均可中断对应阶段。参考分量导出并行只读闭合后的 DuckDB，在各自目录暂存排序文件。

参考分量先由 `full_reference_graph export "$work/joint" --component COMP --threads N` 导出，然后运行 `full_reference_run "$work/joint" COMP --draw-workers 4`。正式抽样次数为 48。四进程方案在具有 2,374,858 个训练星系的归档物理分量上用 `scripts/verify-v24-reference-draws.py` 与串行逐字节核对。BLAS 保持单线程，各分量依据实测内存和时限装箱。


`footprint_held_reference_v24 "$base"` 只读取归档旧图，把预定 1/64 留出星系的实际检测校正回 ICRS。每个新分量用 `full_reference_validation "$work/joint" COMP "$OUT"` 核对原始、校准后及旧/新严格共同检测的跨巡天位置残差。旧对照要求实际星系、检测 ID、物理组均一致，并通过各自位置上的原生参考误差门槛。

完整参考图闭合后，`scripts/dispatch-v24-production.py "$work" "$LAUNCHER" "$DPM_ACCOUNT" --concurrent 4 --watch` 依据实测物理分量行数安排内存和时限。各分量一次导出和求解，小分量合并提交，真实留出星系按旧/新原生支持比较。源拟合按清单序号分批，各源仍可访问全部合格物理组。完成后建立唯一的全域索引和归属回执。跨版本参考复用要求原生精度、来源排序、星系身份、位置、中心、组集合及随机种子逐项一致。任何失败保留既有检查点，调度器允许独立分量和其他源块继续。


对于已有原生分页缓存、其实际归档测量量远超普通装箱的密集尾段，先在 Slurm 运行 `footprint_dense_v24 BASE START END --workers N`。它复用原查询页，并保留逐核心预取回执。随后为大核心单独请求内存，在原取数 checkout 及其 Python 路径下运行 `scripts/run-v24-dense-core.py BASE CATALOGUES SAS CORE`。正式回执仍由原科学代码签发，普通批次可直接复用。预取回执只证明查询页已返回。


原生 Data Lab 接口中断时可用 `DPM_NSC_TRANSPORT=counted-tap`。同一物理矩形和 q3c 索引谓词经官方 TAP 查询，先独立 COUNT，再核对结果行数。过大的分区沿同样的半开边界拆分。已有成功原生页仍按 SHA 核验复用，新页保存实际 ADQL、传输端点及原始 CSV 签名。新传输先用 `footprint_transport_check_v24` 对照独立旧先导。

### 最终目录的外部评价

在同一计费账本中，下面的入口等待最终 J2015.5 归属回执，然后把已冻结的外部对照、原生面积积分和图件提交到 Slurm。调度本身在登录节点读取状态。私有环境中的 `DPM_SPECTRA_FILE` 是合并去重的光谱 CSV，`DPM_XIAO` 是发布 FITS。每份图复用同一目录对应的聚合表。

```bash
python scripts/dispatch-v24-production.py "$work" "$LAUNCHER" "$ACCOUNT" \
  --spectra "$DPM_SPECTRA_FILE" --xiao "$DPM_XIAO" --concurrent 4 --watch
```

`joint/assessment` 保存来源分明的 QSO 与 S82 对照、源数和面积统计及 F1–F6。图、聚合表的分母和阶段回执应在允许列表同步前核对。亮 Gaia 仅检验既有独立亮端控制样本。

### 独立读取 SDSS r 位置

在 Slurm CPU 节点从实际 DR17 SAS 镜像运行以下入口。先导将四个预选核心的检测 CSV、行数和实际检查的 field 数逐项对照已签名产品。序号范围沿用全域注册清单，自动跳过已有完整输入回执的核心。

```sh
python -m deep_pm.footprint_sdss_stage_v24 "$work" "$DPM_SAS" --pilots --workers 4
python -m deep_pm.footprint_sdss_stage_v24 "$work" "$DPM_SAS" --start "$first" --end "$end" --workers "$SLURM_CPUS_PER_TASK"
```

读取暂存数据的 v2.4 取数作业设置 `DPM_SDSS_STAGE="$work/sdss-stage"`。每次复用核对矩形、缓冲区、脚本签名、SAS 镜像与 field 文件大小和修改时间，同时核对复制字节的 SHA256 和实际行数。尚未暂存的核心直接读取 DR17。逐检测文件和 field 清单只保存在 PScratch。SDSS 行数包含相邻缓冲区，最终目录的唯一源数由全域归属给出。

归档数据请求耗尽重试并返回 502、503 或 504 时，新版批次在同一路径单独检查一次极小计数。两次均为网关故障才写入共享暂停标记。常规提交器等待标记解除，进行中的核心保留回执与分页。恢复前核对服务状态并完成一个原先失败的注册核心。

在真实签名参考分包上用 `footprint_link_witness_v24 "$work"` 检查前代图是否出现连接。随后运行 `footprint_chart_geometry_v24 "$work" "$work/chart-witness/$witness_job"`。每个占有的半度格选一条实际训练方向，最远方向对与方向凸组合给出任何单图所需角半径的下界。候选图心另用全部旧训练行核实最大角距。中间连通证据不能替代最终物理图的消歧和剥离。

在原训练几何证明单一 85° TAN 不可用时，运行以下预登记的边界重放。输入是签名几何回执；验证输出写入 registration。通过后绑定其 SHA，再登记最终物理图。仅在全图实际超出 TAN 边界时使用 STG。

```sh
python -m deep_pm.footprint_chart_replay_v24 "$work" "$work/chart-witness/$witness_job/v24_linked_chart_geometry_summary.json" both
```

计数式 TAP 的测量–对象空间连接发生读超时时，改查同一筛选条件的对象 ID，按 ID 分组独立计数和读取 meas。小查询的 HTTP 网关中断仍单独处理。恢复核查使用 `footprint_index_check_v24 WORK CORE PUBLIC_SUMMARY`，对一份已有完整分页按对象身份的固定哈希选样，逐值比较实际测量的全部 17 列。


### 提前处理预计数据量较大的核心

在科学批次到达前，可向 `footprint_dense_v24 BASE START END --workers N` 传入 `--priority-plan FILE --priority-sha SHA256`。START 与 END 是该调度清单内的半开序号；入口同时校验每个核心在原足迹中的序号。当前 20 个核心的清单依据真实 CCD 计数和已处理状态冻结，工作进程仍执行相同的对象及测量分页，并读取对应的 CCD 和亮 Gaia 辅助元数据。分配规范化内存时使用实际已验证的测量行数。最终科学分母继续使用完整足迹清单。

在独立 SDSS 输入审计后，使用 `footprint_empty_early_v24 WORK SAS EARLY_COVERAGE_JSON --workers N` 回查原始扩展、实际矩形及 RUN_PRIMARY。复核官方 SAS 副本时，固定时基机会大于零且候选源表均为空的核心，取其所有候选 field 键 `(run,rerun,camcol,field)` 的并集。按 `sha256(("empty-field-v24:"+str(key)).encode()).digest()` 排序，每个 run 保留第一项，再按同一顺序选前八个。远端与镜像均使用 `sdss_sas.photoobj_path` 的相同相对路径，逐份比较 FITS 字节 SHA256 及源表行数；私有选样清单和逐项结果在 PScratch。

普通取数调度可用 `--runnable-slots` 按可运行的批次控制并发。仅显式用户暂挂的密集核心验收作业退出并发计数，其预算和对应输入范围继续保留。排队等待优先级和依赖的作业仍占并发名额。

在 `counted-tap` 下可用 `DPM_NSC_COUNTS=grouped-v24` 对至少 12 个原始测量分区的单进程全波段取数合并计数。按实际对象数和 ndet 切分赤纬条，每条分别请求测量总数和按对象分组的测量数。原数据页仍使用冻结的半开矩形与原查询列。对每个出现的对象检查实际频数，跨页检查重复与整组缺失。空归档对象允许保留。原始查询、响应 SHA 和最终核验关系保存于私有目录。小分区沿用原计数。归档计数比较见[已签名页核验](../results/summary/v24_sliced_group_count_test_summary.json)和[分页耗时](../results/summary/v24_paired_archive_timing_summary.json)。
