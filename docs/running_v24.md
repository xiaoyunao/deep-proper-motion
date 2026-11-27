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

[调度入口](../scripts/dispatch-v24-intake.py) 在登录节点查询 Slurm 和配额，提交与监控作业。它使用实际父作业计费及运行中作业剩余可计费上界估算预算，计算步骤不重复算费。重启调度沿用 `private_allocations.txt` 与 `private_intake_dispatch.json`。检查 `sacct` 的父作业状态、`ElapsedRaw`、`AllocTRES` 和步骤 MaxRSS；同时保留失败、取消和队列等待记录。先导已观测到零本地参考和高密度长尾，窗口之间动态取任务，同核 BLAS 使用一个线程。

## 共同参考与外部验证

`footprint_meridian_v24` 枚举归档负经度与 ≥360° 的条目。`footprint_longitude_audit_v24` 使用其签名与历史回执核对实际对象、所用 CCD 和跨零旧查询。`footprint_repair_v24` 在独立目录重放确认差异的输入，`resolved` 记录采用版本。

`footprint_reference_v24` 将核验后的旧/新区参考分片绑定到 `$work/joint`。图的前置条件是清单闭合，实际天区必须由完整物理分量共同求解；保留 85° 检查。目标自行与光谱质量验收都以最终冻结目录为准。

```bash
stripe82_external_v24 xiao "$work/s82-xiao-epoch" --xiao "$DPM_XIAO" --workers 1
stripe82_external_v24 nsc "$work/s82-nsc" --workers 4
stripe82_comparison_v24 "$base" "$CATALOGUE" "$PLAN" "$OUTPUT" "$VERSION"
```

`CATALOGUE` 指向已闭合的 `catalogue.duckdb`；`PLAN` 必须属于同一版本。对照先在 J2015.5 做全部定位竞争者的双向唯一检查，随后取 HLC 恒星母样本。沿用同一 HLC r。恒星差值保留共享 SDSS 信息，按真实物理 run 重权只作敏感性描述。旧目录比较标记 `frozen-v22-predecessor`，扩域结果另用独立版本目录。
