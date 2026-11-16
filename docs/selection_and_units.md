# 检测标准化与选择

- 位置是 J2000/ICRS 方向。时间由实际观测给出，统一成 TT MJD，自行拟合用 365.25 天的 Julian year。投影使用球面三维切基，输出列 `pm_x/pm_y` 属于当前天区切平面。
- SDSS `RAERR` 以角秒给出，含 cos(dec)。`DECERR` 同单位。按 1000 转换到 mas。SDSS `TAI_r` 是自 MJD 零点起的 TAI 秒。与 `offsetRa_r = offsetDec_r = 0` 同时满足时采用 r 时刻。不确定参考波段的检测仍进入检测表，但暂缺时刻。统一转换加 32.184 秒到 TT。SAS FITS 类型列是 `OBJC_TYPE`，SQL `type` 是归并类型。
- SDSS 使用 `RUN_PRIMARY`（bit 0，掩码 1）去掉同 run 邻 field 的重复。保留不同 run。同一物理 run 不合并其他版本的处理结果。本次所有观测来自同一 release，未导入 coadd。颜色采用同一次扫描的 PSF g−i，可缺。
- NSC `mjd` 按 UTC 输入。TAP_SCHEMA 标为天，NSC 组合源码从 `DATE-OBS` 明确按 UTC 计算。位置误差为角秒。校准源码及在切平面用法支持将 `raerr` 解释为东向角距离误差，不再乘余弦。本轮公开小区的赤纬接近零度或 35°，尚缺其他高赤纬与误差残差的外部核验。
- 无时间或非正、非有限误差的检测保留记录，从当前拟合排除。演示控制样本共用误差 ≤150 mas、已知原生波段星等 ≤20.5 的阈值，星等缺测不单独删除，形态不作硬切。每轴在加权拟合中加 50 mas 独立位置底噪。阈值与完整度均针对取回、入选的检测。

[SDSS Resolve](https://www.sdss4.org/dr16/algorithms/resolve) · [SDSS 质量位](https://www.sdss4.org/dr16/algorithms/bitmasks) · [NSC schema](https://datalab.noirlab.edu/data-explorer?showTable=nsc_dr2.meas)

在开发区的 DR17 run 3325、camcol 4、field 308，SAS 与 CAS 同一检测的 `TAI_r` 完全一致，位置和 RAERR 在浮点精度内一致。TAI 换算成 MJD 后比该记录的 CAS 整数 MJD 大 0.3972 天。匹配 photoField 确有 airmass、REF_TAI 及 DCR 颜色列。

SDSS CAS 官方 astrometry 说明把质心误差与 frame 标定误差平方相加，并在坐标变换后产生 RA/DEC 协方差。v0.3 保留发布的边际 `RAERR/DECERR`，沿用 50 mas 每轴实验底噪；未将 `photoField` 标定误差再次加到独立逐源项。参考星系重采样传播的是新增零点估计的不确定度，发布误差的逐源与共模分量仍待拆分。 [SDSS CAS 天测说明](https://classic.sdss.org/dr7/algorithms/astrometry.php)
