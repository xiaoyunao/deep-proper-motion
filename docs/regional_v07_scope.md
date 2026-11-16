# v0.7 已锁定区域与覆盖

冻结的 16 个核心为 `west_a/west_b/west_c`、`east_a/east_b/east_c/east_d`、`north_a/north_b/north_c`、`single_c/single_e/deep_a`、`crowd_a/crowd_b/crowd_d`。西段原始两个范围超出 65,000 个缓冲区档案对象上限，分别以 0.62° 和 0.60° 核心复核。西段另一个同样重新缩小的 `west_d` 预留取数失败替代。为保留单扫描高纬对照，优先使用实测帧面 97.7% 为单扫描的 `single_e`。

所有初始通过的场与追加的三个场，在首次查看这些天区的自行之前锁定。入选只看 SDSS rerun 301 扫描几何、NSC r 芯片足迹与基线、档案计数和光谱位置总数。预查的 768 个均匀面积点区分几何覆盖与跨度支持。每块产出仍应用已冻结的点源形态、测量质量、时间以及校准门槛。

新核心不重叠，v0.5 核心和独立 COSMOS 档案吞吐区均单独统计。正式评估根据共享物理 SDSS run 与 NSC exposure 合并观测块，检验跨区不确定度。 [区域配置](../configs/regional-v07.toml)
