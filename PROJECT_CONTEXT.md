# SDSS + NSC 多历元自行

项目研究较早的 SDSS 单历元位置、独立的 NSC DR2 单曝光位置与联合天测校准，能否改善普通恒星和缺少 Gaia 自行的暗星测量。Xiao et al. (2025) 用 SDSS 与 Legacy Surveys 的两端位置估计自行；本项目检验多历元采样和公共位置误差的独立贡献。 [Xiao et al. (2025)](https://doi.org/10.3847/1538-3881/adb72c) · [官方 NSC](https://datalab.noirlab.edu/data/nsc)

- **实验 A** 用公开目录位置，评估检测关联、自行精度和系统误差。v0.2 的定向高自行留出恢复 19/24，纯位置基准为 0/24 和 3/24；普通代表场的运动法受大型冲突影响。 [v0.2 报告](reports/experiment_A_v0.2.md) · [冻结提交](https://github.com/xiaoyunao/deep-proper-motion/commit/b92912c220252349a6050ef25cdab6e711d37ded)
- **实验 B** 比较同曝光的发布位置与自由 PSF 质心，并将新图像历元的贡献单独测量。目录级剩余误差能否稳定归因到质心条件，决定是否开展小规模图像对照。
- **v0.3** 四区 214 颗亮星中，最早一次 SDSS 降低四场的北向 NMAD。固定赤道 47 颗星与检测，M3 将北向偏差从 −4.66 降到 −1.33 mas/yr。额外的颜色和 DCR 项在独立验证中未稳定改善。赤道 64 条 (r=21)–23 轨迹的留出位置没有稳定绝对误差收益；6 个光谱 QSO 的自行对照仍太小。 [v0.3 报告](reports/experiment_A_v0.3_precision.md) · [冻结提交](https://github.com/xiaoyunao/deep-proper-motion/commit/9eacbec17f7c31041ff55be05debe277fd10dd88)
- **当前阶段** 严格同源复核亮星，并用新天区的光谱 QSO 测试冻结 M3 相对官方 NSC 的暗端增益。星系参考和 QSO 验证分离。此阶段暂停寻找已删除的 Xiao2025 成品；论文精度仅作背景。 [计划](PLAN.md) · [去标识汇总](results/summary)

- **v0.4 亮星同源复核** 183 颗有共同 M3 支持。二维绝对误差第 68 百分位相对 NSC 在北天改善 87%，拥挤区改善 32%，两区的对象配对区间排除零。赤道改善 8%，区间跨零，北向点估计仍较差。Pal 5 的 20 颗用 M1 可改善 66%。 [v0.4 报告](reports/experiment_A_v0.4_nsc.md) · [亮星图](figures/v04_bright_nsc_summary.png)
