# 解析模型 vs 蒙特卡洛：系统矩阵分量对比示例

用同一套 `Params_*.dat`，分别用解析模块（`PEGen_RayTracing_CircularHole` + `ScatterGen_RayTracing_CircularHole`）
和 `MCGen_MultiGPU` 计算系统矩阵，并按分量逐项比较：

| 解析模块 | MC 中对应的事件类别 |
|---|---|
| PE（光电，`PE_SysMat_*_v3.sysmat`） | `primary`：未在准直器中散射、首次相互作用就在本晶体中的事件（含本晶体内康普顿后再吸收） |
| 晶体散射（`Scatter_Crystal_SysMat_*`） | `xtalscatter`：先在其它晶体中相互作用、再在本晶体中沉积 |
| 准直器散射（`Scatter_Collimator_SysMat_*`） | `colscatter`：在准直器中发生过散射 |

## 几何

| | 140.5 keV（`tc99m`） | 511 keV（`511`） |
|---|---|---|
| 准直器 | W 板 60×6×60 mm，3×3 圆柱针孔，r = 0.75 mm，间距 12 mm | W 板 60×10×60 mm，孔同左 |
| 晶体 | 16×16 NaI，2.3×6×2.3 mm，间距 2.5 mm，FWHM 10 % | 16×16 LYSO，2.3×10×2.3 mm，间距 2.5 mm，FWHM 12 % |
| 晶体中心到准直器中心 | 30 mm | 30 mm |
| 视野 | 16×1×16 体素，2 mm，位于准直器前 40 mm 的平面 | 同左 |
| 能窗（`Param_Physics[4]=1`） | 126.46–154.56 keV（另算宽窗 60–200 keV） | 459.9–562.1 keV（另算宽窗 250–650 keV） |

准直板比所有 视野→晶体 视线都大，符合解析模型“光子从入射窗进入”的约定；MC 使用
`shield_outside_collimator = absorber`（板外全吸收），两者的屏蔽假设一致。

## 运行

```bash
# 单卡 GPU（默认 GPU 0，可用 GPU=1 指定）。先编译：
#   PEGen_RayTracing_CircularHole/bd, ScatterGen_RayTracing_CircularHole/bd, (cd MCGen_MultiGPU && make)
./run_example.sh tc99m gpu
./run_example.sh 511 gpu

# 没有 GPU：原始解析 CUDA 内核编译成 CPU 版本 (tools/analytic_cpu) + MCGen_CPU（4 核约 10–20 分钟/例）
(cd ../../MCGen_MultiGPU && make cpu)
./run_example.sh tc99m cpu
```

第三个参数为每体素历史数（默认 4×10⁶）。结果在 `run_<case>/`：`report.md`（表格）、`report.json`、
`flood.svg`（各分量对所有体素求和后的探测器图像，解析 vs MC）。

脚本依次运行：

1. `tools/make_demo_params.py` 生成参数；`wide/` 子目录为同一几何、宽能窗的参数。
2. 解析：`PEGen_CircularHole`，`ScatterGen_CircularHole -PE ... -components`（光电峰窗与宽窗各一次）。
3. MC，三组设置（都用 `source_sampling = center`，即体素中心点源，与解析内核一致）：
   * `mc_pe`：`physics = primary_pe`、理想能窗 —— 与解析 PE 模型的物理完全相同，用来检查 PE 内核；
   * `mc_model`：光电 + 康普顿，系数与能量标度（PE ∝ E⁻³、康普顿 ∝ Klein–Nishina）取自参数文件，
     无瑞利、无荧光 —— 解析散射模型假设的物理，差别只来自解析模型的几何/角度近似；
   * `mc_full`：材料库截面 + 瑞利散射 + K 荧光 —— 真实物理。
4. `compare_components.py` 输出对比表。

表中各列：Σ 为全部矩阵元素之和；“ratio” 为 解析/MC；“per-voxel ratio” 为逐体素灵敏度比值的中位数与
5 %–95 % 范围；“shape TVD” = ½ Σ|A/ΣA − M/ΣM|，0 表示分布形状相同、1 表示完全不重叠；
“MC noise” 是 MC 统计涨落本身能造成的 TVD 上界。

<!-- RESULTS -->
