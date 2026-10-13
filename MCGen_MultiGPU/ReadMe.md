# MCGen_MultiGPU — 多 GPU 蒙特卡洛伽马光子系统矩阵（140.5 keV / 511 keV）

`MCGen_MultiGPU` 用蒙特卡洛 (MC) 光子输运计算 SPECT/PET 系统矩阵。它读取与解析模块
(`PEGen_RayTracing_CircularHole`、`ScatterGen_*`) **完全相同** 的四个参数文件，输出
**完全相同布局** 的 `float32 [rotation][detector][voxel]` 系统矩阵，因此可以直接替换或对比解析结果。

与解析模型相比，MC 额外包含：多次康普顿散射、瑞利散射、准直器散射、K 壳层荧光（W/Pb/I/Lu/Bi/Gd…）、
晶体间散射、能量相关的衰减系数，以及基于能量分辨率的能窗响应。这些效应在 **511 keV** 时尤其重要
（准直器穿透与散射、晶体内康普顿占主导），在 **140.5 keV** 时决定散射窗/荧光峰的形状。

## 目录

```
MCGen_MultiGPU/
  MCGen_MultiGPU.cpp   主程序：配置、输出、进度
  mc_core.h            GPU/CPU 共用的物理 + 几何 + 输运代码 (__host__ __device__)
  mc_host.h/.cpp       参数文件、材料库、加速结构 (孔/晶体网格、区域主值)
  mc_gpu.cu            多 GPU 后端 (每卡一个线程、动态任务队列)
  mc_cpu.cpp           CPU 后端 (同一份物理代码，用于无 GPU 时和验证)
  data/materials.db    光子截面库 (xraylib 生成，1–1100 keV)
  tools/gen_material_db.py   重新生成 / 添加材料
  tests/test_physics.cpp     RNG、Klein-Nishina、Rayleigh、荧光自检
  MC_Config_Tc99m.txt  140.5 keV 示例配置
  MC_Config_511keV.txt 511 keV 示例配置
  bd, Makefile         编译脚本
```

## 编译

```bash
cd MCGen_MultiGPU
./bd                       # 与仓库其它模块相同的方式 (CUDA 12.6, -arch=native)
# 或
make                       # -> MCGen_MultiGPU (GPU + CPU 后端)
make cpu                   # -> MCGen_CPU      (无需 CUDA，只有 CPU 后端)
```

* 默认 `-arch=native` 为本机 GPU 编译；混合机群可指定
  `CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89" ./bd`。
* 需要 C++17。已用 CUDA 13.4 + g++ 13 编译验证（sm_75/sm_89，内核 84–88 寄存器、无溢出）；代码只用到 CUDA 12.x 也提供的 API，默认按仓库其它模块使用 CUDA 12.6 路径。
* Windows / Visual Studio：把 `*.cpp`、`mc_gpu.cu` 加入 CUDA 工程并定义 `MC_WITH_CUDA` 即可。

## 快速开始

```bash
# 1) 生成一个示例几何 (钨多针孔 + NaI 像素阵列)，或使用你自己的 Params_*.dat
python3 ../tools/make_demo_params.py --out demo_tc99m
cd demo_tc99m

# 2) 4 张 GPU 计算 140.5 keV 系统矩阵
../MCGen_MultiGPU ../MC_Config_Tc99m.txt -gpus 0,1,2,3 -histories_per_voxel 1e6

# 511 keV (LYSO) 示例
python3 ../tools/make_demo_params.py --out demo_511 --energy 511 --crystal LYSO --thickness 10
cd ../demo_511 && ../MCGen_MultiGPU ../MC_Config_511keV.txt

# 3) 可视化：几何、系统矩阵、能谱、MC 光子径迹
python3 ../../Visualizer/serve.py --data .
```

命令行 `-key value` 覆盖配置文件；`-cuda N` 等价于 `-gpus N`；`-cpu` 使用 CPU 后端。

## 多 GPU 方案

* **任务划分**：对每个旋转位置，把体素切成块 (chunk)；每张 GPU 由一个主机线程驱动，从共享队列
  动态领取体素块，快卡自动多做。每块在显存中有自己的 `[tally][detector][voxel-in-chunk]` 累加器，
  完成后用一次 `cudaMemcpy2D` 散写到主机矩阵的对应列，再按旋转顺序追加写入文件
  （主机内存只需一个旋转的矩阵）。
* **自适应启动**：每次内核启动的历史数按 `launch_target_ms`（默认 300 ms）自动调整，
  显示卡不会卡死，进度/ETA 每 5 s 输出一次。
* **可复现**：每个光子历史使用独立的 Philox4x32-10 流，键为 `(seed)`、计数器为
  `(抽样序号, 历史号, 体素, 旋转)`。结果与 GPU 数量、块大小、线程数、启动拆分 **无关**
  （CPU 后端 1 线程与 4 线程输出逐位相同）。
* **多机扩展**：`rot_begin` / `rot_end` 只计算部分旋转，文件名自动加 `_rotB-E` 后缀，各机器结果
  按旋转顺序拼接 (`cat`) 即得完整矩阵，与单机计算完全一致。
* 显存不足时自动减小块大小（仅受 `nTally × nDet × chunk × 4 B` 限制）。

## 物理模型

| 过程 | 实现 |
|---|---|
| 截面 | `data/materials.db`：xraylib (EPDL97/Scofield 光电、Hubbell 非相干/相干)，1–1100 keV、1024 点对数网格，线性插值 |
| 光电效应 | 全吸收；可选 K 壳层荧光（Z ≥ 30，Kα1/Kα2/Kβ1,3/Kβ2，发射概率 = τK/τ · ωK · 线强），荧光光子各向同性继续输运 |
| 康普顿 | Klein–Nishina（Butcher–Messel / Geant4 采样），电子能量就地沉积 |
| 瑞利 | 独立原子形状因子 F(x)：按 F² 抽样 x²，再以 (1+cos²θ)/2 拒绝 |
| 输运 | 区域内 Woodcock (delta) 追踪 + 能量相关主值；准直孔内精确解析飞行（一步穿过孔）；均匀网格查找孔与晶体 |
| 探测器响应 | 每个晶体独立读出：一次历史中各晶体的沉积能量求和后，按高斯能量分辨率 R(E)=R0·(E0/E)^0.5（R0 = Param_Detector[10]）计算落入各能窗的概率（期望值估计，方差更小） |
| 屏蔽 | `shield_outside_collimator`：准直器板所在 y 范围内、板外区域的处理。`absorber`（默认）全吸收；`slab` 同材料的外框（默认覆盖晶体阵列 + 20 mm）；`none` 不处理。见“验证 4” |
| 计数精度 | GPU 每次启动先用 float 原子累加（每体素每次启动 ≤ 16384 个历史），随即并入 double 累加器；CPU 直接用 double |

### 140.5 keV 与 511 keV 的要点

* **140.5 keV (Tc-99m)**：钨 μ ≈ 3.6 mm⁻¹，穿透小；准直器 K 荧光（W 58–69 keV、Pb 73–87 keV）
  主要落在低能窗；NaI 中碘 K 逃逸峰在 ≈112 keV，会影响 DEW/TEW 散射窗。示例配置同时输出光电峰窗
  与下散射窗两个矩阵。
* **511 keV**：钨 μ ≈ 0.26 mm⁻¹，穿透与准直器散射显著；LYSO 中康普顿占 ~2/3。在示例几何
  （10 mm 钨针孔板 + 10 mm LYSO）中，光电峰窗内 MC 灵敏度为 1.24e-3，而解析 PE 模型为 7.7e-4，
  窗内约 30% 计数来自准直器散射 —— 这正是 MC 相对解析模型的主要价值。
* `yield`：输出默认是“每个发射光子”的探测概率；F-18 每次衰变可设 `yield = 1.934`。

## 方差缩减（无偏）

`direction_sampling = rect`（默认）使用两个“投影矩形”密度的防御性混合：

* A：整个装置（准直器 + 晶体）包围盒投影到其前表面的矩形 —— 覆盖所有可能相互作用的方向；
* B：晶体包围盒的投影矩形。

以概率 `detector_bias` 朝 B 抽样，否则朝 A，权重为 `1/(4π·p_mix)`。所有能到达装置的方向都有正概率，
因此估计 **无偏**（包括准直器任意位置的散射）。在示例几何中相对朝整个板抽样方差降低约 16 倍。
`iso` 为各向同性抽样，用于验证。

## 配置参数

| 键 | 默认 | 说明 |
|---|---|---|
| `params_dir` | `.` | 参数文件目录（`Params_*.dat`，也接受 `Param_*.dat`） |
| `preset` | `none` | `tc99m`（140.511 keV, 126.46–154.56 keV）或 `511`（459.9–562.1 keV） |
| `energy_keV` | Param_Physics[7] | 光子能量 |
| `windows` | `params` | 逗号分隔：`params`（与解析代码相同的能窗逻辑）、`auto`（每晶体 (1∓R/2)E0）或 `lo-hi` keV，最多 4 个 |
| `ewin_mode` | `gauss` | `gauss` / `ideal`（无展宽硬窗）/ `none`（不加能窗） |
| `eres_exponent`, `eres` | 0.5, — | 能量分辨率随能量的指数；全局覆盖 R0 |
| `material_db` | `data/materials.db` | 材料库路径 |
| `collimator_material` | `params` | 材料名、`params`（参数文件系数 + E⁻³ / KN 能量外推）或 `vacuum` |
| `collimator_material_layerN` | — | 第 N 层（0 起）覆盖 |
| `hole_material` | `params` | 通常 `Air` 或 `vacuum` |
| `crystal_material` | `params` | 晶体材料；`crystal_material_flagK` 按 Param_Detector[12]==K 覆盖 |
| `density_<材料>` | 库中密度 | 密度覆盖 (g/cm³) |
| `physics` | `full` | `primary_pe`：任何散射即终止、无荧光，复现解析 PE 模型（用于验证） |
| `rayleigh`, `fluorescence` | 1, 1 | 开关 |
| `energy_cutoff_keV` | 10 | 低于此能量就地沉积 |
| `shield_outside_collimator` | `absorber` | `absorber` / `slab` / `none`，见上；`slab` 外框大小由 `shield_frame_margin_mm`（20）或 `shield_frame_halfwidth_mm` / `shield_frame_halfheight_mm` 指定 |
| `histories_per_voxel` | 1e5 | 每体素历史数 |
| `source_sampling` | `uniform` | `uniform`（体素内均匀）/ `center`（体素中心，同解析代码） |
| `direction_sampling`, `detector_bias` | `rect`, 0.8 | 见“方差缩减” |
| `seed` | 20241221 | 随机数种子 |
| `backend`, `gpus` | `gpu`, `all` | `cpu` 后端；GPU 列表如 `0,1,2,3` |
| `cpu_threads`, `threads_per_voxel`, `voxel_chunk`, `launch_target_ms` | 自动 | 性能参数，不影响结果 |
| `rot_begin`, `rot_end` | 全部 | 只计算部分旋转 |
| `save_components` | 0 | 另存 原发 / 准直器散射 / 晶体间散射 三个分量矩阵 |
| `save_relerr` | 0 | 每个矩阵元素的相对统计误差 |
| `yield` | 1 | 结果乘以每次衰变的光子数 |
| `spectrum_bins`, `spectrum_emax_keV` | 1024, 1.25·E0 | 能谱直方图 |
| `trace_voxel`, `trace_histories`, `trace_rotation` | -1, 200, 0 | 记录某体素前 N 个历史的径迹（`-2` = 中心体素），供可视化 |
| `output_dir`, `output_prefix` | `.`, 自动 | 输出位置与前缀 |

未知的键会给出警告。

## 输出

前缀默认 `MC_SysMat_<E>keV_shift_<x>_<y>_<z>`：

| 文件 | 内容 |
|---|---|
| `_win<k>.sysmat` 或 `_win<k>_total.sysmat` | 能窗 k 的系统矩阵，float32 `[rotation][detector][voxel]`，单位：每发射光子的计数概率 |
| `_win<k>_primary / _colscatter / _xtalscatter.sysmat` | 分量（`save_components=1`）：进入该晶体前未发生相互作用 / 曾在准直器中相互作用 / 曾在其它晶体中相互作用 |
| `_win<k>_relerr.sysmat` | 相对统计误差（`save_relerr=1`） |
| `_spectrum.csv` | 所有晶体的沉积能谱（原始与能量展宽后，按三类事件） |
| `_tracks.json` | 径迹（可视化用） |
| `_summary.json` | 运行信息、材料、平均灵敏度及各分量比例、设备吞吐 |

## 材料库

`data/materials.db` 包含：Air, Water, PMMA, Al, Fe, Cu, W, W90Ni7Fe3, W95Ni3.5Fe1.5, Pb, Ta, Au, Pt,
NaI, CsI, LYSO, LSO, BGO, GAGG, LaBr3, CeBr3, CZT, CdTe, Ge, Si。添加材料：

```bash
pip install xraylib
python3 tools/gen_material_db.py --add "LuAG:Lu3Al5O12:6.73" --add "Steel:MIX(Fe:0.7,Cr:0.19,Ni:0.11):8.0" --out data/materials.db
```

程序会把库中 μ(E0)（光电+康普顿）与参数文件中的系数比较，相差 >15% 时给出警告，避免材料指定错误。

## 验证

1. **与精确积分对比**：单个晶体（μ = 0.243 mm⁻¹，纯吸收）+ 点源，方向空间精细数值积分得
   2.467869e-3；MC（4×10⁷ 历史）为 2.467977e-3，**MC/精确 = 1.00004 ± 0.00010**。
2. **与解析 PE 模块逐元素对比**（`physics=primary_pe, source_sampling=center, ewin_mode=none`，
   系数取自参数文件；解析结果由 `PESysMatGen.cu` 原内核计算）：

   | 几何 | 总灵敏度 MC/解析 | 逐元素 z = (MC−解析)/σ_MC：均值 / rms |
   |---|---|---|
   | 140.5 keV，6 mm W 针孔板 (160 mm)，6 mm NaI | 1.005 | 0.05 / 1.06 |
   | 140.5 keV，小准直板 (30 mm)，大晶体阵列 | 1.002 | 0.00 / 1.01 |
   | 140.5 keV，4 个旋转 + 视野平移，2 层体素 | 1.002–1.010 | 0.01–0.18 / 0.94–1.02 |
   | 511 keV，10 mm W 针孔板 (160 mm)，10 mm LYSO | 1.0005 | 0.22 / 0.97 |

   z 的 rms ≈ 1 说明逐元素在统计误差内一致。140.5 keV 时 MC 比解析高约 0.2–0.5%：把解析内核的晶体
   子单元从 8×16×8 加密到 24×16×24，差距从 0.53% 降到 0.31%，说明这是解析方法对针孔半影的离散化误差。
   可用 `tools/compare_sysmat.py` 在你自己的系统上重复。
3. **物理自检** `tests/test_physics.cpp`：Philox 已知答案向量；Klein–Nishina 角分布 χ²/dof ≈ 0.9；
   Rayleigh 与形状因子表 χ²/dof ≈ 1.0–1.1（表本身与 xraylib DCS_Rayl 相差 < ±2–4%）；
   140.5 keV 时 W/Pb 每次光电吸收的 K X 射线概率 0.78/0.77。

   ```bash
   cd tests && g++ -O2 -std=c++17 -I.. test_physics.cpp ../mc_host.cpp -o test_physics && ./test_physics ../data/materials.db
   ```
4. **准直板边缘（511 keV 重要）**：小准直板 (30 mm, 10 mm W) + 60 mm LYSO 阵列时，MC (`absorber`) 只有解析值的
   75%。按中心射线分类：穿过板前后表面的射线 MC/解析 = 0.97；**从板侧面穿出**的射线解析给出 1.8e-2、MC 为 1.3e-3。
   原因是解析内核对“碰到板”的射线按完整厚度 T/cosθ 衰减（相当于板向外延伸），而完全错过板的射线又被
   视为全挡。用 `shield_outside_collimator = slab` 时，前后表面射线 1.002、侧面射线 1.095，但错过板的射线
   变为 3.2e-3（解析为 0）。140.5 keV 下钨几乎不透明，两者差别可忽略；**511 keV 下请按实际屏蔽选择模式**，
   或让准直板覆盖所有视线。
5. **能谱**：140.5 keV / NaI / W 针孔板中可见 140.5 keV 光电峰、111.9 keV 碘逃逸峰、59.3 keV W Kα、
   28–33 keV 碘 K X 射线的晶体间串扰。

> 本仓库在无 GPU 的环境中开发：CUDA 后端已用 nvcc 编译链接，物理与几何代码与 CPU 后端为同一份并已验证，
> 但多 GPU 运行速度尚未实测，请在你的机器上先用小的 `histories_per_voxel` 试运行。

## 局限与假设

* 电子就地沉积（140–511 keV 电子在晶体中射程 ≲ 1 mm）；不含多普勒展宽与束缚效应对角分布的修正。
* 每个晶体独立读出（像素化探测器）；连续晶体的 Anger 定位未建模。
* 单光子 (singles) 模型：不含正电子射程与湮灭光子非共线性，也不做符合 (PET LOR) 计数。
* 输出文件为 float32（累加在 double 中完成）。
* 与解析代码相同的几何约定：准直器板以 x=z=0 为中心、孔沿 y 轴、晶体绕 y 轴旋转。
