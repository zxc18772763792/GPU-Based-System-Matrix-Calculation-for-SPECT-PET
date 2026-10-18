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
  tests/reference_mc.py      独立 numpy 参考 MC，交叉验证完整输运（多次康普顿 + 光电）
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
# 在 MCGen_MultiGPU/ 目录下执行
# 1) 生成一个示例几何 (钨多针孔 + NaI 像素阵列)，或使用你自己的 Params_*.dat
python3 ../tools/make_demo_params.py --out demo_tc99m
cd demo_tc99m

# 2) 4 张 GPU 计算 140.5 keV 系统矩阵
../MCGen_MultiGPU ../MC_Config_Tc99m.txt -gpus 0,1,2,3 -histories_per_voxel 1e6

# 511 keV (LYSO) 示例
python3 ../../tools/make_demo_params.py --out ../demo_511 --energy 511 --crystal LYSO --thickness 10
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
  （CPU 后端 1 线程与 4 线程输出逐位相同；GPU 上浮点原子加法的顺序不固定，不同运行之间只在
  float 舍入量级 (~1e-7 相对) 上不同）。
* **多机扩展**：`rot_begin` / `rot_end`（左闭右开，即旋转 `rot_begin … rot_end-1`）只计算部分旋转，文件名自动加 `_rotB-E` 后缀，各机器结果
  按旋转顺序拼接 (`cat`) 即得完整矩阵，与单机计算完全一致。
* 显存不足时自动减小块大小：每个块占用 `nTally × nDet × chunk × 12 B`（float 启动累加器 + double
  块累加器；`save_relerr=1` 时再加一倍），块大小取不超过可用显存 60% 的值。

## 物理模型

| 过程 | 实现 |
|---|---|
| 截面 | `data/materials.db`：xraylib (EPDL97/Scofield 光电、Hubbell 非相干/相干)，1–1100 keV、1024 点对数网格，线性插值 |
| 光电效应 | 全吸收；可选 K 壳层荧光（Z ≥ 30，Kα1/Kα2/Kβ1,3/Kβ2，发射概率 = τK/τ · ωK · 线强），荧光光子各向同性继续输运 |
| 康普顿 | Klein–Nishina（Butcher–Messel / Geant4 采样），电子能量就地沉积 |
| 瑞利 | 独立原子形状因子 F(x)：按 F² 抽样 x²，再以 (1+cos²θ)/2 拒绝 |
| 输运 | 区域内 Woodcock (delta) 追踪 + 能量相关主值；准直孔内精确解析飞行（一步穿过孔）；均匀网格查找孔与晶体 |
| 探测器响应 | 每个晶体独立读出：一次历史中各晶体的沉积能量求和后，按高斯能量分辨率 R(E)=R0·(E0/E)^0.5（R0 = Param_Detector[10]）计算落入各能窗的概率（期望值估计，方差更小） |
| 屏蔽 | `shield_outside_collimator`：准直器板所在 y 范围内、板外区域的处理。`absorber`（默认）全吸收；`slab` 同材料的外框（默认覆盖晶体阵列 + 20 mm）；`none` 不处理。见“验证 5” |
| 计数精度 | GPU 每次启动先用 float 原子累加（每体素每次启动 ≤ 16384 个历史），随即并入 double 累加器；CPU 直接用 double |

### 140.5 keV 与 511 keV 的要点

* **140.5 keV (Tc-99m)**：钨 μ ≈ 3.6 mm⁻¹，穿透小；准直器 K 荧光（W 58–69 keV、Pb 73–87 keV）
  主要落在低能窗；NaI 中碘 K 逃逸峰在 ≈112 keV，会影响 DEW/TEW 散射窗。示例配置同时输出光电峰窗
  与下散射窗两个矩阵。
* **511 keV**：钨 μ ≈ 0.26 mm⁻¹，穿透与准直器散射显著；LYSO 中康普顿占 ~2/3。在
  [`examples/analytic_vs_mc`](../examples/analytic_vs_mc/ReadMe.md) 的 511 keV 例子（10 mm 钨板、3×3 针孔，
  10 mm LYSO）中，光电峰窗内 MC 总灵敏度是解析 PE 模型的 1.55 倍，窗内约 31% 计数来自准直器散射
  —— 这正是 MC 相对解析模型的主要价值。
* `yield`：输出默认是“每个发射光子”的探测概率；F-18 每次衰变可设 `yield = 1.934`。

## 方差缩减（无偏）

`direction_sampling = rect`（默认）的发射方向来自三个分量的防御性混合，权重为 `1/(4π·p_mix(ω))`：

| 分量 | 概率 | 抽样方式 |
|---|---|---|
| A 装置 | `1 − hole_bias − detector_bias` | 整个装置（准直器 + 晶体）包围盒投影到其前表面的矩形；覆盖所有可能相互作用的方向，保证无偏 |
| B 晶体 | `detector_bias` | 晶体阵列包围盒的投影矩形 |
| H 准直孔 | `hole_bias` | 每个体素预先列出其“看得见”的目标层准直孔（孔道接收角 × `hole_target_acceptance` 之内，最多 32 个），均匀选一个孔，瞄准其入口处边长 2r(1+`hole_target_margin`) 的正方形 |

H 的孔列表只依赖 (体素, 旋转)，所以结果仍与 GPU 数量和任务切分无关。品质因数（FOM = 1/(误差²·时间)）实测：

| 几何 | 能量 | 准直板 μT | FOM 提升（总灵敏度 / 单元素） |
|---|---|---|---|
| 平行孔，25 mm W，2822 孔 | 140.5 keV | 90 | 7× / 7× |
| 针孔，6 mm W | 140.5 keV | 22 | 5× / 10× |
| 针孔，10 mm W | 511 keV | 2.6 | 0.6× / 0.3×（穿透主导，不宜瞄准） |
| 针孔，1 mm W | 140.5 keV | 3.6 | 0.4× / 0.5× |

因此默认 `hole_bias = auto`：目标层对 E0 不透明（μT ≥ 5）时取 0.5（此时 `detector_bias` = 0.3），否则为 0（`detector_bias` = 0.8）。
程序启动时会打印判定结果。无偏性已用不同种子、两种抽样方式的 MC 互相比较（4 个旋转位置差异 < 0.6σ）以及与解析结果的比较确认。
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
| `physics` | `full` | `primary_pe`：任何散射即终止、无荧光，复现解析 PE 模型（用于验证）；`single_scatter`：最多一次康普顿散射（第二次散射或瑞利散射即终止）、无荧光、只记光电吸收的能量，且不计在散射晶体本身中的再吸收 —— 与解析散射模型 v2 的一阶散射定义完全一致（用于验证 ScatterGen） |
| `rayleigh`, `fluorescence` | 1, 1 | 开关（`1/0`、`on/off`、`true/false`、`yes/no`；其它布尔键同） |
| `energy_cutoff_keV` | 10 | 低于此能量就地沉积 |
| `shield_outside_collimator` | `absorber` | `absorber` / `slab` / `none`，见上；`slab` 外框大小由 `shield_frame_margin_mm`（20）或 `shield_frame_halfwidth_mm` / `shield_frame_halfheight_mm` 指定 |
| `histories_per_voxel` | 1e5 | 每体素历史数 |
| `source_sampling` | `uniform` | `uniform`（体素内均匀）/ `center`（体素中心，同解析代码） |
| `direction_sampling` | `rect` | `rect`（重要性抽样，无偏）/ `iso` |
| `hole_bias`, `detector_bias` | `auto`, `auto` | 混合权重，见“方差缩减” |
| `hole_target_layer`, `hole_target_acceptance`, `hole_target_margin` | 最前层, 1.5, 0.25 | 孔定向的目标层、接收角放宽倍数、入口正方形余量 |
| `seed` | 20241221 | 随机数种子 |
| `backend`, `gpus` | `gpu`, `all` | `cpu` 后端；GPU 列表如 `0,1,2,3` |
| `cpu_threads`, `threads_per_voxel`, `voxel_chunk`, `launch_target_ms` | 自动 | 性能参数，不影响结果 |
| `rot_begin`, `rot_end` | 0, 旋转数 | 只计算旋转 `rot_begin … rot_end-1`（左闭右开） |
| `save_components` | 0 | 另存 原发 / 准直器散射 / 晶体间散射 三个分量矩阵 |
| `save_relerr` | 0 | 每个矩阵元素的相对统计误差 |
| `yield` | 1 | 结果乘以每次衰变的光子数 |
| `spectrum_bins`, `spectrum_emax_keV` | 1024, 1.25·E0 | 能谱直方图 |
| `trace_voxel`, `trace_histories`, `trace_rotation` | -1, 200, `rot_begin` | 记录某体素前 N 个历史的径迹（`-2` = 中心体素），供可视化 |
| `trace_max_vertices` | 200000 | 径迹顶点数上限 |
| `output_dir`, `output_prefix` | `.`, 自动 | 输出位置（不存在时自动创建）与前缀 |

未知的键会给出警告；数值键必须是完整的数字（如 `1e6`，不接受 `1,000`），否则报错退出。

## 输出

前缀默认 `MC_SysMat_<E>keV_shift_<x>_<y>_<z>`：

| 文件 | 内容 |
|---|---|
| `_win<k>.sysmat` 或 `_win<k>_total.sysmat` | 能窗 k 的系统矩阵，float32 `[rotation][detector][voxel]`，单位：每发射光子的计数概率 |
| `_win<k>_primary / _colscatter / _xtalscatter.sysmat` | 分量（`save_components=1`）：进入该晶体前未发生相互作用 / 曾在准直器中相互作用 / 曾在其它晶体中相互作用 |
| `_win<k>_relerr.sysmat` | 相对统计误差（`save_relerr=1`）；同时 `save_components=1` 时另有各分量自己的 `_win<k>_<分量>_relerr.sysmat` |
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
   | 140.5 keV，4 个旋转 + 视野平移，2 层体素 | 1.006–1.011 | 0.06–0.16 / 1.03–1.07 |
   | 511 keV，10 mm W 针孔板 (160 mm)，10 mm LYSO | 1.0005 | 0.22 / 0.97 |

   z 的 rms ≈ 1 说明逐元素在统计误差内一致。140.5 keV 时 MC 比解析高约 0.3–1%：把解析内核的晶体
   子单元从 8×16×8 加密到 24×16×24，差距从 0.53% 降到 0.31%，说明这是解析方法对针孔半影的离散化误差
   （两种 MC 抽样方式彼此一致，见“方差缩减”）。
   可用 `tools/compare_sysmat.py` 在你自己的系统上重复。
3. **物理自检** `tests/test_physics.cpp`：Philox 已知答案向量；Klein–Nishina 角分布 χ²/dof ≈ 0.9；
   Rayleigh 与形状因子表 χ²/dof ≈ 1.0–1.1（表本身与 xraylib DCS_Rayl 相差 < ±2–4%）；
   140.5 keV 时 W/Pb 每次光电吸收的 K X 射线概率 0.78/0.77。

   ```bash
   cd tests && g++ -O2 -std=c++17 -I.. test_physics.cpp ../mc_host.cpp -o test_physics && ./test_physics ../data/materials.db
   ```
4. **完整物理过程的独立交叉验证** `tests/reference_mc.py`：另写的 numpy MC（自己的几何与锥形源抽样、
   Kahn 方法抽样 Klein–Nishina，只共享截面表）模拟单晶体中的多次康普顿散射 + 光电吸收（关闭瑞利与荧光），
   比较沉积能量在 光电峰 / 单次康普顿区 / 康普顿边与峰之间 / 任意沉积 四个能窗的计数（各 2×10⁷ 光子）：

   | 晶体 | 能量 | 光电峰 | 单次康普顿区 | 边–峰之间 | 任意沉积 |
   |---|---|---|---|---|---|
   | NaI 20×10×20 mm | 140.5 keV | 0.9997 | 0.9999 | 0.984 (z −0.7) | 0.9997 |
   | LYSO 20×15×20 mm | 511 keV | 1.0001 | 0.998 | 0.998 | 0.9996 |

   表中为 MCGen/参考 的比值，所有 |z| ≤ 2.1。

   ```bash
   make cpu && python3 tests/reference_mc.py --energy 140.5 --material NaI
   python3 tests/reference_mc.py --energy 511 --material LYSO --thickness 15
   ```
5. **准直板边缘 / 屏蔽约定**：解析模型约定光子**从入射窗（板的前表面）入射**，按完整厚度 T/cosθ 衰减，
   相当于板在侧向无限延伸、板外完全屏蔽；准直板应覆盖所有 视野→晶体 视线，解析结果才有意义。
   MC 则需要给出屏蔽结构的具体细节：`absorber` 与解析约定一致（板外全吸收），`slab` 为同材料外框，
   `none` 为无屏蔽，真实系统请按实际的屏蔽体选择/设置。参考数据（小准直板 30 mm、10 mm W + 60 mm LYSO，
   511 keV，板没有覆盖所有视线）：MC (`absorber`) 为解析值的 75%，差别全部来自从板侧面穿出的射线
   （解析 1.8e-2、MC 1.3e-3，穿过前后表面的射线 MC/解析 = 0.97）；`slab` 时前后表面 1.002、侧面 1.095，
   错过板的射线为 3.2e-3（解析为 0）。140.5 keV 下钨几乎不透明，差别可忽略。
6. **能谱**：140.5 keV / NaI / W 针孔板中可见 140.5 keV 光电峰、111.9 keV 碘逃逸峰、59.3 keV W Kα、
   28–33 keV 碘 K X 射线的晶体间串扰。
7. **解析 PE + 散射 与 MC 的分量对比示例**：见 [`examples/analytic_vs_mc`](../examples/analytic_vs_mc/ReadMe.md)
   （140.5 keV 与 511 keV，单卡 GPU 或 CPU 运行）。`physics = single_scatter` 与 ScatterGen 散射模型 v2 的物理定义
   完全相同，两者的晶体散射在 ≤ 3 %、准直器散射在 ≤ 5 % 内一致（511 keV 均 ≤ 1 %）；`full` 物理下的散射/主成分比约为一阶模型的 2 倍
   （多次散射、瑞利散射、K 荧光、同一晶体内的多次相互作用）。

> 本仓库在无 GPU 的环境中开发：CUDA 后端已用 nvcc 编译链接，物理与几何代码与 CPU 后端为同一份并已验证，
> 但多 GPU 运行速度尚未实测，请在你的机器上先用小的 `histories_per_voxel` 试运行。

## 局限与假设

* 电子就地沉积（140–511 keV 电子在晶体中射程 ≲ 1 mm）；不含多普勒展宽与束缚效应对角分布的修正。
* 每个晶体独立读出（像素化探测器）；连续晶体的 Anger 定位未建模。
* 单光子 (singles) 模型：不含正电子射程与湮灭光子非共线性，也不做符合 (PET LOR) 计数。
* 输出文件为 float32（累加在 double 中完成）。
* 与解析代码相同的几何约定：准直器板以 x=z=0 为中心、孔沿 y 轴、晶体绕 y 轴旋转。
