# Visualizer — SPECT/PET 元素可视化界面

浏览器中的几何与系统矩阵查看器，零依赖（纯 WebGL2 + Canvas，无需 npm、CDN 或联网），中文 / English 界面。

## 启动

**方式 1 — 直接打开**：双击 `index.html`（`file://`），用“打开文件…/打开目录…”或拖放加载文件，
或点“示例几何”。适合参数文件和不太大的系统矩阵。

**方式 2 — 本地/远程服务器**（推荐用于 GPU 服务器上的大系统矩阵）：

```bash
python3 Visualizer/serve.py --data /path/to/run_dir          # 仅需 Python 标准库
# 浏览器打开 http://127.0.0.1:8765
# 远程服务器：ssh -L 8765:127.0.0.1:8765 user@gpu-server，然后在本机浏览器打开同一地址
```

`serve.py` 默认只监听 127.0.0.1，通过 HTTP Range 和 `/api/strided` 按需读取文件，
GB 级系统矩阵不会整体载入内存；数据目录根下的参数文件、MC 输出会自动加载。

## 可加载的文件

| 文件 | 用途 |
|---|---|
| `Params_Collimator.dat`, `Params_Detector.dat`, `Params_Image.dat`, `Params_Physics.dat` | 几何元素（也接受 `Param_*.dat`） |
| `*.sysmat` | 解析或 MC 系统矩阵 `[rotation][detector][voxel]`，大小与几何不符时会报错 |
| `*_spectrum.csv`, `*_tracks.json`, `*_summary.json` | `MCGen_MultiGPU` 的能谱、光子径迹与运行摘要 |

## 功能

* **3D 视图**：准直器板（半透明）、准直孔（圆柱）、晶体（可按 μ、光电比例、能量分辨率、flag、旋转角、
  深度、编号、所选体素的投影、探测器灵敏度着色）、成像视野（随旋转位置转动）、体素数值点云、
  MC 光子径迹（按能量着色；红=光电、青=康普顿、绿=瑞利、紫=荧光）。
  剖切面 (x/y/z)、正交/透视、预设视角；单击拾取元素，双击以其为中心。
* **布局 x–z**：俯视准直孔排布与晶体足迹，可按准直层过滤，缩放/平移/点选。
* **体素图像**：探测器响应（系统矩阵的一行）或灵敏度图（Σ探测器，流式计算），支持切片、对数、色表；
  单击体素读取该列并把投影显示到晶体上。
* **能谱**：MC 沉积能谱（原发 / 准直器散射 / 晶体间散射），叠加能窗。
* **一致性检查**：层数/孔数与文件长度、孔超出板或未贯穿、孔重叠、晶体重叠（精确旋转矩形测试）、
  晶体与准直器相交、未启用晶体、μ 之和、能量分辨率缺失、视野与准直器碰撞、能窗是否包含 E0、
  系统矩阵大小等。
* **元素信息**：所选晶体/孔/准直层/体素的全部参数（含参数文件原始 y 与加上 FOV2Collimator0 后的世界坐标）。

## 坐标约定

与计算代码一致：y 轴从视野指向准直器与探测器，准直层以 x=z=0 为中心，孔沿 y 轴，晶体绕 y 轴旋转
`Param_Detector[11]`，视野绕 z 轴旋转 `index × Param_Image[7]`。参数文件中的 y 加上
`Param_Image[11]`（FOV2Collimator0）得到世界坐标。

## 文件结构

```
index.html, css/style.css
js/util.js      数学、色表、格式化
js/i18n.js      中英文字符串
js/renderer.js  WebGL2 实例化渲染、相机、剖切
js/params.js    参数文件解析、一致性检查、示例几何
js/data.js      本地文件 / HTTP Range 数据源、系统矩阵流式读取
js/plots.js     布局图、体素图像、能谱
js/app.js       界面逻辑
serve.py        数据服务器
```
