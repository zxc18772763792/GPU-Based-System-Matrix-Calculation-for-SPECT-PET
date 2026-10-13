// i18n.js - Chinese / English strings. {0}, {1} ... are replaced by arguments.
(function () {
  'use strict';
  const SV = (window.SV = window.SV || {});
  const S = {
    title: ['SPECT/PET 元素可视化', 'SPECT/PET Element Viewer'],
    openFiles: ['打开文件…', 'Open files…'],
    openDir: ['打开目录…', 'Open folder…'],
    demo: ['示例几何', 'Demo geometry'],
    secData: ['数据', 'Data'],
    serverFiles: ['服务器文件', 'Server files'],
    secDisplay: ['显示', 'Display'],
    plates: ['准直器板', 'Collimator plates'],
    holes: ['准直孔', 'Holes'],
    crystals: ['晶体', 'Crystals'],
    fov: ['成像视野', 'Field of view'],
    voxelValues: ['体素数值', 'Voxel values'],
    tracks: ['MC 光子径迹', 'MC photon tracks'],
    axes: ['坐标轴', 'Axes'],
    disabledXtals: ['未启用晶体', 'Disabled crystals'],
    plateOpacity: ['准直器透明度', 'Plate opacity'],
    colorBy: ['晶体着色', 'Colour crystals by'],
    cMuTot: ['总衰减系数 μ', 'Total attenuation μ'],
    cPeFrac: ['光电比例 μPE/μ', 'Photo fraction μPE/μ'],
    cEres: ['能量分辨率', 'Energy resolution'],
    cFlag: ['标志 (flag)', 'Flag'],
    cRot: ['旋转角', 'Rotation angle'],
    cDepth: ['深度 y', 'Depth y'],
    cIndex: ['编号', 'Index'],
    cProjection: ['所选体素的投影', 'Projection of selected voxel'],
    cDetSum: ['探测器灵敏度 (Σ体素)', 'Detector sensitivity (Σ voxels)'],
    colormap: ['色表', 'Colormap'],
    logScale: ['对数', 'Log'],
    clip: ['剖切面', 'Section plane'],
    flip: ['反向', 'Flip'],
    rotation: ['旋转位置', 'Rotation'],
    vIso: ['等轴', 'Iso'], vFront: ['正视', 'Front'], vTop: ['俯视', 'Top'], vSide: ['侧视', 'Side'], vFit: ['适配', 'Fit'],
    ortho: ['正交', 'Ortho'],
    secSysmat: ['系统矩阵', 'System matrix'],
    sysmatFile: ['文件', 'File'],
    detector: ['探测器 #', 'Detector #'],
    voxel: ['体素 #', 'Voxel #'],
    showRow: ['探测器响应', 'Detector response'],
    showCol: ['体素投影', 'Voxel projection'],
    showSens: ['灵敏度图', 'Sensitivity map'],
    secMC: ['蒙特卡洛', 'Monte Carlo'],
    specMode: ['能谱', 'Spectrum'],
    specBlur: ['能量展宽后', 'Blurred'],
    specRaw: ['沉积能量', 'Deposited'],
    secChecks: ['一致性检查', 'Consistency checks'],
    dropHint: ['拖入参数文件 (Params_*.dat)、系统矩阵 (*.sysmat) 或 MC 输出，或点击“示例几何”。',
      'Drop parameter files (Params_*.dat), system matrices (*.sysmat) or MC outputs here, or click “Demo geometry”.'],
    dropHere: ['释放以加载文件', 'Release to load files'],
    tabLayout: ['布局 x–z', 'Layout x–z'],
    tabImage: ['体素图像', 'Voxel image'],
    tabSpectrum: ['能谱', 'Spectrum'],
    slice: ['切片', 'Slice'],
    secInfo: ['元素信息', 'Element'],
    infoHint: ['在 3D 视图或布局图中点击晶体、准直孔或准直器板。', 'Click a crystal, hole or plate in the 3D view or the layout map.'],
    secHelp: ['操作', 'Controls'],
    help1: ['左键拖动：旋转；右键/Shift：平移；滚轮：缩放', 'Left drag: rotate; right drag / Shift: pan; wheel: zoom'],
    help2: ['单击：选中元素；双击：以其为中心', 'Click: select; double-click: centre on it'],
    help3: ['体素图像中单击体素：显示其在各探测器上的投影', 'Click a voxel in the image to show its projection onto the detectors'],
    hintImage: ['选择探测器后点“探测器响应”，或点“灵敏度图”', 'Pick a detector and press “Detector response”, or press “Sensitivity map”'],
    hintSpectrum: ['加载 MC 输出的 *_spectrum.csv', 'Load a *_spectrum.csv written by the MC'],
    hintEmpty: ['尚未加载几何', 'No geometry loaded'],
    countsPerPhoton: ['计数 / 发射光子 / bin', 'counts / emitted photon / bin'],
    total: ['总计', 'Total'], primary: ['原发', 'Primary'], colscatter: ['准直器散射', 'Collimator scatter'], xtalscatter: ['晶体间散射', 'Inter-crystal scatter'],
    layoutAll: ['全部', 'All'], layoutCrystals: ['仅晶体', 'Crystals only'], layoutLayer: ['准直层 {0}', 'Layer {0}'],
    // summary
    sLayers: ['准直层', 'Collimator layers'], sHoles: ['准直孔', 'Holes'], sCrystals: ['晶体 (启用)', 'Crystals (enabled)'],
    sFov: ['视野体素', 'FOV voxels'], sVoxel: ['体素尺寸', 'Voxel size'], sRot: ['旋转数 × 步长', 'Rotations × step'],
    sE0: ['光子能量', 'Photon energy'], sWindow: ['能窗', 'Energy window'], sFov2col: ['视野→准直器', 'FOV → collimator'],
    sSysmat: ['系统矩阵大小 / 旋转', 'Sysmat size / rotation'], perCrystal: ['每晶体 ±R/2', 'per crystal ±R/2'],
    // info
    iCrystal: ['晶体', 'Crystal'], iHole: ['准直孔', 'Hole'], iLayer: ['准直层', 'Collimator layer'], iVoxel: ['体素', 'Voxel'],
    iCentre: ['中心 (x, y, z)', 'Centre (x, y, z)'], iYraw: ['y (参数文件)', 'y (parameter file)'], iSize: ['尺寸 w×t×h', 'Size w×t×h'],
    iRot: ['旋转角', 'Rotation'], iMu: ['μ 总 / 光电 / 康普顿', 'μ total / PE / Compton'], iPeFrac: ['光电比例', 'Photo fraction'],
    iEres: ['能量分辨率', 'Energy resolution'], iFlag: ['标志', 'Flag'], iEnabled: ['启用', 'Enabled'], yes: ['是', 'yes'], no: ['否', 'no'],
    iLayerOf: ['所属准直层', 'Layer'], iLocal: ['层内编号', 'Index in layer'], iRadius: ['半径', 'Radius'], iYrange: ['y 范围', 'y range'],
    iPlate: ['板 W×T×H', 'Plate W×T×H'], iOffset: ['距第 1 层', 'Offset from layer 0'], iHolesN: ['孔数', 'Holes'],
    iIndex3: ['索引 (ix, iy, iz)', 'Index (ix, iy, iz)'], iLocalPos: ['位置 (未旋转)', 'Position (unrotated)'], iWorldPos: ['位置 (旋转 {0})', 'Position (rotation {0})'],
    iValue: ['当前数值', 'Current value'], iProj: ['投影值 (所选体素)', 'Projection (selected voxel)'], iDetSum: ['探测器灵敏度', 'Detector sensitivity'],
    // MC
    mE0: ['能量', 'Energy'], mHist: ['历史数 / 体素', 'Histories / voxel'], mTotal: ['总历史数', 'Total histories'],
    mSens: ['平均灵敏度 (窗 {0})', 'Mean sensitivity (window {0})'], mFrac: ['原发 / 准直散射 / 晶体散射', 'Primary / col. / xtal scatter'],
    mBackend: ['后端', 'Backend'], mTime: ['用时', 'Wall time'], mPhysics: ['物理', 'Physics'], mTracks: ['径迹', 'Tracks'], mDevices: ['设备', 'Devices'],
    // status
    stLoaded: ['已加载 {0} 个文件', 'Loaded {0} file(s)'], stDemo: ['示例几何：钨多针孔 + NaI(Tl) 像素阵列, 140.5 keV', 'Demo: tungsten multi-pinhole + pixelated NaI(Tl), 140.5 keV'],
    stRow: ['读取探测器 {0} 的响应…', 'Reading response of detector {0}…'], stCol: ['读取体素 {0} 的投影…', 'Reading projection of voxel {0}…'],
    stSens: ['计算灵敏度 (旋转 {0})…', 'Computing sensitivity (rotation {0})…'], stDone: ['完成', 'Done'],
    stNoSysmat: ['请先加载与几何匹配的 .sysmat 文件', 'Load a .sysmat file matching the geometry first'],
    stNeedGeom: ['系统矩阵需要先加载参数文件', 'Load the parameter files before a system matrix'],
    stServer: ['已连接 serve.py，数据目录：{0}', 'Connected to serve.py, data directory: {0}'],
    stError: ['错误：{0}', 'Error: {0}'],
    smOk: ['{0} 个旋转 × {1} 探测器 × {2} 体素', '{0} rotation(s) × {1} detectors × {2} voxels'],
    smBad: ['文件大小与几何不符 ({0})', 'File size does not match the geometry ({0})'],
    vRow: ['探测器 {0} 响应 (旋转 {1})', 'Response of detector {0} (rotation {1})'],
    vSens: ['灵敏度 Σ探测器 (旋转 {0})', 'Sensitivity Σ detectors (rotation {0})'],
    // checks
    chkLayers: ['准直层数 {0} 不在 0–10 范围内', 'Number of collimator layers {0} is outside 0–10'],
    chkHoleFile: ['Params_Collimator 声明 {0} 个孔，但文件只有 {1} 个数', 'Params_Collimator declares {0} holes but only has {1} values'],
    chkHole80k: ['{0} 个孔超出解析模块读取的 80000 个浮点数', '{0} holes exceed the 80000 floats read by the analytic modules'],
    chkDetFile: ['Params_Detector 声明 {0} 个探测器，但文件只有 {1} 个数', 'Params_Detector declares {0} detectors but only has {1} values'],
    chkNoImage: ['未加载 Params_Image：视野与旋转未知', 'Params_Image not loaded: FOV and rotations unknown'],
    chkNoVoxels: ['图像体素数为 0', 'The image has no voxels'],
    chkNoCrystals: ['没有探测器晶体', 'No detector crystals'],
    chkHoleOutside: ['{0} 个孔超出其准直器板边界', '{0} hole(s) extend beyond their plate'],
    chkHoleBlind: ['{0} 个孔未贯穿准直器板厚度', '{0} hole(s) do not span the full plate thickness'],
    chkHoleOverlap: ['{0} 对准直孔相互重叠', '{0} pair(s) of holes overlap'],
    chkDisabled: ['{0} 个晶体总衰减 ≤ 0.01/mm，视为未启用', '{0} crystal(s) have μ ≤ 0.01/mm and are treated as disabled'],
    chkMuSum: ['{0} 个晶体的 μPE + μCompton 大于 μ总', '{0} crystal(s) have μPE + μCompton > μtotal'],
    chkNoRes: ['{0} 个晶体能量分辨率为 0 (每晶体能窗将退化)', '{0} crystal(s) have zero energy resolution (per-crystal windows degenerate)'],
    chkXtalSize: ['晶体 #{0} 尺寸非正', 'Crystal #{0} has a non-positive size'],
    chkXtalOverlap: ['{0} 对晶体相互重叠，例如 {1}', '{0} pair(s) of crystals overlap, e.g. {1}'],
    chkXtalInPlate: ['{0} 个晶体与准直器板相交', '{0} crystal(s) intersect a collimator plate'],
    chkFovCollide: ['视野旋转半径 {0} mm ≥ 准直器前表面 {1} mm', 'FOV rotation radius {0} mm ≥ collimator front face at {1} mm'],
    chkRotSpan: ['旋转总角度 {0}°', 'Rotations span {0}°'],
    chkWindow: ['能窗 [{0}, {1}] 不包含 E0 = {2} keV', 'Energy window [{0}, {1}] does not contain E0 = {2} keV'],
    chkAllOk: ['几何检查通过', 'Geometry checks passed'],
  };
  let lang = SV.storage ? SV.storage.get('lang', null) : null;
  if (!lang) lang = (navigator.language || 'zh').toLowerCase().startsWith('zh') ? 'zh' : 'en';
  SV.lang = () => lang;
  SV.setLang = (l) => { lang = l; if (SV.storage) SV.storage.set('lang', l); };
  SV.t = function (key, ...args) {
    const e = S[key];
    let s = e ? e[lang === 'zh' ? 0 : 1] : key;
    args.forEach((a, i) => { s = s.split(`{${i}}`).join(String(a)); });
    return s;
  };
  SV.applyI18n = function (root) {
    (root || document).querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = SV.t(el.dataset.i18n); });
    document.documentElement.lang = lang === 'zh' ? 'zh' : 'en';
    document.title = SV.t('title');
  };
})();
