<!-- ./run_example.sh tc99m cpu (scatter model v2), 4e6 histories per voxel (1.0e9 per MC run); CPU build of the analytic kernels -->
Geometry: 256 crystals, 256 voxels, 1 rotation(s).  Sensitivities are summed over all matrix elements (counts per emitted photon x voxels).

### Photoelectric model (MC physics = primary_pe, ideal window)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary_pe | 8.440e-03 | 8.554e-03 | 0.987 ± 0.000 | 0.990 [0.962, 1.005] | 0.011 | ≤ 0.003 |

### mc_single, photopeak window (Param_Physics)

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.396e-03 | 1.005 ± 0.000 | 1.008 [0.980, 1.023] | 0.011 | ≤ 0.005 |
| crystal scatter vs MC xtalscatter | 1.183e-04 | 1.208e-04 | 0.979 ± 0.019 | 0.980 [0.901, 1.062] | 0.264 | ≤ 0.353 |
| collimator scatter vs MC colscatter | 5.060e-05 | 5.350e-05 | 0.946 ± 0.041 | 1.109 [0.548, 1.310] | 0.327 | ≤ 0.798 |
| PE + scatter vs MC total | 8.609e-03 | 8.570e-03 | 1.005 ± 0.000 | 1.007 [0.979, 1.024] | 0.015 | ≤ 0.005 |

scatter / primary: analytic 0.0200, MC 0.0208

### mc_single, wide window 60-200 keV

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.554e-03 | 0.987 ± 0.000 | 0.990 [0.962, 1.005] | 0.011 | ≤ 0.006 |
| crystal scatter vs MC xtalscatter | 4.986e-04 | 5.251e-04 | 0.950 ± 0.005 | 0.949 [0.868, 1.054] | 0.220 | ≤ 0.105 |
| collimator scatter vs MC colscatter | 5.958e-05 | 6.308e-05 | 0.945 ± 0.037 | 1.118 [0.555, 1.347] | 0.336 | ≤ 0.875 |
| PE + scatter vs MC total | 8.999e-03 | 9.142e-03 | 0.984 ± 0.000 | 0.986 [0.958, 1.007] | 0.021 | ≤ 0.006 |

scatter / primary: analytic 0.0661, MC 0.0688

### mc_model, photopeak window (Param_Physics)

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.999e-03 | 0.938 ± 0.000 | 0.942 [0.916, 0.952] | 0.009 | ≤ 0.005 |
| crystal scatter vs MC xtalscatter | 1.183e-04 | 1.308e-04 | 0.904 ± 0.016 | 0.905 [0.832, 0.981] | 0.260 | ≤ 0.340 |
| collimator scatter vs MC colscatter | 5.060e-05 | 5.796e-05 | 0.873 ± 0.036 | 1.015 [0.511, 1.191] | 0.319 | ≤ 0.767 |
| PE + scatter vs MC total | 8.609e-03 | 9.187e-03 | 0.937 ± 0.000 | 0.941 [0.915, 0.952] | 0.013 | ≤ 0.005 |

scatter / primary: analytic 0.0200, MC 0.0210

### mc_model, wide window 60-200 keV

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 9.174e-03 | 0.920 ± 0.000 | 0.924 [0.899, 0.933] | 0.009 | ≤ 0.006 |
| crystal scatter vs MC xtalscatter | 4.986e-04 | 5.932e-04 | 0.840 ± 0.004 | 0.839 [0.772, 0.928] | 0.211 | ≤ 0.099 |
| collimator scatter vs MC colscatter | 5.958e-05 | 7.180e-05 | 0.830 ± 0.030 | 0.970 [0.494, 1.161] | 0.322 | ≤ 0.816 |
| PE + scatter vs MC total | 8.999e-03 | 9.839e-03 | 0.915 ± 0.000 | 0.917 [0.891, 0.931] | 0.019 | ≤ 0.006 |

scatter / primary: analytic 0.0661, MC 0.0725

### mc_full, photopeak window (Param_Physics)

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x NaI (rho=3.67 g/cm3); hole targeting: layer 0, mu*T = 21.6 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 7.564e-03 | 1.116 ± 0.000 | 1.120 [1.068, 1.184] | 0.022 | ≤ 0.007 |
| crystal scatter vs MC xtalscatter | 1.183e-04 | 1.815e-04 | 0.652 ± 0.009 | 0.648 [0.598, 0.714] | 0.286 | ≤ 0.285 |
| collimator scatter vs MC colscatter | 5.060e-05 | 1.652e-04 | 0.306 ± 0.004 | 0.322 [0.233, 0.359] | 0.278 | ≤ 0.313 |
| PE + scatter vs MC total | 8.609e-03 | 7.911e-03 | 1.088 ± 0.000 | 1.088 [1.048, 1.152] | 0.026 | ≤ 0.007 |

scatter / primary: analytic 0.0200, MC 0.0458

### mc_full, wide window 60-200 keV

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x NaI (rho=3.67 g/cm3); hole targeting: layer 0, mu*T = 21.6 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 9.006e-03 | 0.937 ± 0.000 | 0.941 [0.909, 0.956] | 0.010 | ≤ 0.010 |
| crystal scatter vs MC xtalscatter | 4.986e-04 | 6.722e-04 | 0.742 ± 0.004 | 0.739 [0.684, 0.814] | 0.220 | ≤ 0.136 |
| collimator scatter vs MC colscatter | 5.958e-05 | 4.556e-04 | 0.131 ± 0.001 | 0.132 [0.112, 0.151] | 0.211 | ≤ 0.200 |
| PE + scatter vs MC total | 8.999e-03 | 1.013e-02 | 0.888 ± 0.000 | 0.886 [0.868, 0.912] | 0.043 | ≤ 0.009 |

scatter / primary: analytic 0.0661, MC 0.1252

