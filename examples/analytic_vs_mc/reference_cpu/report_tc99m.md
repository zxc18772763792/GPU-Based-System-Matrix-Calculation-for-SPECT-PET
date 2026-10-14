<!-- ./run_example.sh tc99m cpu (scatter model v2, commit 32b650c), 4e6 histories per voxel (1.0e9 per MC run); CPU build of the analytic kernels -->
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
| crystal scatter vs MC xtalscatter | 1.175e-04 | 1.208e-04 | 0.972 ± 0.018 | 0.973 [0.920, 1.023] | 0.058 | ≤ 0.353 |
| collimator scatter vs MC colscatter | 5.083e-05 | 5.350e-05 | 0.950 ± 0.041 | 1.115 [0.550, 1.312] | 0.327 | ≤ 0.798 |
| PE + scatter vs MC total | 8.609e-03 | 8.570e-03 | 1.005 ± 0.000 | 1.007 [0.979, 1.023] | 0.013 | ≤ 0.005 |

scatter / primary: analytic 0.0199, MC 0.0208

### mc_single, wide window 60-200 keV

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.554e-03 | 0.987 ± 0.000 | 0.990 [0.962, 1.005] | 0.011 | ≤ 0.006 |
| crystal scatter vs MC xtalscatter | 5.129e-04 | 5.251e-04 | 0.977 ± 0.005 | 0.980 [0.940, 1.011] | 0.047 | ≤ 0.105 |
| collimator scatter vs MC colscatter | 6.010e-05 | 6.308e-05 | 0.953 ± 0.038 | 1.129 [0.559, 1.354] | 0.335 | ≤ 0.875 |
| PE + scatter vs MC total | 9.013e-03 | 9.142e-03 | 0.986 ± 0.000 | 0.989 [0.960, 1.004] | 0.014 | ≤ 0.006 |

scatter / primary: analytic 0.0679, MC 0.0688

### mc_model, photopeak window (Param_Physics)

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.999e-03 | 0.938 ± 0.000 | 0.942 [0.916, 0.952] | 0.009 | ≤ 0.005 |
| crystal scatter vs MC xtalscatter | 1.175e-04 | 1.308e-04 | 0.898 ± 0.016 | 0.899 [0.854, 0.945] | 0.056 | ≤ 0.340 |
| collimator scatter vs MC colscatter | 5.083e-05 | 5.796e-05 | 0.877 ± 0.036 | 1.017 [0.513, 1.191] | 0.319 | ≤ 0.767 |
| PE + scatter vs MC total | 8.609e-03 | 9.187e-03 | 0.937 ± 0.000 | 0.941 [0.915, 0.951] | 0.012 | ≤ 0.005 |

scatter / primary: analytic 0.0199, MC 0.0210

### mc_model, wide window 60-200 keV

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 9.174e-03 | 0.920 ± 0.000 | 0.924 [0.899, 0.933] | 0.009 | ≤ 0.006 |
| crystal scatter vs MC xtalscatter | 5.129e-04 | 5.932e-04 | 0.865 ± 0.004 | 0.867 [0.834, 0.894] | 0.046 | ≤ 0.099 |
| collimator scatter vs MC colscatter | 6.010e-05 | 7.180e-05 | 0.837 ± 0.030 | 0.980 [0.497, 1.166] | 0.321 | ≤ 0.816 |
| PE + scatter vs MC total | 9.013e-03 | 9.839e-03 | 0.916 ± 0.000 | 0.920 [0.894, 0.930] | 0.013 | ≤ 0.006 |

scatter / primary: analytic 0.0679, MC 0.0725

### mc_full, photopeak window (Param_Physics)

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x NaI (rho=3.67 g/cm3); hole targeting: layer 0, mu*T = 21.6 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 7.564e-03 | 1.116 ± 0.000 | 1.120 [1.068, 1.184] | 0.022 | ≤ 0.007 |
| crystal scatter vs MC xtalscatter | 1.175e-04 | 1.815e-04 | 0.647 ± 0.009 | 0.646 [0.618, 0.678] | 0.078 | ≤ 0.285 |
| collimator scatter vs MC colscatter | 5.083e-05 | 1.652e-04 | 0.308 ± 0.005 | 0.323 [0.235, 0.361] | 0.278 | ≤ 0.313 |
| PE + scatter vs MC total | 8.609e-03 | 7.911e-03 | 1.088 ± 0.000 | 1.088 [1.048, 1.153] | 0.028 | ≤ 0.007 |

scatter / primary: analytic 0.0199, MC 0.0458

### mc_full, wide window 60-200 keV

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x NaI (rho=3.67 g/cm3); hole targeting: layer 0, mu*T = 21.6 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 9.006e-03 | 0.937 ± 0.000 | 0.941 [0.909, 0.956] | 0.010 | ≤ 0.010 |
| crystal scatter vs MC xtalscatter | 5.129e-04 | 6.722e-04 | 0.763 ± 0.004 | 0.765 [0.738, 0.786] | 0.050 | ≤ 0.136 |
| collimator scatter vs MC colscatter | 6.010e-05 | 4.556e-04 | 0.132 ± 0.001 | 0.133 [0.113, 0.152] | 0.211 | ≤ 0.200 |
| PE + scatter vs MC total | 9.013e-03 | 1.013e-02 | 0.889 ± 0.000 | 0.887 [0.870, 0.912] | 0.044 | ≤ 0.009 |

scatter / primary: analytic 0.0679, MC 0.1252

