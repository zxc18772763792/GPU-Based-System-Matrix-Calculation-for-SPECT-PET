<!-- ./run_example.sh tc99m cpu (scatter model v2, commit 32b650c); Monte Carlo rerun with the MCGen of the code-review PR, which also writes per-component relative errors), 4e6 histories per voxel (1.0e9 per MC run); CPU build of the analytic kernels -->
Geometry: 256 crystals, 256 voxels, 1 rotation(s).  Sensitivities are summed over all matrix elements (counts per emitted photon x voxels).

### Photoelectric model (MC physics = primary_pe, ideal window)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary_pe | 8.440e-03 | 8.549e-03 | 0.987 ± 0.000 | 0.990 [0.962, 1.005] | 0.011 | ≈ 0.003 |

### mc_single, photopeak window (Param_Physics)

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.391e-03 | 1.006 ± 0.000 | 1.009 [0.980, 1.024] | 0.011 | ≈ 0.003 |
| crystal scatter vs MC xtalscatter | 1.175e-04 | 1.207e-04 | 0.973 ± 0.002 | 0.973 [0.921, 1.023] | 0.058 | ≈ 0.049 |
| collimator scatter vs MC colscatter | 5.083e-05 | 5.365e-05 | 0.947 ± 0.021 | 1.110 [0.549, 1.308] | 0.326 | ≈ 0.239 |
| PE + scatter vs MC total | 8.609e-03 | 8.565e-03 | 1.005 ± 0.000 | 1.008 [0.979, 1.024] | 0.013 | ≈ 0.005 |

scatter / primary: analytic 0.0199, MC 0.0208

### mc_single, wide window 60-200 keV

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.549e-03 | 0.987 ± 0.000 | 0.990 [0.962, 1.005] | 0.011 | ≈ 0.003 |
| crystal scatter vs MC xtalscatter | 5.129e-04 | 5.247e-04 | 0.977 ± 0.001 | 0.980 [0.940, 1.011] | 0.047 | ≈ 0.038 |
| collimator scatter vs MC colscatter | 6.010e-05 | 6.328e-05 | 0.950 ± 0.022 | 1.129 [0.559, 1.349] | 0.335 | ≈ 0.244 |
| PE + scatter vs MC total | 9.013e-03 | 9.137e-03 | 0.986 ± 0.000 | 0.989 [0.960, 1.005] | 0.014 | ≈ 0.006 |

scatter / primary: analytic 0.0679, MC 0.0688

### mc_model, photopeak window (Param_Physics)

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.993e-03 | 0.939 ± 0.000 | 0.942 [0.916, 0.952] | 0.009 | ≈ 0.003 |
| crystal scatter vs MC xtalscatter | 1.175e-04 | 1.308e-04 | 0.899 ± 0.001 | 0.900 [0.856, 0.947] | 0.056 | ≈ 0.047 |
| collimator scatter vs MC colscatter | 5.083e-05 | 5.812e-05 | 0.875 ± 0.019 | 1.015 [0.512, 1.193] | 0.318 | ≈ 0.234 |
| PE + scatter vs MC total | 8.609e-03 | 9.182e-03 | 0.938 ± 0.000 | 0.942 [0.915, 0.951] | 0.012 | ≈ 0.005 |

scatter / primary: analytic 0.0199, MC 0.0210

### mc_model, wide window 60-200 keV

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 9.169e-03 | 0.921 ± 0.000 | 0.924 [0.899, 0.934] | 0.010 | ≈ 0.003 |
| crystal scatter vs MC xtalscatter | 5.129e-04 | 5.929e-04 | 0.865 ± 0.001 | 0.868 [0.834, 0.895] | 0.046 | ≈ 0.036 |
| collimator scatter vs MC colscatter | 6.010e-05 | 7.202e-05 | 0.835 ± 0.018 | 0.980 [0.497, 1.170] | 0.321 | ≈ 0.235 |
| PE + scatter vs MC total | 9.013e-03 | 9.834e-03 | 0.917 ± 0.000 | 0.920 [0.894, 0.930] | 0.013 | ≈ 0.006 |

scatter / primary: analytic 0.0679, MC 0.0725

### mc_full, photopeak window (Param_Physics)

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x NaI (rho=3.67 g/cm3); hole targeting: layer 0, mu*T = 21.6 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 7.560e-03 | 1.117 ± 0.000 | 1.121 [1.068, 1.185] | 0.022 | ≈ 0.003 |
| crystal scatter vs MC xtalscatter | 1.175e-04 | 1.814e-04 | 0.648 ± 0.001 | 0.647 [0.618, 0.678] | 0.078 | ≈ 0.047 |
| collimator scatter vs MC colscatter | 5.083e-05 | 1.656e-04 | 0.307 ± 0.003 | 0.322 [0.234, 0.360] | 0.278 | ≈ 0.153 |
| PE + scatter vs MC total | 8.609e-03 | 7.907e-03 | 1.089 ± 0.000 | 1.089 [1.048, 1.153] | 0.028 | ≈ 0.007 |

scatter / primary: analytic 0.0199, MC 0.0459

### mc_full, wide window 60-200 keV

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x NaI (rho=3.67 g/cm3); hole targeting: layer 0, mu*T = 21.6 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 9.001e-03 | 0.938 ± 0.000 | 0.941 [0.910, 0.957] | 0.010 | ≈ 0.003 |
| crystal scatter vs MC xtalscatter | 5.129e-04 | 6.718e-04 | 0.763 ± 0.001 | 0.765 [0.738, 0.786] | 0.050 | ≈ 0.034 |
| collimator scatter vs MC colscatter | 6.010e-05 | 4.574e-04 | 0.131 ± 0.001 | 0.132 [0.112, 0.151] | 0.211 | ≈ 0.121 |
| PE + scatter vs MC total | 9.013e-03 | 1.013e-02 | 0.890 ± 0.000 | 0.887 [0.871, 0.912] | 0.044 | ≈ 0.009 |

scatter / primary: analytic 0.0679, MC 0.1255

