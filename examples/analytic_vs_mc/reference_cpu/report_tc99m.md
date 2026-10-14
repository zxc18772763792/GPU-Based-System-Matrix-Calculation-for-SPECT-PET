<!-- ./run_example.sh tc99m cpu, 4e6 histories per voxel (1.0e9 per MC run); CPU build of the original analytic kernels -->
Geometry: 256 crystals, 256 voxels, 1 rotation(s).  Sensitivities are summed over all matrix elements (counts per emitted photon x voxels).

### Photoelectric model (MC physics = primary_pe, ideal window)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary_pe | 8.440e-03 | 8.554e-03 | 0.987 ± 0.000 | 0.990 [0.962, 1.005] | 0.011 | ≤ 0.003 |

### mc_model, photopeak window (Param_Physics)

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 8.999e-03 | 0.938 ± 0.000 | 0.942 [0.916, 0.952] | 0.009 | ≤ 0.005 |
| crystal scatter vs MC xtalscatter | 1.966e-07 | 1.308e-04 | 0.002 ± 0.000 | 0.000 [0.000, 0.005] | 0.996 | ≤ 0.340 |
| collimator scatter vs MC colscatter | 1.160e-03 | 5.796e-05 | 20.019 ± 0.820 | 21.709 [12.175, 38.586] | 0.528 | ≤ 0.767 |
| PE + scatter vs MC total | 9.601e-03 | 9.187e-03 | 1.045 ± 0.000 | 1.046 [0.998, 1.086] | 0.108 | ≤ 0.005 |

scatter / primary: analytic 0.1375, MC 0.0210

### mc_model, wide window 60-200 keV

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=3.406,pe=3.216,compton=0.1899/mm @E0), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x params(tot=0.243,pe=0.2032,compton=0.03983/mm @E0); hole targeting: layer 0, mu*T = 20.4 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 9.174e-03 | 0.920 ± 0.000 | 0.924 [0.899, 0.933] | 0.009 | ≤ 0.006 |
| crystal scatter vs MC xtalscatter | 3.209e-03 | 5.932e-04 | 5.410 ± 0.024 | 5.406 [4.964, 5.979] | 0.241 | ≤ 0.099 |
| collimator scatter vs MC colscatter | 1.484e-03 | 7.180e-05 | 20.672 ± 0.750 | 21.988 [12.318, 42.067] | 0.497 | ≤ 0.816 |
| PE + scatter vs MC total | 1.313e-02 | 9.839e-03 | 1.335 ± 0.000 | 1.339 [1.278, 1.373] | 0.266 | ≤ 0.006 |

scatter / primary: analytic 0.5561, MC 0.0725

### mc_full, photopeak window (Param_Physics)

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x NaI (rho=3.67 g/cm3); hole targeting: layer 0, mu*T = 21.6 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 7.564e-03 | 1.116 ± 0.000 | 1.120 [1.068, 1.184] | 0.022 | ≤ 0.007 |
| crystal scatter vs MC xtalscatter | 1.966e-07 | 1.815e-04 | 0.001 ± 0.000 | 0.000 [0.000, 0.003] | 0.996 | ≤ 0.285 |
| collimator scatter vs MC colscatter | 1.160e-03 | 1.652e-04 | 7.023 ± 0.103 | 7.277 [4.894, 11.251] | 0.494 | ≤ 0.313 |
| PE + scatter vs MC total | 9.601e-03 | 7.911e-03 | 1.214 ± 0.000 | 1.219 [1.144, 1.303] | 0.097 | ≤ 0.007 |

scatter / primary: analytic 0.1375, MC 0.0458

### mc_full, wide window 60-200 keV

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 6 x 60 mm at y=[37,43]; crystals: 256 x NaI (rho=3.67 g/cm3); hole targeting: layer 0, mu*T = 21.6 (opaque -> on)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.440e-03 | 9.006e-03 | 0.937 ± 0.000 | 0.941 [0.909, 0.956] | 0.010 | ≤ 0.010 |
| crystal scatter vs MC xtalscatter | 3.209e-03 | 6.722e-04 | 4.775 ± 0.025 | 4.763 [4.404, 5.245] | 0.251 | ≤ 0.136 |
| collimator scatter vs MC colscatter | 1.484e-03 | 4.556e-04 | 3.258 ± 0.025 | 3.339 [2.399, 5.342] | 0.440 | ≤ 0.200 |
| PE + scatter vs MC total | 1.313e-02 | 1.013e-02 | 1.296 ± 0.000 | 1.300 [1.251, 1.331] | 0.233 | ≤ 0.009 |

scatter / primary: analytic 0.5561, MC 0.1252

