<!-- ./run_example.sh 511 cpu (scatter model v2, commit 32b650c; the wide-window scatter was computed with 0f7a809, which differs only by the window sub-sampling of 32b650c: < 0.01 % on a test voxel); Monte Carlo rerun with the MCGen of the code-review PR, which also writes per-component relative errors), 4e6 histories per voxel (1.0e9 per MC run); CPU build of the analytic kernels -->
Geometry: 256 crystals, 256 voxels, 1 rotation(s).  Sensitivities are summed over all matrix elements (counts per emitted photon x voxels).

### Photoelectric model (MC physics = primary_pe, ideal window)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary_pe | 8.416e-02 | 8.431e-02 | 0.998 ± 0.000 | 0.998 [0.990, 1.006] | 0.031 | ≈ 0.031 |

### mc_single, photopeak window (Param_Physics)

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 8.012e-02 | 1.050 ± 0.000 | 1.050 [1.041, 1.059] | 0.031 | ≈ 0.031 |
| crystal scatter vs MC xtalscatter | 1.415e-03 | 1.425e-03 | 0.993 ± 0.002 | 0.992 [0.946, 1.042] | 0.194 | ≈ 0.184 |
| collimator scatter vs MC colscatter | 1.729e-02 | 1.742e-02 | 0.993 ± 0.002 | 0.994 [0.953, 1.038] | 0.146 | ≈ 0.118 |
| PE + scatter vs MC total | 1.029e-01 | 9.897e-02 | 1.039 ± 0.000 | 1.040 [1.029, 1.050] | 0.037 | ≈ 0.035 |

scatter / primary: analytic 0.2222, MC 0.2352

### mc_single, wide window 250-650 keV

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 8.431e-02 | 0.998 ± 0.000 | 0.998 [0.990, 1.006] | 0.031 | ≈ 0.031 |
| crystal scatter vs MC xtalscatter | 2.278e-02 | 2.294e-02 | 0.993 ± 0.001 | 0.993 [0.977, 1.009] | 0.058 | ≈ 0.057 |
| collimator scatter vs MC colscatter | 5.078e-02 | 5.078e-02 | 1.000 ± 0.001 | 1.000 [0.960, 1.039] | 0.145 | ≈ 0.132 |
| PE + scatter vs MC total | 1.577e-01 | 1.580e-01 | 0.998 ± 0.000 | 0.998 [0.986, 1.012] | 0.050 | ≈ 0.047 |

scatter / primary: analytic 0.8741, MC 0.8744

### mc_model, photopeak window (Param_Physics)

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.128e-01 | 0.746 ± 0.000 | 0.747 [0.740, 0.752] | 0.026 | ≈ 0.026 |
| crystal scatter vs MC xtalscatter | 1.415e-03 | 1.966e-03 | 0.719 ± 0.001 | 0.721 [0.688, 0.753] | 0.165 | ≈ 0.160 |
| collimator scatter vs MC colscatter | 1.729e-02 | 2.611e-02 | 0.662 ± 0.001 | 0.662 [0.641, 0.685] | 0.119 | ≈ 0.101 |
| PE + scatter vs MC total | 1.029e-01 | 1.408e-01 | 0.730 ± 0.000 | 0.731 [0.724, 0.736] | 0.031 | ≈ 0.030 |

scatter / primary: analytic 0.2222, MC 0.2490

### mc_model, wide window 250-650 keV

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.557e-01 | 0.541 ± 0.000 | 0.541 [0.537, 0.544] | 0.023 | ≈ 0.023 |
| crystal scatter vs MC xtalscatter | 2.278e-02 | 3.283e-02 | 0.694 ± 0.000 | 0.694 [0.685, 0.703] | 0.048 | ≈ 0.048 |
| collimator scatter vs MC colscatter | 5.078e-02 | 1.106e-01 | 0.459 ± 0.000 | 0.459 [0.448, 0.471] | 0.096 | ≈ 0.092 |
| PE + scatter vs MC total | 1.577e-01 | 2.991e-01 | 0.527 ± 0.000 | 0.527 [0.522, 0.533] | 0.038 | ≈ 0.037 |

scatter / primary: analytic 0.8741, MC 0.9213

### mc_full, photopeak window (Param_Physics)

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x LYSO (rho=7.1 g/cm3); hole targeting: layer 0, mu*T = 2.58 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 8.791e-02 | 0.957 ± 0.000 | 0.958 [0.949, 0.967] | 0.032 | ≈ 0.030 |
| crystal scatter vs MC xtalscatter | 1.415e-03 | 1.993e-03 | 0.710 ± 0.001 | 0.710 [0.678, 0.750] | 0.175 | ≈ 0.167 |
| collimator scatter vs MC colscatter | 1.729e-02 | 4.064e-02 | 0.425 ± 0.000 | 0.426 [0.415, 0.437] | 0.092 | ≈ 0.080 |
| PE + scatter vs MC total | 1.029e-01 | 1.305e-01 | 0.788 ± 0.000 | 0.788 [0.780, 0.796] | 0.035 | ≈ 0.034 |

scatter / primary: analytic 0.2222, MC 0.4850

### mc_full, wide window 250-650 keV

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x LYSO (rho=7.1 g/cm3); hole targeting: layer 0, mu*T = 2.58 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.290e-01 | 0.652 ± 0.000 | 0.652 [0.647, 0.658] | 0.028 | ≈ 0.025 |
| crystal scatter vs MC xtalscatter | 2.278e-02 | 2.442e-02 | 0.933 ± 0.001 | 0.932 [0.918, 0.947] | 0.057 | ≈ 0.056 |
| collimator scatter vs MC colscatter | 5.078e-02 | 1.392e-01 | 0.365 ± 0.000 | 0.365 [0.357, 0.372] | 0.079 | ≈ 0.077 |
| PE + scatter vs MC total | 1.577e-01 | 2.927e-01 | 0.539 ± 0.000 | 0.539 [0.533, 0.543] | 0.040 | ≈ 0.039 |

scatter / primary: analytic 0.8741, MC 1.2685

