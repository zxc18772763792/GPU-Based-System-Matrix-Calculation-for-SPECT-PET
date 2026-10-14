<!-- ./run_example.sh 511 cpu (scatter model v2, commit 32b650c; the wide-window scatter was computed with 0f7a809, which differs only by the window sub-sampling of 32b650c: < 0.01 % on a test voxel), 4e6 histories per voxel (1.0e9 per MC run); CPU build of the analytic kernels -->
Geometry: 256 crystals, 256 voxels, 1 rotation(s).  Sensitivities are summed over all matrix elements (counts per emitted photon x voxels).

### Photoelectric model (MC physics = primary_pe, ideal window)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary_pe | 8.416e-02 | 8.433e-02 | 0.998 ± 0.000 | 0.998 [0.989, 1.006] | 0.031 | ≤ 0.031 |

### mc_single, photopeak window (Param_Physics)

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 8.014e-02 | 1.050 ± 0.000 | 1.050 [1.041, 1.059] | 0.031 | ≤ 0.043 |
| crystal scatter vs MC xtalscatter | 1.415e-03 | 1.425e-03 | 0.993 ± 0.026 | 0.991 [0.946, 1.041] | 0.194 | ≤ 2.433 |
| collimator scatter vs MC colscatter | 1.729e-02 | 1.742e-02 | 0.993 ± 0.002 | 0.994 [0.953, 1.038] | 0.146 | ≤ 0.199 |
| PE + scatter vs MC total | 1.029e-01 | 9.899e-02 | 1.039 ± 0.000 | 1.040 [1.029, 1.050] | 0.037 | ≤ 0.035 |

scatter / primary: analytic 0.2222, MC 0.2351

### mc_single, wide window 250-650 keV

physics single_scatter, Rayleigh True, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 8.433e-02 | 0.998 ± 0.001 | 0.998 [0.989, 1.006] | 0.031 | ≤ 0.089 |
| crystal scatter vs MC xtalscatter | 2.278e-02 | 2.295e-02 | 0.993 ± 0.003 | 0.993 [0.977, 1.009] | 0.058 | ≤ 0.326 |
| collimator scatter vs MC colscatter | 5.078e-02 | 5.079e-02 | 1.000 ± 0.002 | 1.000 [0.960, 1.039] | 0.145 | ≤ 0.148 |
| PE + scatter vs MC total | 1.577e-01 | 1.581e-01 | 0.998 ± 0.000 | 0.997 [0.986, 1.012] | 0.050 | ≤ 0.047 |

scatter / primary: analytic 0.8741, MC 0.8743

### mc_model, photopeak window (Param_Physics)

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.128e-01 | 0.746 ± 0.000 | 0.746 [0.740, 0.752] | 0.026 | ≤ 0.037 |
| crystal scatter vs MC xtalscatter | 1.415e-03 | 1.967e-03 | 0.719 ± 0.016 | 0.720 [0.688, 0.753] | 0.165 | ≤ 2.148 |
| collimator scatter vs MC colscatter | 1.729e-02 | 2.611e-02 | 0.662 ± 0.001 | 0.662 [0.641, 0.685] | 0.119 | ≤ 0.162 |
| PE + scatter vs MC total | 1.029e-01 | 1.409e-01 | 0.730 ± 0.000 | 0.731 [0.724, 0.736] | 0.031 | ≤ 0.030 |

scatter / primary: analytic 0.2222, MC 0.2490

### mc_model, wide window 250-650 keV

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.557e-01 | 0.540 ± 0.000 | 0.541 [0.537, 0.544] | 0.023 | ≤ 0.070 |
| crystal scatter vs MC xtalscatter | 2.278e-02 | 3.284e-02 | 0.694 ± 0.002 | 0.694 [0.685, 0.703] | 0.048 | ≤ 0.333 |
| collimator scatter vs MC colscatter | 5.078e-02 | 1.106e-01 | 0.459 ± 0.000 | 0.459 [0.448, 0.470] | 0.096 | ≤ 0.099 |
| PE + scatter vs MC total | 1.577e-01 | 2.992e-01 | 0.527 ± 0.000 | 0.527 [0.522, 0.532] | 0.038 | ≤ 0.037 |

scatter / primary: analytic 0.8741, MC 0.9213

### mc_full, photopeak window (Param_Physics)

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x LYSO (rho=7.1 g/cm3); hole targeting: layer 0, mu*T = 2.58 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 8.793e-02 | 0.957 ± 0.001 | 0.957 [0.948, 0.967] | 0.032 | ≤ 0.050 |
| crystal scatter vs MC xtalscatter | 1.415e-03 | 1.994e-03 | 0.710 ± 0.017 | 0.709 [0.678, 0.749] | 0.175 | ≤ 2.199 |
| collimator scatter vs MC colscatter | 1.729e-02 | 4.065e-02 | 0.425 ± 0.000 | 0.426 [0.415, 0.436] | 0.092 | ≤ 0.108 |
| PE + scatter vs MC total | 1.029e-01 | 1.306e-01 | 0.788 ± 0.000 | 0.788 [0.780, 0.795] | 0.035 | ≤ 0.034 |

scatter / primary: analytic 0.2222, MC 0.4850

### mc_full, wide window 250-650 keV

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x LYSO (rho=7.1 g/cm3); hole targeting: layer 0, mu*T = 2.58 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.291e-01 | 0.652 ± 0.001 | 0.652 [0.647, 0.658] | 0.028 | ≤ 0.087 |
| crystal scatter vs MC xtalscatter | 2.278e-02 | 2.443e-02 | 0.933 ± 0.004 | 0.932 [0.917, 0.947] | 0.057 | ≤ 0.462 |
| collimator scatter vs MC colscatter | 5.078e-02 | 1.393e-01 | 0.365 ± 0.000 | 0.364 [0.357, 0.372] | 0.079 | ≤ 0.081 |
| PE + scatter vs MC total | 1.577e-01 | 2.927e-01 | 0.539 ± 0.000 | 0.539 [0.533, 0.543] | 0.040 | ≤ 0.039 |

scatter / primary: analytic 0.8741, MC 1.2684

