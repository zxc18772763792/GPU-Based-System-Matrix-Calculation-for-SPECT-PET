<!-- ./run_example.sh 511 cpu, 4e6 histories per voxel (1.0e9 per MC run); CPU build of the original analytic kernels -->
Geometry: 256 crystals, 256 voxels, 1 rotation(s).  Sensitivities are summed over all matrix elements (counts per emitted photon x voxels).

### Photoelectric model (MC physics = primary_pe, ideal window)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary_pe | 8.416e-02 | 8.433e-02 | 0.998 ± 0.000 | 0.998 [0.989, 1.006] | 0.031 | ≤ 0.031 |

### mc_model, photopeak window (Param_Physics)

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.128e-01 | 0.746 ± 0.000 | 0.746 [0.740, 0.752] | 0.026 | ≤ 0.037 |
| crystal scatter vs MC xtalscatter | 0.000e+00 | 1.967e-03 | 0.000 ± 0.000 | 0.000 [0.000, 0.000] | - | ≤ 2.148 |
| collimator scatter vs MC colscatter | 5.458e-03 | 2.611e-02 | 0.209 ± 0.000 | 0.209 [0.191, 0.223] | 0.140 | ≤ 0.162 |
| PE + scatter vs MC total | 8.962e-02 | 1.409e-01 | 0.636 ± 0.000 | 0.636 [0.630, 0.642] | 0.033 | ≤ 0.030 |

scatter / primary: analytic 0.0648, MC 0.2490

### mc_model, wide window 250-650 keV

physics full, Rayleigh False, fluorescence False, 1.02e+09 histories; collimator layer 0: params(tot=0.2406,pe=0.1092,compton=0.1314/mm @E0), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x params(tot=0.07736,pe=0.02537,compton=0.052/mm @E0); hole targeting: layer 0, mu*T = 2.41 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.557e-01 | 0.540 ± 0.000 | 0.541 [0.537, 0.544] | 0.023 | ≤ 0.070 |
| crystal scatter vs MC xtalscatter | 2.193e-01 | 3.284e-02 | 6.678 ± 0.022 | 6.687 [6.469, 6.833] | 0.058 | ≤ 0.333 |
| collimator scatter vs MC colscatter | 1.649e-02 | 1.106e-01 | 0.149 ± 0.000 | 0.149 [0.138, 0.158] | 0.111 | ≤ 0.099 |
| PE + scatter vs MC total | 3.200e-01 | 2.992e-01 | 1.069 ± 0.000 | 1.069 [1.054, 1.082] | 0.050 | ≤ 0.037 |

scatter / primary: analytic 2.8017, MC 0.9213

### mc_full, photopeak window (Param_Physics)

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x LYSO (rho=7.1 g/cm3); hole targeting: layer 0, mu*T = 2.58 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 8.793e-02 | 0.957 ± 0.001 | 0.957 [0.948, 0.967] | 0.032 | ≤ 0.050 |
| crystal scatter vs MC xtalscatter | 0.000e+00 | 1.994e-03 | 0.000 ± 0.000 | 0.000 [0.000, 0.000] | - | ≤ 2.199 |
| collimator scatter vs MC colscatter | 5.458e-03 | 4.065e-02 | 0.134 ± 0.000 | 0.135 [0.123, 0.143] | 0.117 | ≤ 0.108 |
| PE + scatter vs MC total | 8.962e-02 | 1.306e-01 | 0.686 ± 0.000 | 0.686 [0.679, 0.693] | 0.038 | ≤ 0.034 |

scatter / primary: analytic 0.0648, MC 0.4850

### mc_full, wide window 250-650 keV

physics full, Rayleigh True, fluorescence True, 1.02e+09 histories; collimator layer 0: W (rho=19.3 g/cm3), 9 holes, plate 60 x 10 x 60 mm at y=[35,45]; crystals: 256 x LYSO (rho=7.1 g/cm3); hole targeting: layer 0, mu*T = 2.58 (penetrable -> off)

| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |
|---|---|---|---|---|---|---|
| PE vs MC primary | 8.416e-02 | 1.291e-01 | 0.652 ± 0.001 | 0.652 [0.647, 0.658] | 0.028 | ≤ 0.087 |
| crystal scatter vs MC xtalscatter | 2.193e-01 | 2.443e-02 | 8.978 ± 0.042 | 8.987 [8.647, 9.251] | 0.064 | ≤ 0.462 |
| collimator scatter vs MC colscatter | 1.649e-02 | 1.393e-01 | 0.118 ± 0.000 | 0.118 [0.109, 0.126] | 0.098 | ≤ 0.081 |
| PE + scatter vs MC total | 3.200e-01 | 2.927e-01 | 1.093 ± 0.000 | 1.093 [1.077, 1.107] | 0.052 | ≤ 0.039 |

scatter / primary: analytic 2.8017, MC 1.2684

