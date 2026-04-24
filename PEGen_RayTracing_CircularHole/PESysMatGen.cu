// GPU based photon-electric system matrix generation
// Corrected optimized version
//
// Bug fix from previous attempt: maxLateral was divided by dist_vox_det,
// making the spatial hash search area 16x too small and causing 52% of
// nonzero elements to be missed. The correct formula is:
//   maxLateral = detDiag * 0.5 * t_to_col + safety_margin
// (no division by distance — t_to_col is already a dimensionless fraction)
//
// Structure:
//   1. Per-thread early exits (cheap geometric checks)
//   2. Pre-collect nearby holes from spatial hash (ONCE, deduplicated)
//   3. Pre-collect nearby detectors via angular culling (ONCE)
//   4. 1024-subelement loop using pre-collected lists
//
// original author: xingchun zheng @ tsinghua university

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include "PESysMatGen.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

using namespace std;

// ============================================================================
// Ray-geometry intersection functions
// Identical to original physics — all degenerate-case handling preserved.
// ============================================================================

__device__ float length_box_ray(
    float x_in, float y_in, float z_in,
    float x_out, float y_out, float z_out,
    float x1_box, float y1_box, float z1_box,
    float x2_box, float y2_box, float z2_box)
{
    float eps = 0.001f;

    if (fabsf(x_out - x_in) < eps && fabsf(y_out - y_in) < eps && fabsf(z_out - z_in) < eps)
        return 0.000f;

    // Double-degenerate cases
    if (fabsf(x_out - x_in) < eps && fabsf(y_out - y_in) < eps) {
        if ((x_in >= x1_box && x_in <= x2_box) && (y_in >= y1_box && y_in <= y2_box))
            if ((z_in <= z1_box && z_out >= z2_box) || (z_out <= z1_box && z_in >= z2_box))
                return fabsf(z2_box - z1_box);
            else return 0.000f;
        else return 0.000f;
    }
    if (fabsf(z_out - z_in) < eps && fabsf(y_out - y_in) < eps) {
        if ((z_in >= z1_box && z_in <= z2_box) && (y_in >= y1_box && y_in <= y2_box))
            if ((x_in <= x1_box && x_out >= x2_box) || (x_out <= x1_box && x_in >= x2_box))
                return fabsf(x2_box - x1_box);
            else return 0.000f;
        else return 0.000f;
    }
    if (fabsf(x_out - x_in) < eps && fabsf(z_out - z_in) < eps) {
        if ((x_in >= x1_box && x_in <= x2_box) && (z_in >= z1_box && z_in <= z2_box)) {
            if ((y_in <= y1_box && y_out >= y2_box) || (y_out <= y1_box && y_in >= y2_box))
                return fabsf(y2_box - y1_box);
            else return 0.000f;
        } else return 0.000f;
    }

    // Single-degenerate: x constant
    if (fabsf(x_out - x_in) < eps && (x_in >= x2_box || x_in <= x1_box))
        return 0.000f;
    else if (fabsf(x_out - x_in) < eps) {
        float tmin, tmax, tzmin, tzmax;
        float t_inout = sqrtf((y_out-y_in)*(y_out-y_in) + (z_out-z_in)*(z_out-z_in));
        float idy = t_inout / (y_out - y_in);
        float idz = t_inout / (z_out - z_in);
        if (idy < 0.0f) { tmin = (y2_box-y_in)*idy; tmax = (y1_box-y_in)*idy; }
        else             { tmax = (y2_box-y_in)*idy; tmin = (y1_box-y_in)*idy; }
        if (idz < 0.0f) { tzmin = (z2_box-z_in)*idz; tzmax = (z1_box-z_in)*idz; }
        else             { tzmax = (z2_box-z_in)*idz; tzmin = (z1_box-z_in)*idz; }
        if (tmin > tzmax || tzmin > tmax) return 0.0f;
        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;
        if ((tmax-tmin) < eps) return 0.000f;
        else if (tmin >= t_inout || tmax >= t_inout) return 0.000f;
        else if (tmin <= eps || tmax <= eps) return 0.000f;
        else return (tmax - tmin);
    }

    // Single-degenerate: y constant
    if (fabsf(y_out - y_in) < eps && (y_in >= y2_box || y_in <= y1_box))
        return 0.000f;
    else if (fabsf(y_out - y_in) < eps) {
        float tmin, tmax, tzmin, tzmax;
        float t_inout = sqrtf((x_out-x_in)*(x_out-x_in) + (z_out-z_in)*(z_out-z_in));
        float idx = t_inout / (x_out - x_in);
        float idz = t_inout / (z_out - z_in);
        if (idx < 0.0f) { tmin = (x2_box-x_in)*idx; tmax = (x1_box-x_in)*idx; }
        else             { tmax = (x2_box-x_in)*idx; tmin = (x1_box-x_in)*idx; }
        if (idz < 0.0f) { tzmin = (z2_box-z_in)*idz; tzmax = (z1_box-z_in)*idz; }
        else             { tzmax = (z2_box-z_in)*idz; tzmin = (z1_box-z_in)*idz; }
        if (tmin > tzmax || tzmin > tmax) return 0.0f;
        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;
        if ((tmax-tmin) < eps) return 0.000f;
        else if (tmin >= t_inout || tmax >= t_inout) return 0.000f;
        else if (tmin <= eps || tmax <= eps) return 0.000f;
        else return (tmax - tmin);
    }

    // Single-degenerate: z constant
    if (fabsf(z_out - z_in) < eps && (z_in >= z2_box || z_in <= z1_box))
        return 0.000f;
    else if (fabsf(z_out - z_in) < eps) {
        float tmin, tmax, tymin, tymax;
        float t_inout = sqrtf((x_out-x_in)*(x_out-x_in) + (y_out-y_in)*(y_out-y_in));
        float idx = t_inout / (x_out - x_in);
        float idy = t_inout / (y_out - y_in);
        if (idx < 0.0f) { tmin = (x2_box-x_in)*idx; tmax = (x1_box-x_in)*idx; }
        else             { tmax = (x2_box-x_in)*idx; tmin = (x1_box-x_in)*idx; }
        if (idy < 0.0f) { tymin = (y2_box-y_in)*idy; tymax = (y1_box-y_in)*idy; }
        else             { tymax = (y2_box-y_in)*idy; tymin = (y1_box-y_in)*idy; }
        if (tmin > tymax || tymin > tmax) return 0.0f;
        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;
        if ((tmax-tmin) < eps) return 0.000f;
        else if (tmin >= t_inout || tmax >= t_inout) return 0.000f;
        else if (tmin <= eps || tmax <= eps) return 0.000f;
        else return (tmax - tmin);
    }

    // General 3D case
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    float t_inout = sqrtf((x_out-x_in)*(x_out-x_in)+(y_out-y_in)*(y_out-y_in)+(z_out-z_in)*(z_out-z_in));
    float idx = t_inout / (x_out - x_in);
    float idy = t_inout / (y_out - y_in);
    float idz = t_inout / (z_out - z_in);

    if (idx < 0.0f) { tmin = (x2_box-x_in)*idx; tmax = (x1_box-x_in)*idx; }
    else             { tmax = (x2_box-x_in)*idx; tmin = (x1_box-x_in)*idx; }
    if (idy < 0.0f) { tymin = (y2_box-y_in)*idy; tymax = (y1_box-y_in)*idy; }
    else             { tymax = (y2_box-y_in)*idy; tymin = (y1_box-y_in)*idy; }
    if (tmin > tymax || tymin > tmax) return 0.0f;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;
    if (idz < 0.0f) { tzmin = (z2_box-z_in)*idz; tzmax = (z1_box-z_in)*idz; }
    else             { tzmax = (z2_box-z_in)*idz; tzmin = (z1_box-z_in)*idz; }
    if (tmin > tzmax || tzmin > tmax) return 0.0f;
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    if ((tmax-tmin) < eps) return 0.000f;
    else if (tmin >= t_inout || tmax >= t_inout) return 0.000f;
    else if (tmin <= eps || tmax <= eps) return 0.000f;
    else return (tmax - tmin);
}


__device__ float length_box_ray_inside(
    float x_in, float y_in, float z_in,
    float x_out, float y_out, float z_out,
    float x1_box, float y1_box, float z1_box,
    float x2_box, float y2_box, float z2_box)
{
    if (fabsf(y_out - y_in) < 0.001f) return 0.000f;
    float eps = 0.001f;

    if (fabsf(x_out-x_in) < eps && fabsf(y_out-y_in) < eps && fabsf(z_out-z_in) < eps)
        return 0.000f;

    // Double-degenerate (endpoint inside box)
    if (fabsf(x_out-x_in) < eps && fabsf(y_out-y_in) < eps) {
        if ((x_in >= x1_box && x_in <= x2_box) && (y_in >= y1_box && y_in <= y2_box)) {
            if (z_in < z_out) return (z_out - z1_box);
            else if (z_in > z_out) return (z2_box - z_out);
        }
        return 0.000f;
    }
    if (fabsf(z_out-z_in) < eps && fabsf(y_out-y_in) < eps) {
        if ((z_in >= z1_box && z_in <= z2_box) && (y_in >= y1_box && y_in <= y2_box)) {
            if (x_in < x_out) return (x_out - x1_box);
            else if (x_in > x_out) return (x2_box - x_out);
        }
        return 0.000f;
    }
    if (fabsf(x_out-x_in) < eps && fabsf(z_out-z_in) < eps) {
        if ((x_in >= x1_box && x_in <= x2_box) && (z_in >= z1_box && z_in <= z2_box)) {
            if (y_in < y_out) return (y_out - y1_box);
            else if (y_in > y_out) return (y2_box - y_out);
        }
        return 0.000f;
    }

    // Single-degenerate: x constant
    if (fabsf(x_out-x_in) < eps && (x_in >= x2_box || x_in <= x1_box))
        return 0.000f;
    else if (fabsf(x_out-x_in) < eps) {
        float t_inout = sqrtf((y_out-y_in)*(y_out-y_in) + (z_out-z_in)*(z_out-z_in));
        float idy = t_inout / (y_out - y_in);
        float idz = t_inout / (z_out - z_in);
        float tmin, tmax, tzmin, tzmax;
        if (idy < 0.0f) { tmin = (y2_box-y_in)*idy; tmax = (y1_box-y_in)*idy; }
        else             { tmax = (y2_box-y_in)*idy; tmin = (y1_box-y_in)*idy; }
        if (idz < 0.0f) { tzmin = (z2_box-z_in)*idz; tzmax = (z1_box-z_in)*idz; }
        else             { tzmax = (z2_box-z_in)*idz; tzmin = (z1_box-z_in)*idz; }
        if (tmin > tzmax || tzmin > tmax) return 0.0f;
        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;
        if ((tmax-tmin) < eps) return 0.0f;
        else if (tmin >= t_inout) return 0.0f;
        else if (tmax <= eps) return 0.0f;
        else if (tmax >= t_inout && tmin > eps) return (t_inout - tmin);
        else if (tmin <= eps && tmax <= t_inout) return tmax;
        else if (tmin <= eps && tmax >= t_inout) return t_inout;
        else return (tmax - tmin);
    }

    // Single-degenerate: y constant
    if (fabsf(y_out-y_in) < eps && (y_in >= y2_box || y_in <= y1_box))
        return 0.000f;
    else if (fabsf(y_out-y_in) < eps) {
        float t_inout = sqrtf((x_out-x_in)*(x_out-x_in) + (z_out-z_in)*(z_out-z_in));
        float idx = t_inout / (x_out - x_in);
        float idz = t_inout / (z_out - z_in);
        float tmin, tmax, tzmin, tzmax;
        if (idx < 0.0f) { tmin = (x2_box-x_in)*idx; tmax = (x1_box-x_in)*idx; }
        else             { tmax = (x2_box-x_in)*idx; tmin = (x1_box-x_in)*idx; }
        if (idz < 0.0f) { tzmin = (z2_box-z_in)*idz; tzmax = (z1_box-z_in)*idz; }
        else             { tzmax = (z2_box-z_in)*idz; tzmin = (z1_box-z_in)*idz; }
        if (tmin > tzmax || tzmin > tmax) return 0.0f;
        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;
        if ((tmax-tmin) < eps) return 0.0f;
        else if (tmin >= t_inout) return 0.0f;
        else if (tmax <= eps) return 0.0f;
        else if (tmax >= t_inout && tmin > eps) return (t_inout - tmin);
        else if (tmin <= eps && tmax <= t_inout) return tmax;
        else if (tmin <= eps && tmax >= t_inout) return t_inout;
        else return (tmax - tmin);
    }

    // Single-degenerate: z constant
    if (fabsf(z_out-z_in) < eps && (z_in >= z2_box || z_in <= z1_box))
        return 0.000f;
    else if (fabsf(z_out-z_in) < eps) {
        float t_inout = sqrtf((x_out-x_in)*(x_out-x_in) + (y_out-y_in)*(y_out-y_in));
        float idx = t_inout / (x_out - x_in);
        float idy = t_inout / (y_out - y_in);
        float tmin, tmax, tymin, tymax;
        if (idx < 0.0f) { tmin = (x2_box-x_in)*idx; tmax = (x1_box-x_in)*idx; }
        else             { tmax = (x2_box-x_in)*idx; tmin = (x1_box-x_in)*idx; }
        if (idy < 0.0f) { tymin = (y2_box-y_in)*idy; tymax = (y1_box-y_in)*idy; }
        else             { tymax = (y2_box-y_in)*idy; tymin = (y1_box-y_in)*idy; }
        if (tmin > tymax || tymin > tmax) return 0.0f;
        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;
        if ((tmax-tmin) < eps) return 0.0f;
        else if (tmin >= t_inout) return 0.0f;
        else if (tmax <= eps) return 0.0f;
        else if (tmax >= t_inout && tmin > eps) return (t_inout - tmin);
        else if (tmin <= eps && tmax <= t_inout) return tmax;
        else if (tmin <= eps && tmax >= t_inout) return t_inout;
        else return (tmax - tmin);
    }

    // General 3D case
    float t_inout = sqrtf((x_out-x_in)*(x_out-x_in)+(y_out-y_in)*(y_out-y_in)+(z_out-z_in)*(z_out-z_in));
    float idx = t_inout / (x_out - x_in);
    float idy = t_inout / (y_out - y_in);
    float idz = t_inout / (z_out - z_in);
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    if (idx < 0.0f) { tmin = (x2_box-x_in)*idx; tmax = (x1_box-x_in)*idx; }
    else             { tmax = (x2_box-x_in)*idx; tmin = (x1_box-x_in)*idx; }
    if (idy < 0.0f) { tymin = (y2_box-y_in)*idy; tymax = (y1_box-y_in)*idy; }
    else             { tymax = (y2_box-y_in)*idy; tymin = (y1_box-y_in)*idy; }
    if (tmin > tymax || tymin > tmax) return 0.0f;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;
    if (idz < 0.0f) { tzmin = (z2_box-z_in)*idz; tzmax = (z1_box-z_in)*idz; }
    else             { tzmax = (z2_box-z_in)*idz; tzmin = (z1_box-z_in)*idz; }
    if (tmin > tzmax || tzmin > tmax) return 0.0f;
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;  // CORRECT: tzmax (not tymax)

    if ((tmax-tmin) < eps) return 0.0f;
    else if (tmin >= t_inout) return 0.0f;
    else if (tmax <= eps) return 0.0f;
    else if (tmax >= t_inout && tmin > eps) return (t_inout - tmin);
    else if (tmin <= eps && tmax <= t_inout) return tmax;
    else if (tmin <= eps && tmax >= t_inout) return t_inout;
    else return (tmax - tmin);
}


__device__ float length_cylinder_ray(
    float x_in, float y_in, float z_in,
    float x_out, float y_out, float z_out,
    float x_cylinder, float y1_cylinder, float y2_cylinder,
    float z_cylinder, float radius)
{
    if (fabsf(y1_cylinder - y2_cylinder) < 0.001f) return 0.000f;

    float t_inout = sqrtf((x_out-x_in)*(x_out-x_in)+(y_out-y_in)*(y_out-y_in)+(z_out-z_in)*(z_out-z_in));
    float k_x = (x_out - x_in) / t_inout;
    float k_y = (y_out - y_in) / t_inout;
    float k_z = (z_out - z_in) / t_inout;

    float x_lp = x_in + k_x / k_y * (y1_cylinder - y_in);
    float x_rp = x_in + k_x / k_y * (y2_cylinder - y_in);
    float z_lp = z_in + k_z / k_y * (y1_cylinder - y_in);
    float z_rp = z_in + k_z / k_y * (y2_cylinder - y_in);

    float tmin = (y1_cylinder - y_in) / k_y;
    float tmax = (y2_cylinder - y_in) / k_y;

    int fl = ((x_lp-x_cylinder)*(x_lp-x_cylinder)+(z_lp-z_cylinder)*(z_lp-z_cylinder) <= radius*radius) ? 1 : 0;
    int fr = ((x_rp-x_cylinder)*(x_rp-x_cylinder)+(z_rp-z_cylinder)*(z_rp-z_cylinder) <= radius*radius) ? 1 : 0;

    if (fl && fr) {
        if (tmin <= 0.0001f || tmax <= 0.0001f) return 0.0f;
        if (tmin >= t_inout || tmax >= t_inout) return 0.0f;
        return fabsf(tmax - tmin);
    }

    float x_ = x_in - x_cylinder;
    float z_ = z_in - z_cylinder;
    float kxkz = k_x*k_x + k_z*k_z;
    float Delta_ = kxkz * radius*radius - (k_x*z_ - k_z*x_)*(k_x*z_ - k_z*x_);
    if (Delta_ <= 0.00001f) return 0.0f;

    float neg_b = -(k_x*x_ + k_z*z_);
    float sq = sqrtf(Delta_);
    float t1 = (neg_b - sq) / kxkz;
    float t2 = (neg_b + sq) / kxkz;

    if (!fl && !fr) {
        if (t1 >= tmin && t2 <= tmax) {
            if (t2 <= 0.0001f || t1 <= 0.0001f) return 0.0f;
            if (t2 >= t_inout || t1 >= t_inout) return 0.0f;
            return (t2 - t1);
        }
        return 0.0f;
    }
    if (fl && !fr) {
        if (t2 <= 0.0001f || tmin <= 0.0001f) return 0.0f;
        if (t2 >= t_inout || tmin >= t_inout) return 0.0f;
        return (t2 - tmin);
    }
    if (!fl && fr) {
        if (tmax <= 0.0001f || t1 <= 0.0001f) return 0.0f;
        if (tmax >= t_inout || t1 >= t_inout) return 0.0f;
        return (tmax - t1);
    }
    return 0.0f;
}


// ============================================================================
// Main kernel
// ============================================================================
__global__ void photodetectorCudaMe(
    float* __restrict__ dst,
    const float* __restrict__ devCol,
    const float* __restrict__ devDet,
    const float* __restrict__ devImg,
    const int* __restrict__ gridIdx,
    const int* __restrict__ gridCnt,
    float grid_ox, float grid_oz,
    float grid_cell,
    int grid_nx, int grid_nz,
    int grid_mpc,
    long long numProjSingle,
    long long numImagebin)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numProjSingle * numImagebin) return;

    int row = (int)(tid / numImagebin);
    int col = (int)(tid % numImagebin);

    // ---- Image params ----
    int nX = (int)devImg[0], nY = (int)devImg[1], nZ = (int)devImg[2];
    float wX = devImg[3], wY = devImg[4], wZ = devImg[5];
    float rotAngle = devImg[20] * devImg[7];
    float shX = devImg[8], shY = devImg[9], shZ = devImg[10];
    float fov2col = devImg[11];

    int izV = col / (nY * nX);
    int rem = col % (nY * nX);
    int iyV = rem / nX;
    int ixV = rem % nX;

    float xV = (ixV - nX/2.0f + 0.5f)*wX + shX;
    float yV = (iyV - nY/2.0f + 0.5f)*wY + shY;
    float zV = (izV - nZ/2.0f + 0.5f)*wZ + shZ;

    float cosR = cosf(rotAngle), sinR = sinf(rotAngle);
    float xImage = xV*cosR - yV*sinR;
    float yImage = xV*sinR + yV*cosR;
    float zImage = zV;

    // ---- Target detector ----
    unsigned int idxDet = row;
    float xDC = devDet[12*idxDet+1];
    float yDC = devDet[12*idxDet+2] + fov2col;
    float zDC = devDet[12*idxDet+3];
    float wDet = devDet[12*idxDet+4];
    float tDet = devDet[12*idxDet+5];
    float hDet = devDet[12*idxDet+6];
    float coeffTot = devDet[12*idxDet+7];
    float coeffPE  = devDet[12*idxDet+8];
    float rotDet   = devDet[12*idxDet+11];

    if (coeffTot <= 0.01f || yImage >= yDC) {
        dst[tid] = 0.0f;
        return;
    }

    int numColLayers = (int)floorf(devCol[0] + 0.000001f);
    int numDetBins = (int)floorf(devDet[0] + 0.000001f);

    float colHalfX[10], colY1[10], colY2[10], colHalfZ[10], colThick[10], colCoeff[10];
    int colNHoles[10];
    for (int m = 0; m < numColLayers; m++) {
        int b = (m+1)*10;
        colNHoles[m] = (int)devCol[b];
        float cw = devCol[b+1], ct = devCol[b+2], ch = devCol[b+3], co = devCol[b+4];
        colHalfX[m] = cw/2.0f;
        colY1[m] = -ct/2.0f + fov2col + co;
        colY2[m] = ct/2.0f + fov2col + co;
        colHalfZ[m] = ch/2.0f;
        colThick[m] = ct;
        colCoeff[m] = devCol[b+5];
    }

    // ======== EARLY EXIT: center ray misses collimator box ========
    for (int m = 0; m < numColLayers; m++) {
        float tl = length_box_ray(xImage, yImage, zImage, xDC, yDC, zDC,
                                  -colHalfX[m], colY1[m], -colHalfZ[m],
                                   colHalfX[m], colY2[m],  colHalfZ[m]);
        if (tl < 0.1f) { dst[tid] = 0.0f; return; }
    }

    // ======== PRE-COLLECT HOLES (spatial hash, deduplicated) ========
    float colMidY = (colY1[0] + colY2[0]) / 2.0f;
    float rayDY = yDC - yImage;
    float t_to_col = (colMidY - yImage) / rayDY;
    float xAtCol = xImage + (xDC - xImage) * t_to_col;
    float zAtCol = zImage + (zDC - zImage) * t_to_col;

    float detDiag = sqrtf(wDet*wDet + hDet*hDet + tDet*tDet);
    // FIXED: no division by distance. t_to_col is already dimensionless [0,1].
    float maxLateral = detDiag * 0.5f * t_to_col + 2.0f;

    int totalHoles = 0;
    for (int m = 0; m < numColLayers; m++) totalHoles += colNHoles[m];
    float maxHoleR = (totalHoles > 0) ? devCol[104] : 0.0f;
    float searchR = maxLateral + maxHoleR;

    int cx_min = max(0, (int)floorf((xAtCol - searchR - grid_ox) / grid_cell));
    int cx_max = min(grid_nx-1, (int)floorf((xAtCol + searchR - grid_ox) / grid_cell));
    int cz_min = max(0, (int)floorf((zAtCol - searchR - grid_oz) / grid_cell));
    int cz_max = min(grid_nz-1, (int)floorf((zAtCol + searchR - grid_oz) / grid_cell));

    const int MAX_HOLES = 32;
    int holeIds[MAX_HOLES];
    int holeLayers[MAX_HOLES];
    int nHoles = 0;

    for (int gx = cx_min; gx <= cx_max; gx++) {
        for (int gz = cz_min; gz <= cz_max; gz++) {
            int cid = gx * grid_nz + gz;
            int cnt = gridCnt[cid];
            int base = cid * grid_mpc;
            for (int h = 0; h < cnt && nHoles < MAX_HOLES; h++) {
                int hid = gridIdx[base + h];
                bool dup = false;
                for (int k = 0; k < nHoles; k++) {
                    if (holeIds[k] == hid) { dup = true; break; }
                }
                if (dup) continue;
                float hx = devCol[hid*9+100];
                float hz = devCol[hid*9+103];
                float dx = xAtCol - hx, dz = zAtCol - hz;
                if (dx*dx + dz*dz <= searchR*searchR) {
                    int layer = 0, offset = hid;
                    for (int m = 0; m < numColLayers; m++) {
                        if (offset < colNHoles[m]) { layer = m; break; }
                        offset -= colNHoles[m];
                    }
                    holeIds[nHoles] = hid;
                    holeLayers[nHoles] = layer;
                    nHoles++;
                }
            }
        }
    }

    // ======== PRE-COLLECT CROSS-DETECTORS ========
    float dist_vd = sqrtf((xDC-xImage)*(xDC-xImage)+(yDC-yImage)*(yDC-yImage)+(zDC-zImage)*(zDC-zImage));
    float crit_self = detDiag / dist_vd / 2.0f;
    float xDir = (xDC-xImage)/dist_vd;
    float yDir = (yDC-yImage)/dist_vd;
    float zDir = (zDC-zImage)/dist_vd;

    const int MAX_CROSS_DET = 64;
    int crossDetIds[MAX_CROSS_DET];
    int nCrossDet = 0;

    for (int id = 0; id < numDetBins; id++) {
        if (id == (int)idxDet) continue;
        float ow = devDet[12*id+4], ot = devDet[12*id+5], oh = devDet[12*id+6];
        float oc = devDet[12*id+7];
        if (oc <= 0.01f) continue;
        float ox = devDet[12*id+1], oy = devDet[12*id+2]+fov2col, oz = devDet[12*id+3];
        float od = sqrtf((xImage-ox)*(xImage-ox)+(yImage-oy)*(yImage-oy)+(zImage-oz)*(zImage-oz));
        float crit_o = sqrtf(ow*ow+oh*oh+ot*ot) / od / 2.0f;
        float dxo = (ox-xImage)/od, dyo = (oy-yImage)/od, dzo = (oz-zImage)/od;
        float ad = sqrtf((dxo-xDir)*(dxo-xDir)+(dyo-yDir)*(dyo-yDir)+(dzo-zDir)*(dzo-zDir));
        if (ad <= crit_o + crit_self && nCrossDet < MAX_CROSS_DET)
            crossDetIds[nCrossDet++] = id;
    }

    // ======== SUBELEMENT LOOP ========
    float cosD = cosf(-rotDet), sinD = sinf(-rotDet);
    float xIs = (xImage-xDC)*cosD - (zImage-zDC)*sinD;
    float yIs = yImage - yDC;
    float zIs = (xImage-xDC)*sinD + (zImage-zDC)*cosD;
    float x1ds = -wDet/2.0f, x2ds = wDet/2.0f;
    float y1ds = -tDet/2.0f, y2ds = tDet/2.0f;
    float z1ds = -hDet/2.0f, z2ds = hDet/2.0f;

    const unsigned int divX = 8, divY = 16, divZ = 8;
    float cosDetR = cosf(rotDet), sinDetR = sinf(rotDet);
    float areaX = tDet*hDet / (float)(divZ*divY);
    float areaZ = wDet*tDet / (float)(divX*divY);
    float areaY = wDet*hDet / (float)(divX*divZ);

    float final_val = 0.0f;

    for (unsigned int nz = 0; nz < divZ; nz++) {
        float zDs = -hDet/2.0f + (nz+0.5f)/(float)divZ * hDet;
        float z1du = ((float)nz/(float)divZ - 0.5f) * hDet;
        float z2du = (((float)nz+1.0f)/(float)divZ - 0.5f) * hDet;

        for (unsigned int nx = 0; nx < divX; nx++) {
            float xDs = -wDet/2.0f + (nx+0.5f)/(float)divX * wDet;
            float x1du = ((float)nx/(float)divX - 0.5f) * wDet;
            float x2du = (((float)nx+1.0f)/(float)divX - 0.5f) * wDet;

            float xDr = xDs*cosDetR - zDs*sinDetR;
            float zDr = xDs*sinDetR + zDs*cosDetR;
            float xDet = xDC + xDr;
            float zDet = zDC + zDr;

            for (unsigned int ny = 0; ny < divY; ny++) {
                float yDs = -tDet/2.0f + (ny+0.5f)/(float)divY * tDet;
                float y1du = ((float)ny/(float)divY - 0.5f) * tDet;
                float y2du = (((float)ny+1.0f)/(float)divY - 0.5f) * tDet;
                float yDet = yDC + yDs;

                float dsq = (yDet-yImage)*(yDet-yImage)+(xDet-xImage)*(xDet-xImage)+(zDet-zImage)*(zDet-zImage);
                float d_sub = sqrtf(dsq);
                float COSangle = (yDet-yImage) / d_sub;

                float inv_d3 = 1.0f / (4.0f * M_PI * dsq * d_sub);
                float sa_x = areaX * inv_d3 * fabsf((xDet-xImage)*cosDetR - (zDet-zImage)*sinDetR);
                float sa_z = areaZ * inv_d3 * fabsf((xDet-xImage)*sinDetR + (zDet-zImage)*cosDetR);
                float sa_y = areaY * inv_d3 * fabsf(yDet - yImage);
                float solid_angle = fmaxf(fmaxf(sa_x, sa_y), sa_z);

                float atten = 0.0f;
                bool blocked = false;

                for (int m = 0; m < numColLayers; m++) {
                    float tl = length_box_ray(xImage, yImage, zImage, xDet, yDet, zDet,
                                              -colHalfX[m], colY1[m], -colHalfZ[m],
                                               colHalfX[m], colY2[m],  colHalfZ[m]);
                    if (tl < 0.1f) { blocked = true; break; }
                }
                if (blocked) continue;

                // Holes — from pre-collected deduplicated list
                float holeLen[10];
                for (int m = 0; m < numColLayers; m++) holeLen[m] = 0.0f;
                for (int ii = 0; ii < nHoles; ii++) {
                    int hid = holeIds[ii];
                    float tmp = length_cylinder_ray(xImage, yImage, zImage, xDet, yDet, zDet,
                        devCol[hid*9+100], devCol[hid*9+101]+fov2col, devCol[hid*9+102]+fov2col,
                        devCol[hid*9+103], devCol[hid*9+104]);
                    holeLen[holeLayers[ii]] += tmp;
                    atten += devCol[hid*9+105] * tmp;
                }
                for (int m = 0; m < numColLayers; m++) {
                    float LiC = colThick[m] / COSangle;
                    if ((LiC - holeLen[m]) >= 0.00001f)
                        atten += colCoeff[m] * (LiC - holeLen[m]);
                }

                if (atten > 30.0f) continue;

                // Other crystals — from pre-collected list
                for (int ii = 0; ii < nCrossDet; ii++) {
                    int oid = crossDetIds[ii];
                    float ox = devDet[12*oid+1], oy = devDet[12*oid+2]+fov2col, oz = devDet[12*oid+3];
                    float ow = devDet[12*oid+4], ot = devDet[12*oid+5], oh = devDet[12*oid+6];
                    float oc = devDet[12*oid+7], orot = devDet[12*oid+11];
                    float cosA = cosf(-orot), sinA = sinf(-orot);
                    float dxIA = xImage-ox, dzIA = zImage-oz;
                    float dxDA = xDet-ox, dzDA = zDet-oz;
                    float la = length_box_ray(
                        dxIA*cosA-dzIA*sinA, yImage-oy, dxIA*sinA+dzIA*cosA,
                        dxDA*cosA-dzDA*sinA, yDet-oy,   dxDA*sinA+dzDA*cosA,
                        -0.5f*ow,-0.5f*ot,-0.5f*oh, 0.5f*ow,0.5f*ot,0.5f*oh);
                    atten += la * oc;
                }

                float l1 = length_box_ray_inside(xIs, yIs, zIs, xDs, yDs, zDs,
                    x1ds, y1ds, z1ds, x2ds, y2ds, z2ds);
                float l2 = length_box_ray_inside(xIs, yIs, zIs, xDs, yDs, zDs,
                    x1du, y1du, z1du, x2du, y2du, z2du);
                atten += (l1 - l2) * coeffTot;

                float eff = expf(-atten);
                float ext = 1000.0f, inv_d = 1.0f / d_sub;
                float xt = xDs + ext*(xDs-xIs)*inv_d;
                float yt = yDs + ext*(yDs-yIs)*inv_d;
                float zt = zDs + ext*(zDs-zIs)*inv_d;
                float la = length_box_ray(xIs, yIs, zIs, xt, yt, zt,
                    x1du, y1du, z1du, x2du, y2du, z2du);
                float absFactor = (1.0f - expf(-la*coeffTot)) * coeffPE / coeffTot;

                final_val += eff * solid_angle * absFactor;
            }
        }
    }

    dst[tid] = final_val;
}


// ============================================================================
// Host: build spatial hash + launch
// ============================================================================
int PESysMatGen(float* parameter_Collimator, float* parameter_Detector,
                float* parameter_Image, float* dst, int cuda_id)
{
    cout << "Get into PESysMatGen (Optimized v3 — fixed maxLateral)" << endl;

    int nVX = (int)parameter_Image[0];
    int nVY = (int)parameter_Image[1];
    int nVZ = (int)parameter_Image[2];

    size_t numProjSingle = (size_t)floorf(parameter_Detector[0] + 0.0001f);
    size_t numImagebin = (size_t)nVX * (size_t)nVY * (size_t)nVZ;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int d = 0; d < deviceCount; d++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        printf("Device %d: %s (SM %d.%d)\n", d, prop.name, prop.major, prop.minor);
    }
    cudaSetDevice(cuda_id);

    // ---- Build spatial hash ----
    int nLayers = (int)floorf(parameter_Collimator[0] + 0.001f);
    int totalHoles = 0;
    for (int l = 0; l < nLayers; l++)
        totalHoles += (int)parameter_Collimator[(l+1)*10];

    float colW = parameter_Collimator[11];
    float colH = parameter_Collimator[13];
    float maxR = 0.0f;
    for (int h = 0; h < totalHoles; h++) {
        float r = parameter_Collimator[h*9 + 104];
        if (r > maxR) maxR = r;
    }

    float cellSize = fmaxf(2.0f*maxR + 0.5f, 2.0f);
    float originX = -colW/2.0f - cellSize;
    float originZ = -colH/2.0f - cellSize;
    int gridNX = (int)ceilf((colW + 2*cellSize) / cellSize) + 1;
    int gridNZ = (int)ceilf((colH + 2*cellSize) / cellSize) + 1;
    int maxPerCell = 16;

    printf("Grid: %dx%d, cell=%.1fmm, holes=%d\n", gridNX, gridNZ, cellSize, totalHoles);

    int gridSize = gridNX * gridNZ;
    int* hIdx = new int[gridSize * maxPerCell]();
    int* hCnt = new int[gridSize]();

    for (int h = 0; h < totalHoles; h++) {
        float hx = parameter_Collimator[h*9+100];
        float hz = parameter_Collimator[h*9+103];
        float hr = parameter_Collimator[h*9+104];
        int cx0 = (int)floorf((hx-hr-originX)/cellSize);
        int cx1 = (int)floorf((hx+hr-originX)/cellSize);
        int cz0 = (int)floorf((hz-hr-originZ)/cellSize);
        int cz1 = (int)floorf((hz+hr-originZ)/cellSize);
        for (int cx = cx0; cx <= cx1; cx++)
            for (int cz = cz0; cz <= cz1; cz++) {
                if (cx<0||cx>=gridNX||cz<0||cz>=gridNZ) continue;
                int cid = cx*gridNZ + cz;
                int c = hCnt[cid];
                if (c < maxPerCell) { hIdx[cid*maxPerCell+c] = h; hCnt[cid] = c+1; }
            }
    }

    // ---- GPU alloc ----
    size_t matBytes = sizeof(float) * numProjSingle * numImagebin;
    float *devMat, *devCol, *devDet, *devImg;
    int *d_gIdx, *d_gCnt;

    cudaMalloc(&devMat, matBytes);
    cudaMemset(devMat, 0, matBytes);
    cudaMalloc(&devCol, 80000*sizeof(float));
    cudaMemcpy(devCol, parameter_Collimator, 80000*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&devDet, 80000*sizeof(float));
    cudaMemcpy(devDet, parameter_Detector, 80000*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&devImg, 100*sizeof(float));
    cudaMemcpy(devImg, parameter_Image, 100*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_gIdx, sizeof(int)*gridSize*maxPerCell);
    cudaMemcpy(d_gIdx, hIdx, sizeof(int)*gridSize*maxPerCell, cudaMemcpyHostToDevice);
    cudaMalloc(&d_gCnt, sizeof(int)*gridSize);
    cudaMemcpy(d_gCnt, hCnt, sizeof(int)*gridSize, cudaMemcpyHostToDevice);

    long long total = (long long)numProjSingle * (long long)numImagebin;
    int tpb = 256;
    long long nBlk = (total + tpb - 1) / tpb;
    if (nBlk > 2147483647LL) { cerr << "Grid overflow" << endl; return -1; }

    printf("Launch: %lld threads\n", total);

    photodetectorCudaMe<<<(int)nBlk, tpb>>>(
        devMat, devCol, devDet, devImg,
        d_gIdx, d_gCnt,
        originX, originZ, cellSize, gridNX, gridNZ, maxPerCell,
        (long long)numProjSingle, (long long)numImagebin);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "Launch: " << cudaGetErrorString(err) << endl; return -1; }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "Exec: " << cudaGetErrorString(err) << endl; return -1; }

    cudaMemcpy(dst, devMat, matBytes, cudaMemcpyDeviceToHost);
    cudaFree(devCol); cudaFree(devDet); cudaFree(devImg);
    cudaFree(devMat); cudaFree(d_gIdx); cudaFree(d_gCnt);
    delete[] hIdx; delete[] hCnt;

    return (int)numImagebin;
}
