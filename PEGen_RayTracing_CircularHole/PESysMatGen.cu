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
#include <vector>
#include <algorithm>
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
// Limits of the per-thread lists (beyond them the kernel scans everything: slower, never wrong)
// ============================================================================
#define PE_MAX_LAYERS 10        // collimator layers (the parameter format holds at most 9)
#ifndef PE_MAX_HOLES
#define PE_MAX_HOLES 256        // holes a voxel -> crystal cone can cross
#endif
#ifndef PE_MAX_CROSS_DET
#define PE_MAX_CROSS_DET 256    // other crystals a voxel -> crystal cone can cross
#endif

// Can a ray from the voxel to crystal idxDet pass through crystal id?  (cones around the two
// voxel -> crystal-centre directions with the half-angles of the crystals' bounding spheres overlap)
__device__ inline bool peCrossCandidate(const float* devDet, int id, int idxDet, float fov2col,
                                        float xImage, float yImage, float zImage,
                                        float xDir, float yDir, float zDir, float crit_self)
{
    if (id == idxDet) return false;
    float ow = devDet[12*id+4], ot = devDet[12*id+5], oh = devDet[12*id+6];
    if (devDet[12*id+7] <= 0.01f) return false;
    float ox = devDet[12*id+1], oy = devDet[12*id+2]+fov2col, oz = devDet[12*id+3];
    float od = sqrtf((xImage-ox)*(xImage-ox)+(yImage-oy)*(yImage-oy)+(zImage-oz)*(zImage-oz));
    float crit_o = sqrtf(ow*ow+oh*oh+ot*ot) / od / 2.0f;
    float dxo = (ox-xImage)/od, dyo = (oy-yImage)/od, dzo = (oz-zImage)/od;
    float ad = sqrtf((dxo-xDir)*(dxo-xDir)+(dyo-yDir)*(dyo-yDir)+(dzo-zDir)*(dzo-zDir));
    return ad <= crit_o + crit_self;
}

// Cheap exact rejection for one ray (voxel -> detector sub-element): holes are cylinders along y and
// crystals are boxes rotated about y, so each lies inside a vertical prism of radius r around (cx, cz)
// over [ya, yb].  If the x-z track of the ray over [ya, yb] never comes within r of the axis, the
// intersection length is exactly 0 and the full intersection function can be skipped.
// Valid for rays with dy = yDet - yImage > 0 (all rays of the kernel; others are not tested).
__device__ inline bool peRayNearAxis(float xI, float yI, float zI, float dxdy, float dzdy,
                                     float ya, float yb, float cx, float cz, float r)
{
    float ax = xI + dxdy * (ya - yI) - cx, az = zI + dzdy * (ya - yI) - cz;   // track at y = ya
    float ex = dxdy * (yb - ya), ez = dzdy * (yb - ya);                       // ... to y = yb
    float ee = ex*ex + ez*ez;
    float t = ee > 0.0f ? fminf(fmaxf(-(ax*ex + az*ez) / ee, 0.0f), 1.0f) : 0.0f;
    float qx = ax + t*ex, qz = az + t*ez;
    float rr = r * 1.0001f + 1e-4f;                                           // margin for rounding
    return qx*qx + qz*qz <= rr*rr;
}

// Spatial hash of the collimator holes (host): every hole is listed in all cells its disk overlaps;
// cells hold as many holes as needed (no truncation).
struct PEHoleHash {
    std::vector<int> idx, cnt;
    float originX, originZ, cellSize, maxR;
    int nx, nz, maxPerCell, totalHoles;
};
static PEHoleHash pe_buildHoleHash(const float* col)
{
    PEHoleHash g;
    int nLayers = (int)floorf(col[0] + 0.001f);
    g.totalHoles = 0;
    for (int l = 0; l < nLayers; l++) g.totalHoles += (int)col[(l+1)*10];
    float colW = col[11], colH = col[13];
    g.maxR = 0.0f;
    for (int h = 0; h < g.totalHoles; h++) g.maxR = fmaxf(g.maxR, col[h*9 + 104]);
    g.cellSize = fmaxf(2.0f*g.maxR + 0.5f, 2.0f);
    g.originX = -colW/2.0f - g.cellSize;
    g.originZ = -colH/2.0f - g.cellSize;
    g.nx = (int)ceilf((colW + 2*g.cellSize) / g.cellSize) + 1;
    g.nz = (int)ceilf((colH + 2*g.cellSize) / g.cellSize) + 1;
    g.cnt.assign((size_t)g.nx * g.nz, 0);
    for (int pass = 0; pass < 2; pass++) {               // pass 0 counts, pass 1 fills
        if (pass == 1) {
            g.maxPerCell = 1;
            for (int c : g.cnt) g.maxPerCell = c > g.maxPerCell ? c : g.maxPerCell;
            g.idx.assign((size_t)g.nx * g.nz * g.maxPerCell, 0);
            std::fill(g.cnt.begin(), g.cnt.end(), 0);
        }
        for (int h = 0; h < g.totalHoles; h++) {
            float hx = col[h*9+100], hz = col[h*9+103], hr = col[h*9+104];
            int cx0 = (int)floorf((hx-hr-g.originX)/g.cellSize), cx1 = (int)floorf((hx+hr-g.originX)/g.cellSize);
            int cz0 = (int)floorf((hz-hr-g.originZ)/g.cellSize), cz1 = (int)floorf((hz+hr-g.originZ)/g.cellSize);
            for (int cx = cx0; cx <= cx1; cx++)
                for (int cz = cz0; cz <= cz1; cz++) {
                    if (cx < 0 || cx >= g.nx || cz < 0 || cz >= g.nz) continue;
                    int cid = cx*g.nz + cz;
                    if (pass == 1) g.idx[(size_t)cid*g.maxPerCell + g.cnt[cid]] = h;
                    g.cnt[cid]++;
                }
        }
    }
    return g;
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
    float maxHoleR,
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

    int numColLayers = min((int)floorf(devCol[0] + 0.000001f), PE_MAX_LAYERS);
    int numDetBins = (int)floorf(devDet[0] + 0.000001f);

    float colHalfX[PE_MAX_LAYERS], colY1[PE_MAX_LAYERS], colY2[PE_MAX_LAYERS], colHalfZ[PE_MAX_LAYERS], colThick[PE_MAX_LAYERS], colCoeff[PE_MAX_LAYERS];
    int colNHoles[PE_MAX_LAYERS];
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
    // At height y every ray from the voxel to a point of the crystal lies inside the lateral box spanned by
    // the rays to the crystal's bounding sphere.  For each layer the box over the layer's thickness is
    // searched, so no hole that a ray can cross is missed.  If more than PE_MAX_HOLES holes qualify, the
    // sub-element loop falls back to all holes (slower, never wrong).
    float detDiag = sqrtf(wDet*wDet + hDet*hDet + tDet*tDet);
    float detR = 0.5f * detDiag;
    int holeIds[PE_MAX_HOLES];
    int holeLayers[PE_MAX_HOLES];
    int nHoles = 0;
    bool holeOverflow = false;
    int totalHoles = 0;
    for (int m = 0, h0 = 0; m < numColLayers && !holeOverflow; h0 += colNHoles[m], m++) {
        totalHoles += colNHoles[m];
        if (colNHoles[m] <= 0) continue;
        float bx0 = 1e30f, bx1 = -1e30f, bz0 = 1e30f, bz1 = -1e30f;
        for (int a = 0; a < 2; a++) {                       // front / back face of the layer
            float yP = a ? colY2[m] : colY1[m];
            for (int b = 0; b < 2; b++) {                   // nearest / farthest y of the crystal
                float tt = (yP - yImage) / fmaxf(yDC + (b ? detR : -detR) - yImage, 1e-3f);
                for (int c = 0; c < 2; c++) {               // lateral extremes of the crystal
                    float xP = xImage + (xDC + (c ? detR : -detR) - xImage) * tt;
                    float zP = zImage + (zDC + (c ? detR : -detR) - zImage) * tt;
                    bx0 = fminf(bx0, xP); bx1 = fmaxf(bx1, xP);
                    bz0 = fminf(bz0, zP); bz1 = fmaxf(bz1, zP);
                }
            }
        }
        int cx_min = max(0, (int)floorf((bx0 - grid_ox) / grid_cell));
        int cx_max = min(grid_nx-1, (int)floorf((bx1 - grid_ox) / grid_cell));
        int cz_min = max(0, (int)floorf((bz0 - grid_oz) / grid_cell));
        int cz_max = min(grid_nz-1, (int)floorf((bz1 - grid_oz) / grid_cell));
        for (int gx = cx_min; gx <= cx_max && !holeOverflow; gx++) {
            for (int gz = cz_min; gz <= cz_max && !holeOverflow; gz++) {
                int cid = gx * grid_nz + gz;
                int cnt = gridCnt[cid];
                int base = cid * grid_mpc;
                for (int h = 0; h < cnt; h++) {
                    int hid = gridIdx[base + h];
                    if (hid < h0 || hid >= h0 + colNHoles[m]) continue;          // other layer
                    float hx = devCol[hid*9+100], hz = devCol[hid*9+103], hr = devCol[hid*9+104];
                    if (hx + hr < bx0 || hx - hr > bx1 || hz + hr < bz0 || hz - hr > bz1) continue;
                    bool dup = false;
                    for (int k = 0; k < nHoles; k++) {
                        if (holeIds[k] == hid) { dup = true; break; }
                    }
                    if (dup) continue;
                    if (nHoles == PE_MAX_HOLES) { holeOverflow = true; break; }
                    holeIds[nHoles] = hid;
                    holeLayers[nHoles] = m;
                    nHoles++;
                }
            }
        }
    }

    // ======== PRE-COLLECT CROSS-DETECTORS ========
    // crystals whose bounding cone overlaps the cone voxel -> this crystal; beyond PE_MAX_CROSS_DET the
    // sub-element loop tests all crystals (slower, never wrong)
    float dist_vd = sqrtf((xDC-xImage)*(xDC-xImage)+(yDC-yImage)*(yDC-yImage)+(zDC-zImage)*(zDC-zImage));
    float crit_self = detDiag / dist_vd / 2.0f;
    float xDir = (xDC-xImage)/dist_vd;
    float yDir = (yDC-yImage)/dist_vd;
    float zDir = (zDC-zImage)/dist_vd;

    int crossDetIds[PE_MAX_CROSS_DET];
    int nCrossDet = 0;
    bool crossOverflow = false;
    for (int id = 0; id < numDetBins && !crossOverflow; id++) {
        if (!peCrossCandidate(devDet, id, (int)idxDet, fov2col, xImage, yImage, zImage, xDir, yDir, zDir, crit_self)) continue;
        if (nCrossDet == PE_MAX_CROSS_DET) { crossOverflow = true; break; }
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
                // slopes of the ray in x and z per unit y, for the prism pre-tests (dy > 0 for every ray
                // that reaches the crystal through the collimator; otherwise the pre-tests are skipped)
                float dyRay = yDet - yImage;
                bool preTest = dyRay > 1e-3f;
                float dxdy = preTest ? (xDet - xImage) / dyRay : 0.0f;
                float dzdy = preTest ? (zDet - zImage) / dyRay : 0.0f;

                for (int m = 0; m < numColLayers; m++) {
                    float tl = length_box_ray(xImage, yImage, zImage, xDet, yDet, zDet,
                                              -colHalfX[m], colY1[m], -colHalfZ[m],
                                               colHalfX[m], colY2[m],  colHalfZ[m]);
                    if (tl < 0.1f) { blocked = true; break; }
                }
                if (blocked) continue;

                // Holes — from pre-collected deduplicated list
                float holeLen[PE_MAX_LAYERS];
                for (int m = 0; m < numColLayers; m++) holeLen[m] = 0.0f;
                int nScan = holeOverflow ? totalHoles : nHoles;
                for (int ii = 0, layerEnd = numColLayers > 0 ? colNHoles[0] : 0, layer = 0; ii < nScan; ii++) {
                    int hid = ii, hl;
                    if (holeOverflow) {                     // all holes, layer by running count
                        while (layer < numColLayers - 1 && hid >= layerEnd) layerEnd += colNHoles[++layer];
                        hl = layer;
                    } else { hid = holeIds[ii]; hl = holeLayers[ii]; }
                    float hx = devCol[hid*9+100], hy1 = devCol[hid*9+101]+fov2col, hy2 = devCol[hid*9+102]+fov2col;
                    float hz = devCol[hid*9+103], hr = devCol[hid*9+104];
                    if (preTest && !peRayNearAxis(xImage, yImage, zImage, dxdy, dzdy, fminf(hy1, hy2), fmaxf(hy1, hy2), hx, hz, hr))
                        continue;                                   // the ray misses this hole: length 0
                    float tmp = length_cylinder_ray(xImage, yImage, zImage, xDet, yDet, zDet, hx, hy1, hy2, hz, hr);
                    holeLen[hl] += tmp;
                    atten += devCol[hid*9+105] * tmp;
                }
                for (int m = 0; m < numColLayers; m++) {
                    float LiC = colThick[m] / COSangle;
                    if ((LiC - holeLen[m]) >= 0.00001f)
                        atten += colCoeff[m] * (LiC - holeLen[m]);
                }

                if (atten > 30.0f) continue;

                // Other crystals — from pre-collected list
                for (int ii = 0; ii < (crossOverflow ? numDetBins : nCrossDet); ii++) {
                    int oid = ii;
                    if (!crossOverflow) oid = crossDetIds[ii];
                    else if (!peCrossCandidate(devDet, oid, (int)idxDet, fov2col, xImage, yImage, zImage, xDir, yDir, zDir, crit_self)) continue;
                    float ox = devDet[12*oid+1], oy = devDet[12*oid+2]+fov2col, oz = devDet[12*oid+3];
                    float ow = devDet[12*oid+4], ot = devDet[12*oid+5], oh = devDet[12*oid+6];
                    if (preTest && !peRayNearAxis(xImage, yImage, zImage, dxdy, dzdy, oy - 0.5f*ot, oy + 0.5f*ot,
                                                  ox, oz, 0.5f * sqrtf(ow*ow + oh*oh)))
                        continue;                                   // the ray misses this crystal: length 0
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
    cout << "Get into PESysMatGen (spatial hash of the holes, per-layer search)" << endl;

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
    if (cudaSetDevice(cuda_id) != cudaSuccess) { cerr << "cannot use CUDA device " << cuda_id << endl; return -1; }

    // ---- Build spatial hash ----
    PEHoleHash g = pe_buildHoleHash(parameter_Collimator);
    printf("Grid: %dx%d, cell=%.1fmm, holes=%d, max %d holes per cell\n", g.nx, g.nz, g.cellSize, g.totalHoles, g.maxPerCell);
    int gridSize = g.nx * g.nz;

    // ---- GPU alloc ----
    size_t matBytes = sizeof(float) * numProjSingle * numImagebin;
    float *devMat = 0, *devCol = 0, *devDet = 0, *devImg = 0;
    int *d_gIdx = 0, *d_gCnt = 0;
    int ret = -1;
    cudaError_t err;
#define PE_CHECK(call) do { err = (call); if (err != cudaSuccess) { cerr << "CUDA error in " #call ": " << cudaGetErrorString(err) << endl; goto cleanup; } } while (0)
    {
        PE_CHECK(cudaMalloc(&devMat, matBytes));
        PE_CHECK(cudaMemset(devMat, 0, matBytes));
        PE_CHECK(cudaMalloc(&devCol, 80000*sizeof(float)));
        PE_CHECK(cudaMemcpy(devCol, parameter_Collimator, 80000*sizeof(float), cudaMemcpyHostToDevice));
        PE_CHECK(cudaMalloc(&devDet, 80000*sizeof(float)));
        PE_CHECK(cudaMemcpy(devDet, parameter_Detector, 80000*sizeof(float), cudaMemcpyHostToDevice));
        PE_CHECK(cudaMalloc(&devImg, 100*sizeof(float)));
        PE_CHECK(cudaMemcpy(devImg, parameter_Image, 100*sizeof(float), cudaMemcpyHostToDevice));
        PE_CHECK(cudaMalloc(&d_gIdx, sizeof(int)*g.idx.size()));
        PE_CHECK(cudaMemcpy(d_gIdx, g.idx.data(), sizeof(int)*g.idx.size(), cudaMemcpyHostToDevice));
        PE_CHECK(cudaMalloc(&d_gCnt, sizeof(int)*gridSize));
        PE_CHECK(cudaMemcpy(d_gCnt, g.cnt.data(), sizeof(int)*gridSize, cudaMemcpyHostToDevice));

        long long total = (long long)numProjSingle * (long long)numImagebin;
        int tpb = 256;
        long long nBlk = (total + tpb - 1) / tpb;
        if (nBlk > 2147483647LL) { cerr << "Grid overflow" << endl; goto cleanup; }

        printf("Launch: %lld threads\n", total);
        photodetectorCudaMe<<<(int)nBlk, tpb>>>(
            devMat, devCol, devDet, devImg,
            d_gIdx, d_gCnt,
            g.originX, g.originZ, g.cellSize, g.nx, g.nz, g.maxPerCell, g.maxR,
            (long long)numProjSingle, (long long)numImagebin);
        PE_CHECK(cudaGetLastError());
        PE_CHECK(cudaDeviceSynchronize());
        PE_CHECK(cudaMemcpy(dst, devMat, matBytes, cudaMemcpyDeviceToHost));
        ret = (int)numImagebin;
    }
cleanup:
#undef PE_CHECK
    cudaFree(devCol); cudaFree(devDet); cudaFree(devImg);
    cudaFree(devMat); cudaFree(d_gIdx); cudaFree(d_gCnt);
    return ret;
}
