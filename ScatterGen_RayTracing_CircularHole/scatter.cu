// Generate System Matrix on GPU with primary compton scatter
// author: xingchun zheng @ tsinghua university
// last modified: 2024/12/21
// version: 1.0

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <chrono> 
#include <algorithm>
#include <cstring>

#include "scatter.h"
using namespace std;


#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define max(a,b) ((a>=b)?a:b) 
#define min(a,b) ((a<=b)?a:b)

__device__ float length_box_ray(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x1_box, float y1_box, float z1_box, float x2_box, float y2_box, float z2_box)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Down box position: (x1_box, y1_box, z1_box)
	// Right-Up box position: (x2_box, y2_box, z2_box)
	float eps = 0.001;

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps & fabs(z_out - z_in) < eps)
	{
		return 0.000;
	}

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps)
	{
		if ((x_in >= x1_box & x_in <= x2_box) & (y_in >= y1_box & y_in <= y2_box))
			if ((z_in <= z1_box & z_out >= z2_box) || (z_out <= z1_box & z_in >= z2_box))
				return fabs(z2_box - z1_box);
			else
				return 0.000;
		else
			return 0.000;
	}

	if (fabs(z_out - z_in) < eps & fabs(y_out - y_in) < eps)
	{
		if ((z_in >= z1_box & z_in <= z2_box) & (y_in >= y1_box & y_in <= y2_box))
			if ((x_in <= x1_box & x_out >= x2_box) || (x_out <= x1_box & x_in >= x2_box))
				return fabs(x2_box -x1_box);
			else
				return 0.000;
		else
			return 0.000;
	}

	if (fabs(x_out - x_in) < eps & fabs(z_out - z_in) < eps)
	{
		if ((x_in >= x1_box & x_in <= x2_box) & (z_in >= z1_box & z_in <= z2_box))
		{
			if ((y_in <= y1_box & y_out >= y2_box) || (y_out <= y1_box & y_in >= y2_box))
				return fabs(y2_box - y1_box);
			else
				return 0.000;
		}
		else
			return 0.000;
	}


	if (fabs(x_out - x_in) < eps & (x_in >= x2_box || x_in <= x1_box))
	{
		return 0.000;
	}
	else if (fabs(x_out - x_in) < eps)
	{
		float tmin, tmax, tzmin, tzmax, t_inout;
		t_inout = sqrt((y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));


		float inv_direction_y = t_inout / (y_out - y_in);
		float inv_direction_z = t_inout / (z_out - z_in);

		if (inv_direction_y < 0)
		{
			tmin = (y2_box - y_in) * inv_direction_y;
			tmax = (y1_box - y_in) * inv_direction_y;
		}
		else
		{
			tmax = (y2_box - y_in) * inv_direction_y;
			tmin = (y1_box - y_in) * inv_direction_y;
		}


		if (inv_direction_z < 0)
		{
			tzmin = (z2_box - z_in) * inv_direction_z;
			tzmax = (z1_box - z_in) * inv_direction_z;
		}
		else
		{
			tzmax = (z2_box - z_in) * inv_direction_z;
			tzmin = (z1_box - z_in) * inv_direction_z;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return 0.0;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;

		if ((tmax - tmin) < eps)
			return 0.000;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.000;
		else if (tmin <= eps || tmax <= eps)
			return 0.000;
		else
			return (tmax - tmin);

	}

	if (fabs(y_out - y_in) < eps & (y_in >= y2_box || y_in <= y1_box))
	{
		return 0.000;
	}
	else if (fabs(y_out - y_in) < eps)
	{
		float tmin, tmax, tzmin, tzmax, t_inout;
		t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (z_out - z_in) * (z_out - z_in));


		float inv_direction_x = t_inout / (x_out - x_in);
		float inv_direction_z = t_inout / (z_out - z_in);

		if (inv_direction_x < 0)
		{
			tmin = (x2_box - x_in) * inv_direction_x;
			tmax = (x1_box - x_in) * inv_direction_x;
		}
		else
		{
			tmax = (x2_box - x_in) * inv_direction_x;
			tmin = (x1_box - x_in) * inv_direction_x;
		}


		if (inv_direction_z < 0)
		{
			tzmin = (z2_box - z_in) * inv_direction_z;
			tzmax = (z1_box - z_in) * inv_direction_z;
		}
		else
		{
			tzmax = (z2_box - z_in) * inv_direction_z;
			tzmin = (z1_box - z_in) * inv_direction_z;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return 0.0;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;

		if ((tmax - tmin) < eps)
			return 0.000;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.000;
		else if (tmin <= eps || tmax <= eps)
			return 0.000;
		else
			return (tmax - tmin);
	}

	if (fabs(z_out - z_in) < eps & (z_in >= z2_box || z_in <= z1_box))
	{
		return 0.000;
	}
	else if (fabs(z_out - z_in) < eps)
	{
		float tmin, tmax, tymin, tymax, t_inout;
		t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in));

		float inv_direction_x = t_inout / (x_out - x_in);
		float inv_direction_y = t_inout / (y_out - y_in);


		if (inv_direction_x < 0)
		{
			tmin = (x2_box - x_in) * inv_direction_x;
			tmax = (x1_box - x_in) * inv_direction_x;
		}
		else
		{
			tmax = (x2_box - x_in) * inv_direction_x;
			tmin = (x1_box - x_in) * inv_direction_x;
		}


		if (inv_direction_y < 0)
		{
			tymin = (y2_box - y_in) * inv_direction_y;
			tymax = (y1_box - y_in) * inv_direction_y;
		}
		else
		{
			tymax = (y2_box - y_in) * inv_direction_y;
			tymin = (y1_box - y_in) * inv_direction_y;
		}

		if ((tmin > tymax) || (tymin > tmax))
			return 0.0;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;
		if ((tmax - tmin) < eps)
			return 0.000;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.000;
		else if (tmin <= eps || tmax <= eps)
			return 0.000;
		else
			return (tmax - tmin);
	}



	float tmin, tmax, tymin, tymax, tzmin, tzmax, t_inout;
	t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));

	float inv_direction_x = t_inout / (x_out - x_in);
	float inv_direction_y = t_inout / (y_out - y_in);
	float inv_direction_z = t_inout / (z_out - z_in);


	if (inv_direction_x < 0)
	{
		tmin = (x2_box - x_in) * inv_direction_x;
		tmax = (x1_box - x_in) * inv_direction_x;
	}
	else
	{
		tmax = (x2_box - x_in) * inv_direction_x;
		tmin = (x1_box - x_in) * inv_direction_x;
	}


	if (inv_direction_y < 0)
	{
		tymin = (y2_box - y_in) * inv_direction_y;
		tymax = (y1_box - y_in) * inv_direction_y;
	}
	else
	{
		tymax = (y2_box - y_in) * inv_direction_y;
		tymin = (y1_box - y_in) * inv_direction_y;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return 0.0;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	if (inv_direction_z < 0)
	{
		tzmin = (z2_box - z_in) * inv_direction_z;
		tzmax = (z1_box - z_in) * inv_direction_z;
	}
	else
	{
		tzmax = (z2_box - z_in) * inv_direction_z;
		tzmin = (z1_box - z_in) * inv_direction_z;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return 0.0;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;


	if ((tmax - tmin) < eps)
		return 0.000;
	else if (tmin >= t_inout || tmax >= t_inout)
		return 0.000;
	else if (tmin <= eps || tmax <= eps)
		return 0.000;
	else
		return (tmax - tmin);
	// post condition:
	// if tmin > tmax (in the code above this is represented by a return value of INFINITY)
	//     no intersection
	// else
	//     front intersection point = ray.origin + ray.direction * tmin (normally only this point matters)
	//     back intersection point  = ray.origin + ray.direction * tmax

}

__device__ float length_box_ray_inside(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x1_box, float y1_box, float z1_box, float x2_box, float y2_box, float z2_box)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Down box position: (x1_box, y1_box, z1_box)
	// Right-Up box position: (x2_box, y2_box, z2_box)
	// Outgoing position is inside the box!!!!
	if (fabs(y_out - y_in) < 0.001)
	{
		return 0.000;
	}

	float eps = 0.001;

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps & fabs(z_out - z_in) < eps)
	{
		return 0.000;
	}

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps)
	{
		if ((x_in >= x1_box & x_in <= x2_box) & (y_in >= y1_box & y_in <= y2_box))
		{
			if (z_in < z_out)
			{
				return (z_out - z1_box);
			}
			else if (z_in > z_out)
			{
				return (z2_box - z_out);
			}
		}
		else
			return 0.000;
	}

	if (fabs(z_out - z_in) < eps & fabs(y_out - y_in) < eps)
	{
		if ((z_in >= z1_box & z_in <= z2_box) & (y_in >= y1_box & y_in <= y2_box))
		{
			if (x_in < x_out)
			{
				return (x_out - x1_box);
			}
			else if (x_in > x_out)
			{
				return (x2_box - x_out);
			}
		}
		else
			return 0.000;
	}

	if (fabs(x_out - x_in) < eps & fabs(z_out - z_in) < eps)
	{
		if ((x_in >= x1_box & x_in <= x2_box) & (z_in >= z1_box & z_in <= z2_box))
		{
			if (y_in < y_out)
			{
				return (y_out - y1_box);
			}
			else if (y_in > y_out)
			{
				return (y2_box - y_out);
			}
		}
		else
			return 0.000;
	}


	if (fabs(x_out - x_in) < eps & (x_in >= x2_box || x_in <= x1_box))
	{
		return 0.000;
	}
	else if (fabs(x_out - x_in) < eps)
	{
		float tmin, tmax, tzmin, tzmax, t_inout;
		t_inout = sqrt((y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));


		float inv_direction_y = t_inout / (y_out - y_in);
		float inv_direction_z = t_inout / (z_out - z_in);

		if (inv_direction_y < 0)
		{
			tmin = (y2_box - y_in) * inv_direction_y;
			tmax = (y1_box - y_in) * inv_direction_y;
		}
		else
		{
			tmax = (y2_box - y_in) * inv_direction_y;
			tmin = (y1_box - y_in) * inv_direction_y;
		}


		if (inv_direction_z < 0)
		{
			tzmin = (z2_box - z_in) * inv_direction_z;
			tzmax = (z1_box - z_in) * inv_direction_z;
		}
		else
		{
			tzmax = (z2_box - z_in) * inv_direction_z;
			tzmin = (z1_box - z_in) * inv_direction_z;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return 0.0;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;



		if ((tmax - tmin) < eps)
			return 0.0;

		else if (tmin >= t_inout)
			return 0.0;

		else if (tmax <= eps)
			return 0.0;

		else if ((tmax >= t_inout) && (tmin > eps))
			return (t_inout - tmin);

		else if ((tmin <= eps) && (tmax <= t_inout))
			return tmax;

		else if ((tmin <= eps) && (tmax >= t_inout))
			return t_inout;

		else
			return (tmax - tmin);

	}

	if (fabs(y_out - y_in) < eps & (y_in >= y2_box || y_in <= y1_box))
	{
		return 0.000;
	}
	else if (fabs(y_out - y_in) < eps)
	{
		float tmin, tmax, tzmin, tzmax, t_inout;
		t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (z_out - z_in) * (z_out - z_in));


		float inv_direction_x = t_inout / (x_out - x_in);
		float inv_direction_z = t_inout / (z_out - z_in);

		if (inv_direction_x < 0)
		{
			tmin = (x2_box - x_in) * inv_direction_x;
			tmax = (x1_box - x_in) * inv_direction_x;
		}
		else
		{
			tmax = (x2_box - x_in) * inv_direction_x;
			tmin = (x1_box - x_in) * inv_direction_x;
		}


		if (inv_direction_z < 0)
		{
			tzmin = (z2_box - z_in) * inv_direction_z;
			tzmax = (z1_box - z_in) * inv_direction_z;
		}
		else
		{
			tzmax = (z2_box - z_in) * inv_direction_z;
			tzmin = (z1_box - z_in) * inv_direction_z;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return 0.0;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;


		if ((tmax - tmin) < eps)
			return 0.0;

		else if (tmin >= t_inout)
			return 0.0;

		else if (tmax <= eps)
			return 0.0;

		else if ((tmax >= t_inout) && (tmin > eps))
			return (t_inout - tmin);

		else if ((tmin <= eps) && (tmax <= t_inout))
			return tmax;

		else if ((tmin <= eps) && (tmax >= t_inout))
			return t_inout;

		else
			return (tmax - tmin);
	}

	if (fabs(z_out - z_in) < eps & (z_in >= z2_box || z_in <= z1_box))
	{
		return 0.000;
	}
	else if (fabs(z_out - z_in) < eps)
	{
		float tmin, tmax, tymin, tymax, t_inout;
		t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in));

		float inv_direction_x = t_inout / (x_out - x_in);
		float inv_direction_y = t_inout / (y_out - y_in);


		if (inv_direction_x < 0)
		{
			tmin = (x2_box - x_in) * inv_direction_x;
			tmax = (x1_box - x_in) * inv_direction_x;
		}
		else
		{
			tmax = (x2_box - x_in) * inv_direction_x;
			tmin = (x1_box - x_in) * inv_direction_x;
		}


		if (inv_direction_y < 0)
		{
			tymin = (y2_box - y_in) * inv_direction_y;
			tymax = (y1_box - y_in) * inv_direction_y;
		}
		else
		{
			tymax = (y2_box - y_in) * inv_direction_y;
			tymin = (y1_box - y_in) * inv_direction_y;
		}

		if ((tmin > tymax) || (tymin > tmax))
			return 0.0;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;


		if ((tmax - tmin) < eps)
			return 0.0;

		else if (tmin >= t_inout)
			return 0.0;

		else if (tmax <= eps)
			return 0.0;

		else if ((tmax >= t_inout) && (tmin > eps))
			return (t_inout - tmin);

		else if ((tmin <= eps) && (tmax <= t_inout))
			return tmax;

		else if ((tmin <= eps) && (tmax >= t_inout))
			return t_inout;

		else
			return (tmax - tmin);
	}






	float tmin, tmax, tymin, tymax, tzmin, tzmax, t_inout;
	t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));

	float inv_direction_x = t_inout / (x_out - x_in);
	float inv_direction_y = t_inout / (y_out - y_in);
	float inv_direction_z = t_inout / (z_out - z_in);


	if (inv_direction_x < 0)
	{
		tmin = (x2_box - x_in) * inv_direction_x;
		tmax = (x1_box - x_in) * inv_direction_x;
	}
	else
	{
		tmax = (x2_box - x_in) * inv_direction_x;
		tmin = (x1_box - x_in) * inv_direction_x;
	}


	if (inv_direction_y < 0)
	{
		tymin = (y2_box - y_in) * inv_direction_y;
		tymax = (y1_box - y_in) * inv_direction_y;
	}
	else
	{
		tymax = (y2_box - y_in) * inv_direction_y;
		tymin = (y1_box - y_in) * inv_direction_y;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return 0.0;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	if (inv_direction_z < 0)
	{
		tzmin = (z2_box - z_in) * inv_direction_z;
		tzmax = (z1_box - z_in) * inv_direction_z;
	}
	else
	{
		tzmax = (z2_box - z_in) * inv_direction_z;
		tzmin = (z1_box - z_in) * inv_direction_z;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return 0.0;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;


	if ((tmax - tmin) < eps)
		return 0.0;

	else if (tmin >= t_inout)
		return 0.0;

	else if (tmax <= eps)
		return 0.0;

	else if ((tmax >= t_inout) && (tmin > eps))
		return (t_inout - tmin);

	else if ((tmin <= eps) && (tmax <= t_inout))
		return tmax;

	else if ((tmin <= eps) && (tmax >= t_inout))
		return t_inout;

	else
		return (tmax - tmin);


	// post condition:
	// if tmin > tmax (in the code above this is represented by a return value of INFINITY)
	//     no intersection
	// else
	//     front intersection point = ray.origin + ray.direction * tmin (normally only this point matters)
	//     back intersection point  = ray.origin + ray.direction * tmax

}

__device__ float length_cylinder_ray(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x_cylinder, float y1_cylinder, float y2_cylinder, float z_cylinder, float radius)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Plane position: (y==y1)
	// Right-Plane position: (y==y2)
	// (x-x_cylinder)^2+(z-z_cylinder)^2=radius^2
	if (fabs(y1_cylinder - y2_cylinder) < 0.001)
	{
		return 0.000;
	}
	float t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));
	float k_x = (x_out - x_in) / t_inout;
	float k_y = (y_out - y_in) / t_inout;
	float k_z = (z_out - z_in) / t_inout;

	float x_leftPlane = x_in + k_x / k_y * (y1_cylinder - y_in);
	float x_rightPlane = x_in + k_x / k_y * (y2_cylinder - y_in);

	float z_leftPlane = z_in + k_z / k_y * (y1_cylinder - y_in);
	float z_rightPlane = z_in + k_z / k_y * (y2_cylinder - y_in);

	float tmin = (y1_cylinder - y_in) / k_y;
	float tmax = (y2_cylinder - y_in) / k_y;


	int flag_leftPlane = 0;
	int flag_rightPlane = 0;

	float t1 = 0;
	float t2 = 0;

	if ((x_leftPlane - x_cylinder) * (x_leftPlane - x_cylinder) + (z_leftPlane - z_cylinder) * (z_leftPlane - z_cylinder) <= radius * radius)
	{
		flag_leftPlane = 1;
	}
	if ((x_rightPlane - x_cylinder) * (x_rightPlane - x_cylinder) + (z_rightPlane - z_cylinder) * (z_rightPlane - z_cylinder) <= radius * radius)
	{
		flag_rightPlane = 1;
	}


	if ((flag_rightPlane == 1) && (flag_leftPlane == 1))
	{
		if (tmin <= 0.0001 || tmax <= 0.0001)
			return 0.0;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.0;
		else
			return fabs(tmax - tmin);
	}
	else
	{
		float x_ = x_in - x_cylinder;
		float z_ = z_in - z_cylinder;
		float Delta_ = (k_x * k_x + k_z * k_z) * radius * radius - (k_x * z_ - k_z * x_) * (k_x * z_ - k_z * x_);

		if (Delta_ <= 0.00001)
		{
			return 0.00000000;
		}
		else
		{
			t1 = (-(k_x * x_ + k_z * z_) - sqrt(Delta_)) / (k_x * k_x + k_z * k_z);
			t2 = (-(k_x * x_ + k_z * z_) + sqrt(Delta_)) / (k_x * k_x + k_z * k_z);
		}

		if ((flag_rightPlane == 0) && (flag_leftPlane == 0))
		{
			if ((t1 >= tmin) && (t2 <= tmax))
			{
				if (t2 <= 0.0001 || t1 <= 0.0001)
					return 0.0;
				else if (t2 >= t_inout || t1 >= t_inout)
					return 0.0;
				else
					return (t2 - t1);

			}
			else
			{
				return 0.000000;
			}

		}
		else if ((flag_leftPlane == 1) && (flag_rightPlane == 0))
		{
			if (t2 <= 0.0001 || tmin <= 0.0001)
				return 0.0;
			else if (t2 >= t_inout || tmin >= t_inout)
				return 0.0;
			else
				return (t2 - tmin);			

		}
		else if ((flag_leftPlane == 0) && (flag_rightPlane == 1))
		{
			if (tmax <= 0.0001 || t1 <= 0.0001)
				return 0.0;
			else if (tmax >= t_inout || t1 >= t_inout)
				return 0.0;
			else
				return (tmax - t1);
			
		}
		else
		{
			return 0.000000;
		}
	}

}

__device__ float length_ellipticalcylinder_ray(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x_cylinder, float y1_cylinder, float y2_cylinder, float z_cylinder, float x_rho, float z_rho, float theta)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Plane position: (y==y1)
	// Right-Plane position: (y==y2)
	// [(x-x_cylinder)/x_rho]^2+[(z-z_cylinder)/z_rho]^2=1
	// with rotation angle theta, defined inverse conunter-clock wise from x axis
	//                        ^ z
	//                        .
	//                        .
	//                        .
	//                        .
	//                        .        . x'
	//                        .       .
	//                        .      .
	//                        .     .
	//                        .    .
	//                        .   .
	//                        .  .
	//                        . .  theta
	//........................................................>x
	//
	//

	float x_in_ = x_in - x_cylinder;
	float z_in_ = z_in - z_cylinder;

	float x_out_ = x_out - x_cylinder;
	float z_out_ = z_out - z_cylinder;

	x_in = x_cylinder + x_in_ * cos(theta) + z_in_ * sin(theta);
	z_in = z_cylinder - x_in_ * sin(theta) + z_in_ * cos(theta);

	x_out = x_cylinder + x_out_ * cos(theta) + z_out_ * sin(theta);
	z_out = z_cylinder - x_out_ * sin(theta) + z_out_ * cos(theta);

	if (fabs(y1_cylinder - y2_cylinder) < 0.001)
	{
		return 0.000;
	}
	float t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));
	float k_x = (x_out - x_in) / t_inout;
	float k_y = (y_out - y_in) / t_inout;
	float k_z = (z_out - z_in) / t_inout;

	float x_leftPlane = x_in + k_x / k_y * (y1_cylinder - y_in);
	float x_rightPlane = x_in + k_x / k_y * (y2_cylinder - y_in);

	float z_leftPlane = z_in + k_z / k_y * (y1_cylinder - y_in);
	float z_rightPlane = z_in + k_z / k_y * (y2_cylinder - y_in);

	float tmin = (y1_cylinder - y_in) / k_y;
	float tmax = (y2_cylinder - y_in) / k_y;


	int flag_leftPlane = 0;
	int flag_rightPlane = 0;

	float t1 = 0;
	float t2 = 0;

	if (((x_leftPlane - x_cylinder) * (x_leftPlane - x_cylinder) / x_rho / x_rho + (z_leftPlane - z_cylinder) * (z_leftPlane - z_cylinder) / z_rho / z_rho) <= 1)
	{
		flag_leftPlane = 1;
	}
	if (((x_rightPlane - x_cylinder) * (x_rightPlane - x_cylinder) / x_rho / x_rho + (z_rightPlane - z_cylinder) * (z_rightPlane - z_cylinder) / z_rho / z_rho) <= 1)
	{
		flag_rightPlane = 1;
	}


	if ((flag_rightPlane == 1) && (flag_leftPlane == 1))
	{
		if (tmin <= 0.0001 || tmax <= 0.0001)
			return 0.0;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.0;
		else
			return fabs(tmax - tmin);
	}
	else
	{
		float x_ = x_in - x_cylinder;
		float z_ = z_in - z_cylinder;

		float a = (k_x * k_x * z_rho * z_rho + k_z * k_z * x_rho * x_rho);
		float b = k_x * x_ * z_rho * z_rho + k_z * z_ * x_rho * x_rho;
		float c = z_rho * z_rho * x_ * x_ + x_rho * x_rho * z_ * z_ - x_rho * x_rho * z_rho * z_rho;

		float Delta_ = (b * b - a * c);

		if (Delta_ <= 0.00001)
		{
			return 0.00000000;
		}
		else
		{
			t1 = (-b - sqrt(Delta_)) / a;
			t2 = (-b + sqrt(Delta_)) / a;
		}

		if ((flag_rightPlane == 0) && (flag_leftPlane == 0))
		{
			if ((t1 >= tmin) && (t2 <= tmax))
			{
				if (t2 <= 0.0001 || t1 <= 0.0001)
					return 0.0;
				else if (t2 >= t_inout || t1 >= t_inout)
					return 0.0;
				else
					return (t2 - t1);

			}
			else
			{
				return 0.000000;
			}

		}
		else if ((flag_leftPlane == 1) && (flag_rightPlane == 0))
		{
			if (t2 <= 0.0001 || tmin <= 0.0001)
				return 0.0;
			else if (t2 >= t_inout || tmin >= t_inout)
				return 0.0;
			else
				return (t2 - tmin);

		}
		else if ((flag_leftPlane == 0) && (flag_rightPlane == 1))
		{
			if (tmax <= 0.0001 || t1 <= 0.0001)
				return 0.0;
			else if (tmax >= t_inout || t1 >= t_inout)
				return 0.0;
			else
				return (tmax - t1);

		}
		else
		{
			return 0.000000;
		}
	}

}

// Device function to compute the differential Compton cross section
__device__ float diffComptonSection(float theta, float E0)
{
	// Calculate the cosine of the angle, in [0,2pi]
	float cos_theta = cos(theta);

	// Calculate the normalized energy alpha
	float alpha = E0 / 511.0f;

	// Factor1 and Factor2 computations
	float factor1 = alpha * (1.0f - cos_theta);
	float factor2 = 1.0f + cos_theta * cos_theta;

	// Calculate the result
	float result = factor2 / (1.0f + factor1) / (1.0f + factor1) * (1.0f + factor1 * factor1 / factor2 / (1.0f + factor1));

	return result;
}

// Compute the differential cross section integral
__device__ float computeComptonIntegral(float E0, float theta_low, float theta_high, float DeltaTheta)
{

	// midpoint rule with n >= 1 equal steps of at most DeltaTheta over [theta_low, theta_high]
	// (a fixed step DeltaTheta from theta_low to theta_low + numSteps*DeltaTheta overshot narrow ranges)
	float total_cross_section = 0.0f;
	if (!(theta_high > theta_low)) return 0.0f;
	int numSteps = (int)ceilf((theta_high - theta_low) / DeltaTheta);
	if (numSteps < 1) numSteps = 1;
	float h = (theta_high - theta_low) / numSteps;

	for (int i = 0; i < numSteps; ++i) {
		float theta = theta_low + (i + 0.5f) * h;
		total_cross_section += diffComptonSection(theta, E0) * sin(theta) * h;
	}

	return total_cross_section;
	
}

// Calculate the cone angle theta
__device__ float calculateConeAngle(float xSource, float ySource, float zSource, float xA, float yA, float zA, float xDetector, float yDetector, float zDetector)
{
	// Source------> Point A (Compton Scatter) --------> Detector
	float vSourceA[3] = { xA - xSource, yA - ySource, zA - zSource };	
	float vAD[3] = { xDetector - xA, yDetector - yA, zDetector - zA };

	float dotProduct = vAD[0] * vSourceA[0] + vAD[1] * vSourceA[1] + vAD[2] * vSourceA[2];

	float magnitude_vSourceA = sqrt(vSourceA[0] * vSourceA[0] + vSourceA[1] * vSourceA[1] + vSourceA[2] * vSourceA[2]);
	float magnitude_vAD = sqrt(vAD[0] * vAD[0] + vAD[1] * vAD[1] + vAD[2] * vAD[2]);

	float cosTheta = dotProduct / (magnitude_vSourceA * magnitude_vAD);

	if (cosTheta > 1.0f) cosTheta = 1.0f;
	if (cosTheta < -1.0f) cosTheta = -1.0f;

	return acos(cosTheta);
}

// Calculate the energy of scattered photon
__device__ float calculateScatterEnergy(float theta, float E0)
{

	float cosTheta = cos(theta);
	float E_scatter = E0 / (1.0f + (E0 / 511.0f) * (1.0f - cosTheta));

	return E_scatter;
}

__device__ float calculategaussianIntegral(float scatterEnergy, float energy_resolution_scatterphoton, float lowerThresholdofEnergyWindow, float upperThresholdofEnergyWindow)
{
	float sigma = energy_resolution_scatterphoton / 2.35482f * scatterEnergy;
	float sqrt2sigma = sigma * sqrt(2.0);
	float z1 = (lowerThresholdofEnergyWindow - scatterEnergy) / sqrt2sigma;
	float z2 = (upperThresholdofEnergyWindow - scatterEnergy) / sqrt2sigma;
	float probability = 0.5f * (erf(z2) - erf(z1));
	return probability;
}

__device__ inline float calculateDist(float x1, float y1, float z1, float x2, float y2, float z2)
{
	return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

__device__ int indexFrombitmap_crystal(int i, int j, int k, unsigned int* d_bit_array, int numProjectionSingle, int bits_per_word)
{	
	if (i > j) 
	{
		int temp = i;
		i = j;
		j = temp;
	}
	
	long long pair_idx = static_cast<long long>(i) * numProjectionSingle - (static_cast<long long>(i) * (i + 1)) / 2 + j;
	long long bit_idx = pair_idx * numProjectionSingle + k;
	
	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	unsigned int word = d_bit_array[word_idx];
	int flag = 0;
	flag = (word >> bit_offset) & 1;
	return flag;
}


__device__ int indexFrombitmap_collimator(int i, int j, int k, unsigned int* d_bit_array, int numProjectionSingle, int bits_per_word)
{

	long long bit_idx = static_cast<long long>(i) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle) + static_cast<long long>(j) * static_cast<long long>(numProjectionSingle) + static_cast<long long>(k);

	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	unsigned int word = d_bit_array[word_idx];
	int flag = 0;
	flag = (word >> bit_offset) & 1;
	return flag;
}


__global__ void crystalScatterSysMatCuda(float* dst,
	float* deviceparameter_Detector,
	float* deviceparameter_Image,
	float* deviceparameter_Physics,
	float* devicePESysMat,
	unsigned int* deviceGeometryRelationShip_Crystal2Crystal,
	int numProjectionSingle,
	int numImagebin)

{
	// Calculate the primary compton scatter between crystals
	// Image---->Scatter crystal------>Detector crystal

	int numDetectorbins = numProjectionSingle;
	float _float_FOV2Collimator = deviceparameter_Image[11];
	
	//////////////////////////////////////////// Image Parameters ////////////////////////////////////////////

	float _float_widthImageVoxelX = deviceparameter_Image[3];
	float _float_widthImageVoxelY = deviceparameter_Image[4];
	float _float_widthImageVoxelZ = deviceparameter_Image[5];

	//float _float_numRotation = deviceparameter_Image[6];//numRotation;
	float _float_angelPerRotation = deviceparameter_Image[7];//Angel per Rotation;
	float _float_idxrotation = deviceparameter_Image[20];//idxRotation
	//float RotationAngle = _float_idxrotation / _float_numRotation * (2 * M_PI);
	float RotationAngle = _float_idxrotation * _float_angelPerRotation;
	float shiftFOVX_physics = deviceparameter_Image[8];
	float shiftFOVY_physics = deviceparameter_Image[9];
	float shiftFOVZ_physics = deviceparameter_Image[10];

	int numImageVoxelX = (int)floor(deviceparameter_Image[0]);
	int numImageVoxelY = (int)floor(deviceparameter_Image[1]);
	int numImageVoxelZ = (int)floor(deviceparameter_Image[2]);
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////// Threads Allocation ///////////////////////////////////////////

	// grid x runs over voxels, grid y over detectors (gridDim.y <= 65535 would limit the voxels)
	long long int row = blockIdx.y * (long long)blockDim.x + threadIdx.x;
	if (row < 0 || row > numProjectionSingle - 1) { return; }
	long long int col = blockIdx.x * (long long)blockDim.y + threadIdx.y;
	if (col < 0 || col > numImagebin - 1) { return; }
	long long int slice = blockIdx.z * blockDim.z + threadIdx.z;
	if (slice < 0 || slice > numProjectionSingle - 1) { return; }

	long long int dstIndex = row * numImagebin + col;

	unsigned int idxDetector = row; // index of detector
	unsigned int id_Detector = slice; // index of scatter
	if (idxDetector == id_Detector) 
	{
		return;
	}

	//Image Domain demensions=(z->y->x)
	int idxImageVoxelZ = col / (numImageVoxelY * numImageVoxelX);
	col = col % (numImageVoxelY * numImageVoxelX);
	int idxImageVoxelY = col / numImageVoxelX;
	int idxImageVoxelX = col % numImageVoxelX;


	// Finite Divisions of Detector Crystal
	const unsigned int divideX = 1, divideY = 1, divideZ = 1;


	///////////////////////////////////////// Physic Progress Parameters /////////////////////////////////////
	int flagUsingCompton = (int)floor(deviceparameter_Physics[0] + 0.5f);
	int flagUsingSameEnergyWindow = (int)floor(deviceparameter_Physics[4] + 0.5f);

	float lowerThresholdofEnergyWindow = deviceparameter_Physics[5];
	float upperThresholdofEnergyWindow = deviceparameter_Physics[6];

	float target_PE_Energy = deviceparameter_Physics[7];
	float energy_resolution_detector_targetPE = deviceparameter_Detector[idxDetector * 12 + 10];

	// Energy Window of detector crystal
	if (flagUsingSameEnergyWindow > 0)
	{
		lowerThresholdofEnergyWindow = deviceparameter_Physics[5];
		upperThresholdofEnergyWindow = deviceparameter_Physics[6];
	}
	else
	{
		lowerThresholdofEnergyWindow = (1 - energy_resolution_detector_targetPE / 2.0f) * target_PE_Energy;
		upperThresholdofEnergyWindow = (1 + energy_resolution_detector_targetPE / 2.0f) * target_PE_Energy;
	}

	float coeff_detector_total = deviceparameter_Detector[idxDetector * 12 + 7];
	float coeff_detector_pe = deviceparameter_Detector[idxDetector * 12 + 8];
	float coeff_detector_compton = deviceparameter_Detector[idxDetector * 12 + 9];

	float integration_Compton = computeComptonIntegral(target_PE_Energy, 0.0f, M_PI, 0.01);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////// Image Rotation Shift Parameters /////////////////////////////////////
	float xImage = (idxImageVoxelX - numImageVoxelX / 2.0f + 0.5f) * _float_widthImageVoxelX;
	float yImage = (idxImageVoxelY - numImageVoxelY / 2.0f + 0.5f) * _float_widthImageVoxelY;
	float zImage = (idxImageVoxelZ - numImageVoxelZ / 2.0f + 0.5f) * _float_widthImageVoxelZ;

	xImage = xImage + shiftFOVX_physics;
	yImage = yImage + shiftFOVY_physics;
	zImage = zImage + shiftFOVZ_physics;

	float xImage_rot = xImage * cos(RotationAngle) - yImage * sin(RotationAngle);
	float yImage_rot = xImage * sin(RotationAngle) + yImage * cos(RotationAngle);
	float zImage_rot = zImage;
	xImage = xImage_rot;
	yImage = yImage_rot;
	zImage = zImage_rot;

	int ImageVoxel_index = idxImageVoxelX + idxImageVoxelY * numImageVoxelX + idxImageVoxelZ * numImageVoxelY * numImageVoxelX;

	// All variables without a suffix are in the real-world physical coordinate system
	// All parameters with 'self' suffix are in the detector crystal coordinate system
	float xDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 1];
	float yDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 2] + _float_FOV2Collimator;
	float zDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 3];
	
	float widthDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 4];
	float heightDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 6];
	float thicknessDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 5];

	float rotationAngel_DetectorCrystal = deviceparameter_Detector[12 * idxDetector + 11];

	float xImage_self = (xImage - xDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal) - (zImage - zDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal);
	float yImage_self = yImage - yDetectorCrystalCenter;
	float zImage_self = (xImage - xDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal) + (zImage - zDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal);

	float x1_detectorcrystal_self = -widthDetectorCrystal / 2.0f;
	float x2_detectorcrystal_self = widthDetectorCrystal / 2.0f;
	float y1_detectorcrystal_self = -thicknessDetectorCrystal / 2.0f;
	float y2_detectorcrystal_self = thicknessDetectorCrystal / 2.0f;
	float z1_detectorcrystal_self = -heightDetectorCrystal / 2.0f;
	float z2_detectorcrystal_self = heightDetectorCrystal / 2.0f;
		
	//////////////////////////////////////////////////////////////////////////////////////////////////////////


	///////////////  Probability of Compton Scatter Happened on scatter crystal id_Detector //////////////////
	long long PESysMat_index = (long long)numImagebin * id_Detector + ImageVoxel_index;
	// Compton / photoelectric ratio of the scatter crystal id_Detector (not of the detecting crystal)
	float coeff_scatter_pe = deviceparameter_Detector[id_Detector * 12 + 8];
	float coeff_scatter_compton = deviceparameter_Detector[id_Detector * 12 + 9];
	float prob_Compton_othercrystal = coeff_scatter_pe > 0.0f ? devicePESysMat[PESysMat_index] * coeff_scatter_compton / coeff_scatter_pe : 0.0f;

	float x_scatter = deviceparameter_Detector[id_Detector * 12 + 1];
	float y_scatter = deviceparameter_Detector[id_Detector * 12 + 2] + _float_FOV2Collimator;
	float z_scatter = deviceparameter_Detector[id_Detector * 12 + 3];

	///////////////////////////////////////// CalCulation Starts Below ///////////////////////////////////////	
	if (flagUsingCompton == 1)
	{
		if (coeff_detector_total > 0.01f)
		{
			//float prob_Compton_to_detectionCrystal = 0.000;
			
			for (int NumZ = 0; NumZ < divideZ; NumZ++)
			{
				for (int NumX = 0; NumX < divideX; NumX++)
				{
					for (int NumY = 0; NumY < divideY; NumY++)
					{

						/////////////////////////////////  Parameters of the detector unit ////////////////////////////////
						// All variables without a suffix are in the real-world physical coordinate system
						// All parameters with 'self' suffix are in the detector crystal coordinate system
						float xDetector_self = -widthDetectorCrystal / 2.0f + (float)(NumX + 0.5f) / (float)divideX * widthDetectorCrystal;
						float zDetector_self = -heightDetectorCrystal / 2.0f + (float)(NumZ + 0.5f) / (float)divideZ * heightDetectorCrystal;
						float yDetector_self =  - thicknessDetectorCrystal / 2.0f + (float)(NumY + 0.5f) / (float)divideY * thicknessDetectorCrystal;

						float xDetector_rot = xDetector_self * cos(rotationAngel_DetectorCrystal) - zDetector_self * sin(rotationAngel_DetectorCrystal);
						float zDetector_rot = xDetector_self * sin(rotationAngel_DetectorCrystal) + zDetector_self * cos(rotationAngel_DetectorCrystal);
						float yDetector_rot = yDetector_self;

						float xDetector = xDetectorCrystalCenter + xDetector_rot;
						float zDetector = zDetectorCrystalCenter + zDetector_rot;
						float yDetector = yDetectorCrystalCenter + yDetector_rot;


						/////////////////////////////////  Compton scatter probability from the other crystal to detection crystal /////////////////////////////
						float length = 0;
						float comptonConeAngle = calculateConeAngle(xImage, yImage, zImage, x_scatter, y_scatter, z_scatter, xDetector, yDetector, zDetector);
						float scatterEnergy = calculateScatterEnergy(comptonConeAngle, target_PE_Energy);
						// Relative FWHM scales as R(E) = R0 * sqrt(E0 / E): absolute FWHM grows as sqrt(E), so lower-energy scattered photons have worse relative resolution
						float energy_resolution_detector_scatterphoton = energy_resolution_detector_targetPE * sqrt(target_PE_Energy / scatterEnergy);

						//  The probability that a Compton scatterred photon being detected within the energy window of detector unit

						if (((scatterEnergy * (1 + 2 * energy_resolution_detector_scatterphoton / 2.35482f)) <= lowerThresholdofEnergyWindow) || (scatterEnergy * (1 - 2 * energy_resolution_detector_scatterphoton / 2.35482f) >= upperThresholdofEnergyWindow))
						{
							continue; 
							// The energy of the scattered photon detected within a detector element follows a Gaussian distribution. 
							// If the 2 sigma range of this Gaussian does not overlap with the full energy peak window of the detector element, 
							// then it is considered that the scattering does not affect the result.
						}
						float energyDetected_probability = calculategaussianIntegral(scatterEnergy, energy_resolution_detector_scatterphoton, lowerThresholdofEnergyWindow, upperThresholdofEnergyWindow);
								
								
						// The probability that a Compton scattered photon, among all the photons scattered at the scattering point, 
						// is scattered towards the direction of the detector element.
						float L_comptonAngle = calculateDist(x_scatter, y_scatter, z_scatter, xDetector, yDetector, zDetector);

						// Calculate the phi range, using the detector unit's minimum enclosing sphere as an approximation.
						float R_detector = sqrt(widthDetectorCrystal * widthDetectorCrystal / (float)divideX / (float)divideX + heightDetectorCrystal * heightDetectorCrystal / (float)divideZ / (float)divideZ + thicknessDetectorCrystal * thicknessDetectorCrystal / (float)divideY / (float)divideY) / 2.0f;
						float Range_Phi = 0.000f;
						if (L_comptonAngle * sin(comptonConeAngle) * 2.0f <= R_detector)
						{
							Range_Phi = 2.0f * M_PI;
						}
						else
						{
							Range_Phi = 4.0f * asin(min(R_detector / 2.0f / L_comptonAngle / sin(comptonConeAngle), 1.0f));
						}
								
						// Calculate the theta range
						float x_scatter_self = (x_scatter - xDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal) - (z_scatter - zDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal);
						float y_scatter_self = y_scatter - yDetectorCrystalCenter;
						float z_scatter_self = (x_scatter - xDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal) + (z_scatter - zDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal);
								
						float x1_detectorunit_self = ((float)NumX / (float)divideX - 0.5f) * widthDetectorCrystal;
						float x2_detectorunit_self = (((float)NumX + 1.0f) / (float)divideX - 0.5f) * widthDetectorCrystal;

						float y1_detectorunit_self = ((float)NumY / (float)divideY - 0.5f) * thicknessDetectorCrystal;
						float y2_detectorunit_self = (((float)NumY + 1.0f) / (float)divideY - 0.5f) * thicknessDetectorCrystal;

						float z1_detectorunit_self = ((float)NumZ / (float)divideZ - 0.5f) * heightDetectorCrystal;
						float z2_detectorunit_self = (((float)NumZ + 1.0f) / (float)divideZ - 0.5f) * heightDetectorCrystal;


						float dist_extend = 1000.0f;
						float dist_Image_scatterer = calculateDist(x_scatter_self, y_scatter_self, z_scatter_self, xImage_self, yImage_self, zImage_self);
						float x_tmp = x_scatter_self + dist_extend * (x_scatter_self - xImage_self) / dist_Image_scatterer;
						float y_tmp = y_scatter_self + dist_extend * (y_scatter_self - yImage_self) / dist_Image_scatterer;
						float z_tmp = z_scatter_self + dist_extend * (z_scatter_self - zImage_self) / dist_Image_scatterer;

						length = length_box_ray(xImage_self, yImage_self, zImage_self, x_tmp, y_tmp, z_tmp, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);
									
						float theta[8];
						theta[0] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self);
						theta[1] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self);
						theta[2] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y2_detectorunit_self, z1_detectorunit_self);
						theta[3] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y1_detectorunit_self, z2_detectorunit_self);
						theta[4] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y2_detectorunit_self, z1_detectorunit_self);
						theta[5] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y1_detectorunit_self, z2_detectorunit_self);
						theta[6] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);
						theta[7] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						float theta_min = theta[0];
						float theta_max = theta[0];
						for (int i = 1; i < 8; i++)
						{
							if (theta[i] > theta_max)
								theta_max = theta[i];
							if (theta[i] < theta_min)
								theta_min = theta[i];
						}
									
						if (length > 0.001f) 
						{
							theta_min=0.000; // If the extension of the line from the image to the scatterer passes through the detector unit, then theta_min=0
						}
									
						// Range_Theta = 2.0f * asin(min(1.0f, R_detector / R_comptonAngle));
						float comptonAngleRatio = computeComptonIntegral(target_PE_Energy, theta_min, theta_max, 0.01) / integration_Compton;

						/////////////////////////////////  Attenuation from the scatterer crystal to detector unit /////////////////////////////
						float attenuation_dist_crystal_crystal = 0.000f;
								
						for (int id_Detector_att = 0; id_Detector_att < numDetectorbins; id_Detector_att++)
						{
							if ((id_Detector_att != idxDetector) && (id_Detector_att != id_Detector))
							{
								int bits_per_word = 32;
								int flagCross = indexFrombitmap_crystal(id_Detector, idxDetector, id_Detector_att, deviceGeometryRelationShip_Crystal2Crystal, numProjectionSingle, bits_per_word);
								if (flagCross == 0) 
								{ 
									continue; 
								}
								else 
								{	
									float length_att = 0;

									float x_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 1];
									float y_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 2] + _float_FOV2Collimator;
									float z_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 3];

									float width_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 4];
									float height_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 6];
									float thickness_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 5];

									float rotationAngel_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 11];

									float coeff_total_att = deviceparameter_Detector[12 * id_Detector_att + 7];
									float coeff_pe_att = deviceparameter_Detector[12 * id_Detector_att + 8];
									float coeff_compton_att = deviceparameter_Detector[12 * id_Detector_att + 9];

									// 100keV~1MeV,cross section of photon electric effect is propotional to E^-3;
									// 100keV~1MeV, cross section of compton effect is propotional to E^-1;
									coeff_pe_att = coeff_pe_att * target_PE_Energy * target_PE_Energy * target_PE_Energy / scatterEnergy / scatterEnergy / scatterEnergy;
									coeff_compton_att = coeff_compton_att * target_PE_Energy / scatterEnergy;
									coeff_total_att = coeff_pe_att + coeff_compton_att;

									float x_scatter_Att = (x_scatter - x_AttCrystalCenter) * cos(-rotationAngel_AttCrystal) - (z_scatter - z_AttCrystalCenter) * sin(-rotationAngel_AttCrystal);
									float y_scatter_Att = y_scatter - y_AttCrystalCenter;
									float z_scatter_Att = (x_scatter - x_AttCrystalCenter) * sin(-rotationAngel_AttCrystal) + (z_scatter - z_AttCrystalCenter) * cos(-rotationAngel_AttCrystal);

									float x1_Att = -0.5f * width_AttCrystal;
									float x2_Att = 0.5f * width_AttCrystal;

									float y1_Att = -0.5f * thickness_AttCrystal;
									float y2_Att = 0.5f * thickness_AttCrystal;

									float z1_Att = -0.5f * height_AttCrystal;
									float z2_Att = 0.5f * height_AttCrystal;

									float xDetector_Att = (xDetector - x_AttCrystalCenter) * cos(-rotationAngel_AttCrystal) - (zDetector - z_AttCrystalCenter) * sin(-rotationAngel_AttCrystal);
									float yDetector_Att = yDetector - y_AttCrystalCenter;
									float zDetector_Att = (xDetector - x_AttCrystalCenter) * sin(-rotationAngel_AttCrystal) + (zDetector - z_AttCrystalCenter) * cos(-rotationAngel_AttCrystal);

									length_att = length_box_ray(x_scatter_Att, y_scatter_Att, z_scatter_Att, xDetector_Att, yDetector_Att, zDetector_Att, x1_Att, y1_Att, z1_Att, x2_Att, y2_Att, z2_Att);
									attenuation_dist_crystal_crystal = attenuation_dist_crystal_crystal + length_att * coeff_total_att;

								}

							}

						}


						///////////////  Attenuation from the scatter crystal to detector unit within the detector crystal /////////////////
						float length_crystalself_att1 = 0.0000f;
						float length_crystalself_att2 = 0.0000f;

						length_crystalself_att1 = length_box_ray_inside(x_scatter_self, y_scatter_self, z_scatter_self, xDetector_self, yDetector_self, zDetector_self, x1_detectorcrystal_self, y1_detectorcrystal_self, z1_detectorcrystal_self, x2_detectorcrystal_self, y2_detectorcrystal_self, z2_detectorcrystal_self);
						length_crystalself_att2 = length_box_ray_inside(x_scatter_self, y_scatter_self, z_scatter_self, xDetector_self, yDetector_self, zDetector_self, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						float absorp_coeff_detector_total = coeff_detector_total;
						// 100keV~1MeV,cross section of photon electric effect is propotional to E^-3;
						// 100keV~1MeV, cross section of compton effect is propotional to E^-1;
						float absorp_coeff_detector_pe = coeff_detector_pe * target_PE_Energy * target_PE_Energy * target_PE_Energy / scatterEnergy / scatterEnergy / scatterEnergy;
						float absorp_coeff_detector_compton = coeff_detector_compton * target_PE_Energy / scatterEnergy;
						absorp_coeff_detector_total = absorp_coeff_detector_pe + absorp_coeff_detector_compton;
						attenuation_dist_crystal_crystal = attenuation_dist_crystal_crystal + (length_crystalself_att1 - length_crystalself_att2) * absorp_coeff_detector_total;

						/////////////////////////////////  Absoption of scattered photons within the detector unit /////////////////////////////
						x_tmp = xDetector_self + dist_extend * (xDetector_self - x_scatter_self) / L_comptonAngle;
						y_tmp = yDetector_self + dist_extend * (yDetector_self - y_scatter_self) / L_comptonAngle;
						z_tmp = zDetector_self + dist_extend * (zDetector_self - z_scatter_self) / L_comptonAngle;
						float length_absorp = 0.000f;
						length_absorp = length_box_ray(x_scatter_self, y_scatter_self, z_scatter_self, x_tmp, y_tmp, z_tmp, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						//prob_Compton_to_detectionCrystal += prob_Compton_othercrystal * Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f-exp(-length_absorp* absorp_coeff_detector_total))* absorp_coeff_detector_pe/ absorp_coeff_detector_total;																		
						atomicAdd(&dst[dstIndex], prob_Compton_othercrystal* Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f - exp(-length_absorp * absorp_coeff_detector_total)) * absorp_coeff_detector_pe / absorp_coeff_detector_total);
					}
				}
			}
			//atomicAdd(&dst[dstIndex], prob_Compton_to_detectionCrystal);
		}

	}


}


__global__ void geometryRelationShip_Crystal2Crystal(unsigned int * dst_relation_crystal2crystal, float* deviceparameter_Detector)
{
	// clculate global index
	int numProjectionSingle =(int) deviceparameter_Detector[0];
	long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) + static_cast<long long>(threadIdx.x);
	long long total_threads = static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);

	if (idx >= total_threads) return;

	int k = idx % numProjectionSingle;
	int j = (idx / numProjectionSingle) % numProjectionSingle;
	int i = idx / (numProjectionSingle * numProjectionSingle);

	if (i >= j) return;
	if (k == i) return;
	if (k == j) return;

	long long pair_idx = static_cast<long long>(i) * static_cast<long long>(numProjectionSingle) - (static_cast<long long>(i) * static_cast<long long>((i + 1))) / 2 + static_cast<long long>(j);

	long long bit_idx = pair_idx * static_cast<long long>(numProjectionSingle) + static_cast<long long>(k);

	int bits_per_word = 32;
	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	float xDetector_i = deviceparameter_Detector[12 * i + 1];
	float yDetector_i = deviceparameter_Detector[12 * i + 2];
	float zDetector_i = deviceparameter_Detector[12 * i + 3];
	float widthDetector_i = deviceparameter_Detector[12 * i + 4];
	float heightDetector_i = deviceparameter_Detector[12 * i + 6];
	float thicknessDetector_i = deviceparameter_Detector[12 * i + 5];
	float R_detector_i = sqrt(widthDetector_i * widthDetector_i + heightDetector_i * heightDetector_i + thicknessDetector_i * thicknessDetector_i) / 2.0f;
	
	float xDetector_j = deviceparameter_Detector[12 * j + 1];
	float yDetector_j = deviceparameter_Detector[12 * j + 2];
	float zDetector_j = deviceparameter_Detector[12 * j + 3];
	float widthDetector_j = deviceparameter_Detector[12 * j + 4];
	float heightDetector_j = deviceparameter_Detector[12 * j + 6];
	float thicknessDetector_j  = deviceparameter_Detector[12 * j + 5];
	float R_detector_j = sqrt(widthDetector_j * widthDetector_j + heightDetector_j * heightDetector_j + thicknessDetector_j * thicknessDetector_j) / 2.0f;

	float L_ij = calculateDist(xDetector_i, yDetector_i, zDetector_i, xDetector_j, yDetector_j, zDetector_j);

	float crit_i_j = R_detector_i / L_ij;
	float crit_j_i = R_detector_j / L_ij;

	float x_projectionOnUnitSphere_i_j = (xDetector_i - xDetector_j) / L_ij;
	float y_projectionOnUnitSphere_i_j = (yDetector_i - yDetector_j) / L_ij;
	float z_projectionOnUnitSphere_i_j = (zDetector_i - zDetector_j) / L_ij;

	float x_projectionOnUnitSphere_j_i = (xDetector_j - xDetector_i) / L_ij;
	float y_projectionOnUnitSphere_j_i = (yDetector_j - yDetector_i) / L_ij;
	float z_projectionOnUnitSphere_j_i = (zDetector_j - zDetector_i) / L_ij;

	
	float xDetector_k = deviceparameter_Detector[12 * k + 1];
	float yDetector_k = deviceparameter_Detector[12 * k + 2];
	float zDetector_k = deviceparameter_Detector[12 * k + 3];
	float widthDetector_k = deviceparameter_Detector[12 * k + 4];
	float heightDetector_k = deviceparameter_Detector[12 * k + 6];
	float thicknessDetector_k = deviceparameter_Detector[12 * k + 5];
	float R_detector_k = sqrt(widthDetector_k * widthDetector_k + heightDetector_k * heightDetector_k + thicknessDetector_k * thicknessDetector_k) / 2.0f;

	
	float L_ik = calculateDist(xDetector_k, yDetector_k, zDetector_k, xDetector_i, yDetector_i, zDetector_i);
	float crit_k_i = R_detector_k / L_ik;

	float L_jk = calculateDist(xDetector_k, yDetector_k, zDetector_k, xDetector_j, yDetector_j, zDetector_j);
	float crit_k_j = R_detector_k / L_jk;

	float x_projectionOnUnitSphere_k_i = (xDetector_k - xDetector_i) / L_ik;
	float y_projectionOnUnitSphere_k_i = (yDetector_k - yDetector_i) / L_ik;
	float z_projectionOnUnitSphere_k_i = (zDetector_k - zDetector_i) / L_ik;

	float x_projectionOnUnitSphere_k_j = (xDetector_k - xDetector_j) / L_jk;
	float y_projectionOnUnitSphere_k_j = (yDetector_k - yDetector_j) / L_jk;
	float z_projectionOnUnitSphere_k_j = (zDetector_k - zDetector_j) / L_jk;
	
	
	int flagcross = 0;
	// Whether the cover sphere of detector k is cross with the line between i and j, centered at i 
	float distOnUnitSphere_i = calculateDist(x_projectionOnUnitSphere_k_i, y_projectionOnUnitSphere_k_i, z_projectionOnUnitSphere_k_i, x_projectionOnUnitSphere_j_i, y_projectionOnUnitSphere_j_i, z_projectionOnUnitSphere_j_i);
	if (distOnUnitSphere_i <= crit_k_i + crit_j_i)
	{
		flagcross = 1;
	}
	else
	{	
		// Whether the cover sphere of detector k is cross with the line between i and j, centered at j 
		float distOnUnitSphere_j = calculateDist(x_projectionOnUnitSphere_k_j, y_projectionOnUnitSphere_k_j, z_projectionOnUnitSphere_k_j, x_projectionOnUnitSphere_i_j, y_projectionOnUnitSphere_i_j, z_projectionOnUnitSphere_i_j);
		if (distOnUnitSphere_j <= crit_k_j + crit_i_j)
		{
			flagcross = 1;
		}
	}

	if (flagcross==1)
	{		
		atomicOr(&dst_relation_crystal2crystal[word_idx], 1U << bit_offset);
	}
}



__global__ void collimatorScatterSysMatCuda(float* dst,
	float* deviceparameter_Collimator,
	float* deviceparameter_Detector,
	float* deviceparameter_Image,
	float* deviceparameter_Physics,
	unsigned int* deviceGeometryRelationShip_Collimator2Crystal,
	int numProjectionSingle,
	int numImagebin)

{

	int numCollimator_Holes = (int)floor(deviceparameter_Collimator[10] + 0.000001f);
	int numDetectorbins = numProjectionSingle;

	float _float_FOV2Collimator = deviceparameter_Image[11];

	////////////////////////////////////////// Collimator Parameters //////////////////////////////////////////
	/*
	float _float_numCollimatorHoles[5];
	int numCollimatorHoles[5];
	float _float_widthCollimatorLayers[5];
	float _float_heightCollimatorLayers[5];
	float _float_thicknessCollimatorLayers[5];
	float _float_coeffCollimatorLayers[5];

	int numCollimatorHoles_tot = 0;
	for (unsigned int id_CollimatorLayer = 0; id_CollimatorLayer < numCollimatorLayer; id_CollimatorLayer++)
	{
		_float_numCollimatorHoles[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0];
		numCollimatorHoles[id_CollimatorLayer] = (int)floor(_float_numCollimatorHoles[id_CollimatorLayer]);
		_float_widthCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1];
		_float_thicknessCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2];
		_float_heightCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3];
		_float_coeffCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5];
		numCollimatorHoles_tot = numCollimatorHoles_tot + numCollimatorHoles[id_CollimatorLayer];
	}
	*/
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////// Image Parameters ////////////////////////////////////////////

	float _float_widthImageVoxelX = deviceparameter_Image[3];
	float _float_widthImageVoxelY = deviceparameter_Image[4];
	float _float_widthImageVoxelZ = deviceparameter_Image[5];

	//float _float_numRotation = deviceparameter_Image[6];//numRotation;
	float _float_angelPerRotation = deviceparameter_Image[7];//Angel per Rotation;
	float _float_idxrotation = deviceparameter_Image[20];//idxRotation
	//float RotationAngle = _float_idxrotation / _float_numRotation * (2 * M_PI);
	float RotationAngle = _float_idxrotation * _float_angelPerRotation;
	float shiftFOVX_physics = deviceparameter_Image[8];
	float shiftFOVY_physics = deviceparameter_Image[9];
	float shiftFOVZ_physics = deviceparameter_Image[10];

	int numImageVoxelX = (int)floor(deviceparameter_Image[0] + 0.001f);
	int numImageVoxelY = (int)floor(deviceparameter_Image[1] + 0.001f);
	int numImageVoxelZ = (int)floor(deviceparameter_Image[2] + 0.001f);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// grid x runs over voxels, grid y over detectors (gridDim.y <= 65535 would limit the voxels)
	long long int row = blockIdx.y * (long long)blockDim.x + threadIdx.x;
	if (row < 0 || row > numProjectionSingle - 1) { return; }
	long long int col = blockIdx.x * (long long)blockDim.y + threadIdx.y;
	if (col < 0 || col > numImagebin - 1) { return; }
	long long int slice = blockIdx.z * blockDim.z + threadIdx.z;
	if (slice < 0 || slice > numCollimator_Holes - 1) { return; }

	long long int dstIndex = row * numImagebin + col;

	/*
	if (row < 5 && col < 5 && slice < 1) {
		printf("Thread (%lld, %lld, %lld) set dst[%lld] = %d\n", row, col, slice, dstIndex, numCollimator_Holes);
		printf("NumCollimator Holes = %d\n", numCollimator_Holes);
		printf("numImageVoxelX = %d\n", numImageVoxelX);
		
	}
	*/


	unsigned int idxDetector = row; // index of detector
	unsigned int id_CollimatorHole = slice; // index of scatter

	int idxImageVoxelZ = col / (numImageVoxelY * numImageVoxelX);
	col = col % (numImageVoxelY * numImageVoxelX);
	int idxImageVoxelY = col / numImageVoxelX;
	int idxImageVoxelX = col % numImageVoxelX;


	const unsigned int divideX = 1, divideY = 1, divideZ = 1;

	///////////////////////////////////////// Physic Progress Parameters /////////////////////////////////////
	int flagUsingCompton = (int)floor(deviceparameter_Physics[0] + 0.5f);
	int flagUsingSameEnergyWindow = (int)floor(deviceparameter_Physics[4] + 0.5f);

	float lowerThresholdofEnergyWindow = deviceparameter_Physics[5];
	float upperThresholdofEnergyWindow = deviceparameter_Physics[6];

	float target_PE_Energy = deviceparameter_Physics[7];
	float energy_resolution_detector_targetPE = deviceparameter_Detector[idxDetector * 12 + 10];

	// Energy Window of detector crystal
	if (flagUsingSameEnergyWindow > 0)
	{
		lowerThresholdofEnergyWindow = deviceparameter_Physics[5];
		upperThresholdofEnergyWindow = deviceparameter_Physics[6];
	}
	else
	{
		lowerThresholdofEnergyWindow = (1 - energy_resolution_detector_targetPE / 2.0f) * target_PE_Energy;
		upperThresholdofEnergyWindow = (1 + energy_resolution_detector_targetPE / 2.0f) * target_PE_Energy;
	}

	float coeff_detector_total = deviceparameter_Detector[idxDetector * 12 + 7];
	float coeff_detector_pe = deviceparameter_Detector[idxDetector * 12 + 8];
	float coeff_detector_compton = deviceparameter_Detector[idxDetector * 12 + 9];

	float integration_Compton = computeComptonIntegral(target_PE_Energy, 0.0f, M_PI, 0.01);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////// Image Rotation Shift Parameters /////////////////////////////////////
	float xImage = (idxImageVoxelX - numImageVoxelX / 2.0f + 0.5f) * _float_widthImageVoxelX;
	float yImage = (idxImageVoxelY - numImageVoxelY / 2.0f + 0.5f) * _float_widthImageVoxelY;
	float zImage = (idxImageVoxelZ - numImageVoxelZ / 2.0f + 0.5f) * _float_widthImageVoxelZ;

	xImage = xImage + shiftFOVX_physics;
	yImage = yImage + shiftFOVY_physics;
	zImage = zImage + shiftFOVZ_physics;

	float xImage_rot = xImage * cos(RotationAngle) - yImage * sin(RotationAngle);
	float yImage_rot = xImage * sin(RotationAngle) + yImage * cos(RotationAngle);
	float zImage_rot = zImage;
	xImage = xImage_rot;
	yImage = yImage_rot;
	zImage = zImage_rot;


	// All variables without a suffix are in the real-world physical coordinate system
	// All parameters with 'self' suffix are in the detector crystal coordinate system
	float xDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 1];
	float yDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 2] + _float_FOV2Collimator;
	float zDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 3];

	float widthDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 4];
	float heightDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 6];
	float thicknessDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 5];

	float rotationAngel_DetectorCrystal = deviceparameter_Detector[12 * idxDetector + 11];

	float xImage_self = (xImage - xDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal) - (zImage - zDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal);
	float yImage_self = yImage - yDetectorCrystalCenter;
	float zImage_self = (xImage - xDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal) + (zImage - zDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal);

	float x1_detectorcrystal_self = -widthDetectorCrystal / 2.0f;
	float x2_detectorcrystal_self = widthDetectorCrystal / 2.0f;
	float y1_detectorcrystal_self = -thicknessDetectorCrystal / 2.0f;
	float y2_detectorcrystal_self = thicknessDetectorCrystal / 2.0f;
	float z1_detectorcrystal_self = -heightDetectorCrystal / 2.0f;
	float z2_detectorcrystal_self = heightDetectorCrystal / 2.0f;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////  Compton Scatter from other collimator holes ///////////////////////////////
	// Solid angel of collimator holes toward the center of image voxel
	float x_scatter = deviceparameter_Collimator[id_CollimatorHole * 9 + 100];
	float y_scatter = (deviceparameter_Collimator[id_CollimatorHole * 9 + 101] + deviceparameter_Collimator[id_CollimatorHole * 9 + 102]) / 2.0f + _float_FOV2Collimator;
	float z_scatter = deviceparameter_Collimator[id_CollimatorHole * 9 + 103];

	float virtual_r = 0.1f;
	float L_image_hole = calculateDist(xImage, yImage, zImage, x_scatter, y_scatter, z_scatter);


	float solid_angle = (virtual_r * virtual_r * M_PI + (deviceparameter_Collimator[id_CollimatorHole * 9 + 102] - deviceparameter_Collimator[id_CollimatorHole * 9 + 101]) * virtual_r) / (4.0f * M_PI * L_image_hole * L_image_hole) * fabs(y_scatter - yImage) / L_image_hole;
	float prob_Compton_otherCollimatorHole = solid_angle / deviceparameter_Collimator[15] * deviceparameter_Collimator[17];
	float length = 0.0f;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	
	///////////////////////////////////////// CalCulation Starts Below ///////////////////////////////////////
	if (flagUsingCompton == 1)
	{
		if (coeff_detector_total > 0.01f)
		{
			//float prob_Compton_to_detectionCrystal = 0.000;

			for (int NumZ = 0; NumZ < divideZ; NumZ++)
			{
				for (int NumX = 0; NumX < divideX; NumX++)
				{
					for (int NumY = 0; NumY < divideY; NumY++)
					{

						/////////////////////////////////  Parameters of the detector unit ////////////////////////////////
						// All variables without a suffix are in the real-world physical coordinate system
						// All parameters with 'self' suffix are in the detector crystal coordinate system
						float xDetector_self = -widthDetectorCrystal / 2.0f + (float)(NumX + 0.5f) / (float)divideX * widthDetectorCrystal;
						float zDetector_self = -heightDetectorCrystal / 2.0f + (float)(NumZ + 0.5f) / (float)divideZ * heightDetectorCrystal;
						float yDetector_self = -thicknessDetectorCrystal / 2.0f + (float)(NumY + 0.5f) / (float)divideY * thicknessDetectorCrystal;

						float xDetector_rot = xDetector_self * cos(rotationAngel_DetectorCrystal) - zDetector_self * sin(rotationAngel_DetectorCrystal);
						float zDetector_rot = xDetector_self * sin(rotationAngel_DetectorCrystal) + zDetector_self * cos(rotationAngel_DetectorCrystal);
						float yDetector_rot = yDetector_self;

						float xDetector = xDetectorCrystalCenter + xDetector_rot;
						float zDetector = zDetectorCrystalCenter + zDetector_rot;
						float yDetector = yDetectorCrystalCenter + yDetector_rot;


						/////////////////////////////////  Compton scatter probability from the hole to detection crystal /////////////////////////////

						float comptonConeAngle = calculateConeAngle(xImage, yImage, zImage, x_scatter, y_scatter, z_scatter, xDetector, yDetector, zDetector);
						float scatterEnergy = calculateScatterEnergy(comptonConeAngle, target_PE_Energy);
						// Relative FWHM scales as R(E) = R0 * sqrt(E0 / E): absolute FWHM grows as sqrt(E), so lower-energy scattered photons have worse relative resolution
						float energy_resolution_detector_scatterphoton = energy_resolution_detector_targetPE * sqrt(target_PE_Energy / scatterEnergy);

						//  The probability that a Compton scatterred photon being detected within the energy window of detector unit

						if (((scatterEnergy * (1 + 2 * energy_resolution_detector_scatterphoton / 2.35482f)) <= lowerThresholdofEnergyWindow) || (scatterEnergy * (1 - 2 * energy_resolution_detector_scatterphoton / 2.35482f) >= upperThresholdofEnergyWindow))
						{
							continue;
							// The energy of the scattered photon detected within a detector element follows a Gaussian distribution. 
							// If the 2 sigma range of this Gaussian does not overlap with the full energy peak window of the detector element, 
							// then it is considered that the scattering does not affect the result.
						}
						float energyDetected_probability = calculategaussianIntegral(scatterEnergy, energy_resolution_detector_scatterphoton, lowerThresholdofEnergyWindow, upperThresholdofEnergyWindow);


						// The probability that a Compton scattered photon, among all the photons scattered at the scattering point, 
						// is scattered towards the direction of the detector element.
						float L_comptonAngle = calculateDist(x_scatter, y_scatter, z_scatter, xDetector, yDetector, zDetector);

						// Calculate the phi range, using the detector unit's minimum enclosing sphere as an approximation.
						float R_detector = sqrt(widthDetectorCrystal * widthDetectorCrystal / (float)divideX / (float)divideX + heightDetectorCrystal * heightDetectorCrystal / (float)divideZ / (float)divideZ + thicknessDetectorCrystal * thicknessDetectorCrystal / (float)divideY / (float)divideY) / 2.0f;
						float Range_Phi = 0.000f;
						if (L_comptonAngle * sin(comptonConeAngle) * 2.0f <= R_detector)
						{
							Range_Phi = 2.0f * M_PI;
						}
						else
						{
							Range_Phi = 4.0f * asin(min(R_detector / 2.0f / L_comptonAngle / sin(comptonConeAngle), 1.0f));
						}

						// Calculate the theta range
						float x_scatter_self = (x_scatter - xDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal) - (z_scatter - zDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal);
						float y_scatter_self = y_scatter - yDetectorCrystalCenter;
						float z_scatter_self = (x_scatter - xDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal) + (z_scatter - zDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal);

						float x1_detectorunit_self = ((float)NumX / (float)divideX - 0.5f) * widthDetectorCrystal;
						float x2_detectorunit_self = (((float)NumX + 1.0f) / (float)divideX - 0.5f) * widthDetectorCrystal;

						float y1_detectorunit_self = ((float)NumY / (float)divideY - 0.5f) * thicknessDetectorCrystal;
						float y2_detectorunit_self = (((float)NumY + 1.0f) / (float)divideY - 0.5f) * thicknessDetectorCrystal;

						float z1_detectorunit_self = ((float)NumZ / (float)divideZ - 0.5f) * heightDetectorCrystal;
						float z2_detectorunit_self = (((float)NumZ + 1.0f) / (float)divideZ - 0.5f) * heightDetectorCrystal;


						float dist_extend = 1000.0f;
						float dist_Image_scatterer = calculateDist(x_scatter_self, y_scatter_self, z_scatter_self, xImage_self, yImage_self, zImage_self);
						float x_tmp = x_scatter_self + dist_extend * (x_scatter_self - xImage_self) / dist_Image_scatterer;
						float y_tmp = y_scatter_self + dist_extend * (y_scatter_self - yImage_self) / dist_Image_scatterer;
						float z_tmp = z_scatter_self + dist_extend * (z_scatter_self - zImage_self) / dist_Image_scatterer;

						length = length_box_ray(xImage_self, yImage_self, zImage_self, x_tmp, y_tmp, z_tmp, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						float theta[8];
						theta[0] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self);
						theta[1] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self);
						theta[2] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y2_detectorunit_self, z1_detectorunit_self);
						theta[3] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y1_detectorunit_self, z2_detectorunit_self);
						theta[4] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y2_detectorunit_self, z1_detectorunit_self);
						theta[5] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y1_detectorunit_self, z2_detectorunit_self);
						theta[6] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);
						theta[7] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						float theta_min = theta[0];
						float theta_max = theta[0];
						for (int i = 1; i < 8; i++)
						{
							if (theta[i] > theta_max)
								theta_max = theta[i];
							if (theta[i] < theta_min)
								theta_min = theta[i];
						}

						if (length > 0.001f)
						{
							theta_min = 0.000; // If the extension of the line from the image to the scatterer passes through the detector unit, then theta_min=0
						}

						// Range_Theta = 2.0f * asin(min(1.0f, R_detector / R_comptonAngle));
						float comptonAngleRatio = computeComptonIntegral(target_PE_Energy, theta_min, theta_max, 0.01) / integration_Compton;

						/////////////////////////////////  Attenuation from the scatterer hole to detector unit /////////////////////////////
						float attenuation_dist_crystal_crystal = 0.000f;

						for (int id_Detector_att = 0; id_Detector_att < numDetectorbins; id_Detector_att++)
						{
							if (id_Detector_att != idxDetector)
							{
								int bits_per_word = 32;
								int flagCross = indexFrombitmap_collimator(id_CollimatorHole, idxDetector, id_Detector_att, deviceGeometryRelationShip_Collimator2Crystal, numProjectionSingle, bits_per_word);
								if (flagCross == 0)
								{
									continue;
								}
								else
								{
									float length_att = 0;

									float x_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 1];
									float y_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 2] + _float_FOV2Collimator;
									float z_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 3];

									float width_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 4];
									float height_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 6];
									float thickness_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 5];

									float rotationAngel_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 11];

									float coeff_total_att = deviceparameter_Detector[12 * id_Detector_att + 7];
									float coeff_pe_att = deviceparameter_Detector[12 * id_Detector_att + 8];
									float coeff_compton_att = deviceparameter_Detector[12 * id_Detector_att + 9];

									// 100keV~1MeV,cross section of photon electric effect is propotional to E^-3;
									// 100keV~1MeV, cross section of compton effect is propotional to E^-1;
									coeff_pe_att = coeff_pe_att * target_PE_Energy * target_PE_Energy * target_PE_Energy / scatterEnergy / scatterEnergy / scatterEnergy;
									coeff_compton_att = coeff_compton_att * target_PE_Energy / scatterEnergy;
									coeff_total_att = coeff_pe_att + coeff_compton_att;

									float x_scatter_Att = (x_scatter - x_AttCrystalCenter) * cos(-rotationAngel_AttCrystal) - (z_scatter - z_AttCrystalCenter) * sin(-rotationAngel_AttCrystal);
									float y_scatter_Att = y_scatter - y_AttCrystalCenter;
									float z_scatter_Att = (x_scatter - x_AttCrystalCenter) * sin(-rotationAngel_AttCrystal) + (z_scatter - z_AttCrystalCenter) * cos(-rotationAngel_AttCrystal);

									float x1_Att = -0.5f * width_AttCrystal;
									float x2_Att = 0.5f * width_AttCrystal;

									float y1_Att = -0.5f * thickness_AttCrystal;
									float y2_Att = 0.5f * thickness_AttCrystal;

									float z1_Att = -0.5f * height_AttCrystal;
									float z2_Att = 0.5f * height_AttCrystal;

									float xDetector_Att = (xDetector - x_AttCrystalCenter) * cos(-rotationAngel_AttCrystal) - (zDetector - z_AttCrystalCenter) * sin(-rotationAngel_AttCrystal);
									float yDetector_Att = yDetector - y_AttCrystalCenter;
									float zDetector_Att = (xDetector - x_AttCrystalCenter) * sin(-rotationAngel_AttCrystal) + (zDetector - z_AttCrystalCenter) * cos(-rotationAngel_AttCrystal);

									length_att = length_box_ray(x_scatter_Att, y_scatter_Att, z_scatter_Att, xDetector_Att, yDetector_Att, zDetector_Att, x1_Att, y1_Att, z1_Att, x2_Att, y2_Att, z2_Att);
									attenuation_dist_crystal_crystal = attenuation_dist_crystal_crystal + length_att * coeff_total_att;

								}

							}

						}


						///////////////  Attenuation from the scatterer hole to detector unit  within the detection crystal /////////////////
						float length_crystalself_att1 = 0.0000f;
						float length_crystalself_att2 = 0.0000f;

						length_crystalself_att1 = length_box_ray_inside(x_scatter_self, y_scatter_self, z_scatter_self, xDetector_self, yDetector_self, zDetector_self, x1_detectorcrystal_self, y1_detectorcrystal_self, z1_detectorcrystal_self, x2_detectorcrystal_self, y2_detectorcrystal_self, z2_detectorcrystal_self);
						length_crystalself_att2 = length_box_ray_inside(x_scatter_self, y_scatter_self, z_scatter_self, xDetector_self, yDetector_self, zDetector_self, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						float absorp_coeff_detector_total = coeff_detector_total;
						// 100keV~1MeV,cross section of photon electric effect is propotional to E^-3;
						// 100keV~1MeV, cross section of compton effect is propotional to E^-1;
						float absorp_coeff_detector_pe = coeff_detector_pe * target_PE_Energy * target_PE_Energy * target_PE_Energy / scatterEnergy / scatterEnergy / scatterEnergy;
						float absorp_coeff_detector_compton = coeff_detector_compton * target_PE_Energy / scatterEnergy;
						absorp_coeff_detector_total = absorp_coeff_detector_pe + absorp_coeff_detector_compton;
						attenuation_dist_crystal_crystal = attenuation_dist_crystal_crystal + (length_crystalself_att1 - length_crystalself_att2) * absorp_coeff_detector_total;

						/////////////////////////////////  Absoption of scattered photons within the detector unit /////////////////////////////
						x_tmp = xDetector_self + dist_extend * (xDetector_self - x_scatter_self) / L_comptonAngle;
						y_tmp = yDetector_self + dist_extend * (yDetector_self - y_scatter_self) / L_comptonAngle;
						z_tmp = zDetector_self + dist_extend * (zDetector_self - z_scatter_self) / L_comptonAngle;
						float length_absorp = 0.000f;
						length_absorp = length_box_ray(x_scatter_self, y_scatter_self, z_scatter_self, x_tmp, y_tmp, z_tmp, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						//prob_Compton_to_detectionCrystal += prob_Compton_othercrystal * Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f-exp(-length_absorp* absorp_coeff_detector_total))* absorp_coeff_detector_pe/ absorp_coeff_detector_total;
						
						atomicAdd(&dst[dstIndex], prob_Compton_otherCollimatorHole * Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f - exp(-length_absorp * absorp_coeff_detector_total)) * absorp_coeff_detector_pe / absorp_coeff_detector_total);


					}
				}
			}
			//atomicAdd(&dst[dstIndex], prob_Compton_to_detectionCrystal);
		}

	}


}


__global__ void geometryRelationShip_Collimator2Crystal(unsigned int* dst_relation_collimator2crystal, float* deviceparameter_Detector, float* deviceparameter_Collimator)
{
	
	int numProjectionSingle = deviceparameter_Detector[0];
	int numCollimatorHoles = deviceparameter_Collimator[10];
	long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) + static_cast<long long>(threadIdx.x);
	long long total_threads = static_cast<long long>(numCollimatorHoles) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);

	if (idx >= total_threads) return;

	
	int k = idx % numProjectionSingle;
	int j = (idx / numProjectionSingle) % numProjectionSingle;
	int i = idx / (numProjectionSingle * numProjectionSingle);

	
	long long bit_idx = static_cast<long long>(i) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle) + static_cast<long long>(j) * static_cast<long long>(numProjectionSingle) + static_cast<long long>(k);

	
	int bits_per_word = 32;
	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	
	float xCollimator_i = deviceparameter_Collimator[9 * i + 100];
	float yCollimator_i = (deviceparameter_Collimator[9 * i + 101] + deviceparameter_Collimator[9 * i + 102]) / 2.0f;
	float zCollimator_i = deviceparameter_Collimator[9 * i + 103];
	//float rCollimator_i = deviceparameter_Collimator[9 * i + 104];
	//float thicknessCollimator_i = deviceparameter_Collimator[9 * i + 102] - deviceparameter_Collimator[9 * i + 101];

	float xDetector_j = deviceparameter_Detector[12 * j + 1];
	float yDetector_j = deviceparameter_Detector[12 * j + 2];
	float zDetector_j = deviceparameter_Detector[12 * j + 3];
	float widthDetector_j = deviceparameter_Detector[12 * j + 4];
	float heightDetector_j = deviceparameter_Detector[12 * j + 6];
	float thicknessDetector_j = deviceparameter_Detector[12 * j + 5];
	float R_detector_j = sqrt(widthDetector_j * widthDetector_j + heightDetector_j * heightDetector_j + thicknessDetector_j * thicknessDetector_j) / 2.0f;

	float L_ij = calculateDist(xCollimator_i, yCollimator_i, zCollimator_i, xDetector_j, yDetector_j, zDetector_j);

	float crit_j_i = R_detector_j / L_ij;

	float x_projectionOnUnitSphere_j_i = (xDetector_j - xCollimator_i) / L_ij;
	float y_projectionOnUnitSphere_j_i = (yDetector_j - yCollimator_i) / L_ij;
	float z_projectionOnUnitSphere_j_i = (zDetector_j - zCollimator_i) / L_ij;

	float xDetector_k = deviceparameter_Detector[12 * k + 1];
	float yDetector_k = deviceparameter_Detector[12 * k + 2];
	float zDetector_k = deviceparameter_Detector[12 * k + 3];
	float widthDetector_k = deviceparameter_Detector[12 * k + 4];
	float heightDetector_k = deviceparameter_Detector[12 * k + 6];
	float thicknessDetector_k = deviceparameter_Detector[12 * k + 5];
	float R_detector_k = sqrt(widthDetector_k * widthDetector_k + heightDetector_k * heightDetector_k + thicknessDetector_k * thicknessDetector_k) / 2.0f;

	float L_ik = calculateDist(xDetector_k, yDetector_k, zDetector_k, xCollimator_i, yCollimator_i, zCollimator_i);
	float crit_k_i = R_detector_k / L_ik;

	float x_projectionOnUnitSphere_k_i = (xDetector_k - xCollimator_i) / L_ik;
	float y_projectionOnUnitSphere_k_i = (yDetector_k - yCollimator_i) / L_ik;
	float z_projectionOnUnitSphere_k_i = (zDetector_k - zCollimator_i) / L_ik;


	// Whether the cover sphere of detector k is cross with the line between i and j, centered at i 
	float distOnUnitSphere_i = calculateDist(x_projectionOnUnitSphere_k_i, y_projectionOnUnitSphere_k_i, z_projectionOnUnitSphere_k_i, x_projectionOnUnitSphere_j_i, y_projectionOnUnitSphere_j_i, z_projectionOnUnitSphere_j_i);
	if (distOnUnitSphere_i <= crit_k_i + crit_j_i)
	{
		atomicOr(&dst_relation_collimator2crystal[word_idx], 1U << bit_offset);
	}


}



////////////////////////////////////////////////////////////////////////////////////////////////////
// Scatter model v2 (default; "-model legacy" selects the kernels above)
//
// First-order (single Compton) scatter, integrated over the volume where the scatter happens.
//  * Crystal scatter (crystalScatterV2): the Compton interactions of the beam from voxel v in
//    crystal j (total probability PE[j,v] * mu_C/mu_PE) are spread over a grid of points filling j
//    (Gauss-Legendre across j), weighted by the transmission of the incoming beam to each point
//    (collimator layers and holes, crystals on the way, j itself), so a pinhole spot covering part
//    of j is followed.  From every point the scattered photon is followed to the faces of detector
//    crystal i that it can see, divided into cells of equal angle (exact rectangle solid angles,
//    Klein-Nishina angular distribution, energy and window per cell: at 511 keV the lower edge of
//    a wide window lies near 90 deg, the direction of the neighbouring crystals), attenuated in j
//    itself and in the crystals in between, and photo-absorbed in i (chord through i averaged
//    where it drops towards the edges of a face).
//  * Collimator scatter: scatter points fill the collimator material.  collimatorScatterShellV2
//    integrates a thin shell around every hole (all of the scatter of an opaque plate comes from
//    there, mostly from the walls hit by the beam inside the hole), collimatorScatterBulkV2 the rest
//    of the plate (scatter of photons that penetrate it, important at 511 keV; the part of every
//    grid cell outside the holes and shells is sub-sampled).  Incoming and scattered photons are
//    attenuated through all collimator layers and their holes.
//  The scattered photon is counted if it is photo-absorbed in i (the definition of the PE matrix),
//  with the Gaussian energy-window probability of i at its energy E' (FWHM/E ~ E^-1/2).
//  Attenuation coefficients at E': photoelectric ~ E^-3, Compton ~ Klein-Nishina total cross section
//  (the same scaling as MCGen_MultiGPU with material "params", so "physics = single_scatter" of the
//  Monte Carlo is an exact reference for this model).
////////////////////////////////////////////////////////////////////////////////////////////////////

// Quadrature settings (can be overridden with -D at compile time)
#ifndef S2_XT_GRID
#define S2_XT_GRID 0.2f       // crystal scatter: scatter-point spacing <= S2_XT_GRID x distance of the crystal centres
#endif
#ifndef S2_XT_NXZ
#define S2_XT_NXZ 4           //   at most this many points across the scatter crystal (x, z) ...
#endif
#ifndef S2_XT_NY
#define S2_XT_NY 6            //   ... and along its thickness (y)
#endif
#ifndef S2_XT_MIN
#define S2_XT_MIN 2           //   at least this many points across the crystal (x, z)
#endif
#ifndef S2_XT_GAUSS
#define S2_XT_GAUSS 1         //   x, z: Gauss-Legendre nodes (1) or midpoints (0)
#endif
#ifndef S2_PATCH_ANG
#define S2_PATCH_ANG 0.5f     // faces of the detector crystal seen from a scatter point: patches (collimator attenuation)
#endif                        //   of at most S2_PATCH_ANG rad ...
#ifndef S2_PATCH_MAX
#define S2_PATCH_MAX 8        //   ... and at most this many patches per face edge
#endif
#ifndef S2_CELL_ANG
#define S2_CELL_ANG 0.1f      //   cells (angle, energy, window, chord, crystal shadowing) of at most S2_CELL_ANG rad ...
#endif
#ifndef S2_CHORD_N
#define S2_CHORD_N 4          //   ... and at least S2_CHORD_N cells per face edge
#endif
#ifndef S2_WIN_SUB
#define S2_WIN_SUB 3          //   energy window: S2_WIN_SUB^2 sub-points per cell where the energies reach an edge of the
#endif                        //   window (4 sigma) and change by more than sigma/4 across a cell
#ifndef S2_CHORD_SUB
#define S2_CHORD_SUB 4        //   chord through the detector crystal: S2_CHORD_SUB^2 sub-points in cells next to the edges
#endif                        //   the rays run towards (there the chord drops to 0 through the side face)
#ifndef S2_SH_NPSI
#define S2_SH_NPSI 12         // collimator shell: points around the hole ...
#endif
#ifndef S2_SH_NY
#define S2_SH_NY 3            //   ... along the hole (x3: clustered at the front face, the back face, uniform) ...
#endif
#ifndef S2_SH_ND
#define S2_SH_ND 3            //   ... and into the wall (x2: quantiles of the expected attenuation and of 1/16 of it)
#endif
#ifndef S2_SH_MFP
#define S2_SH_MFP 5.0f        //   shell thickness = min(S2_SH_MFP mean free paths at E0, S2_SH_MAX mm)
#endif
#ifndef S2_SH_MAX
#define S2_SH_MAX 1.5f
#endif
#ifndef S2_BK_N
#define S2_BK_N 24            // collimator bulk: lateral grid (x and z) ...
#endif
#ifndef S2_BK_NY
#define S2_BK_NY 4            //   ... and points through the thickness (Gauss-Legendre)
#endif
#ifndef S2_BK_SUB
#define S2_BK_SUB 4           //   lateral sub-points per cell for the part of the cell outside holes and shells
#endif
#ifndef S2_BK_SKIP_MUT
#define S2_BK_SKIP_MUT 15.0f  //   no bulk term for layers with mu*T above this (opaque: < e^-15)
#endif
#ifndef S2_PRUNE
#define S2_PRUNE 1e-5f        // collimator scatter: skip scatter points whose incoming weight is below S2_PRUNE x the largest
#endif
#ifndef S2_PE_REL
#define S2_PE_REL 1e-7f       // crystal scatter: skip (j, v) with PE[j,v] < S2_PE_REL x max(PE)
#endif
#define S2_MAXK 64            // crystals between scatter crystal and detector crystal kept per thread
#define S2_MAXNB 24           // nearest crystals checked for shadowing of collimator scatter
#define S2_MAXLAYERS 10
#define S2_NXI 12             // floats per crystal in the crystal-info array

// Gauss-Legendre nodes on [-1, 1] and weights (sum 2) for n = 1..8, node k of n at n(n-1)/2 + k
__constant__ float s2GLx[36] = {
    0.0f,
    -0.5773502692f, 0.5773502692f,
    -0.7745966692f, 0.0f, 0.7745966692f,
    -0.8611363116f, -0.3399810436f, 0.3399810436f, 0.8611363116f,
    -0.9061798459f, -0.5384693101f, 0.0f, 0.5384693101f, 0.9061798459f,
    -0.9324695142f, -0.6612093865f, -0.2386191861f, 0.2386191861f, 0.6612093865f, 0.9324695142f,
    -0.9491079123f, -0.7415311856f, -0.4058451514f, 0.0f, 0.4058451514f, 0.7415311856f, 0.9491079123f,
    -0.9602898565f, -0.7966664774f, -0.5255324099f, -0.1834346425f, 0.1834346425f, 0.5255324099f, 0.7966664774f, 0.9602898565f };
__constant__ float s2GLw[36] = {
    2.0f,
    1.0f, 1.0f,
    0.5555555556f, 0.8888888889f, 0.5555555556f,
    0.3478548451f, 0.6521451549f, 0.6521451549f, 0.3478548451f,
    0.2369268851f, 0.4786286705f, 0.5688888889f, 0.4786286705f, 0.2369268851f,
    0.1713244924f, 0.3607615730f, 0.4679139346f, 0.4679139346f, 0.3607615730f, 0.1713244924f,
    0.1294849662f, 0.2797053915f, 0.3818300505f, 0.4179591837f, 0.3818300505f, 0.2797053915f, 0.1294849662f,
    0.1012285363f, 0.2223810345f, 0.3137066459f, 0.3626837834f, 0.3626837834f, 0.3137066459f, 0.2223810345f, 0.1012285363f };
// node k of n points on [-1, 1] and its weight (Gauss-Legendre for n <= 8 if gauss, else midpoints)
__device__ inline float s2node(int n, int k, bool gauss, float& w) {
    if (gauss && n <= 8) { int o = n * (n - 1) / 2 + k; w = s2GLw[o]; return s2GLx[o]; }
    w = 2.0f / n;
    return 2.0f * (k + 0.5f) / n - 1.0f;
}

struct S2V { float x, y, z; };
__device__ inline S2V s2v(float x, float y, float z) { S2V r; r.x = x; r.y = y; r.z = z; return r; }
__device__ inline S2V s2add(S2V a, S2V b) { return s2v(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline S2V s2sub(S2V a, S2V b) { return s2v(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline S2V s2mul(S2V a, float s) { return s2v(a.x * s, a.y * s, a.z * s); }
__device__ inline float s2dot(S2V a, S2V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float s2len(S2V a) { return sqrtf(s2dot(a, a)); }
__device__ inline float s2get(S2V a, int k) { return k == 0 ? a.x : (k == 1 ? a.y : a.z); }
__device__ inline void s2put(S2V& a, int k, float v) { if (k == 0) a.x = v; else if (k == 1) a.y = v; else a.z = v; }

// Crystal k as an oriented box. Crystal-info layout (S2_NXI floats, built by s2_crystalInfo):
// centre x, y (+FOV2Collimator), z | half width, half thickness, half height | cos, sin of the
// rotation about y | bounding radius | half extents of the world-aligned bounding box
struct S2Box { S2V c, h, a; float cr, sr, R; };
__device__ inline S2Box s2box(const float* xi, int k) {
    const float* p = xi + S2_NXI * k;
    S2Box b;
    b.c = s2v(p[0], p[1], p[2]); b.h = s2v(p[3], p[4], p[5]);
    b.cr = p[6]; b.sr = p[7]; b.R = p[8]; b.a = s2v(p[9], p[10], p[11]);
    return b;
}
// world <-> crystal frame (same convention as the kernels above: rotation about y)
__device__ inline S2V s2dirToLocal(const S2Box& b, S2V d) { return s2v(d.x * b.cr + d.z * b.sr, d.y, -d.x * b.sr + d.z * b.cr); }
__device__ inline S2V s2toLocal(const S2Box& b, S2V p) { return s2dirToLocal(b, s2sub(p, b.c)); }
__device__ inline S2V s2dirToWorld(const S2Box& b, S2V d) { return s2v(d.x * b.cr - d.z * b.sr, d.y, d.x * b.sr + d.z * b.cr); }
__device__ inline S2V s2toWorld(const S2Box& b, S2V p) { return s2add(b.c, s2dirToWorld(b, p)); }

// Clip the segment p0 -> p1 to the box [lo, hi]: parameter range [t0, t1] in [0, 1]; false if outside
__device__ inline bool s2clip(S2V p0, S2V p1, S2V lo, S2V hi, float& t0, float& t1) {
    S2V d = s2sub(p1, p0);
    t0 = 0.0f; t1 = 1.0f;
    for (int k = 0; k < 3; ++k) {
        float o = s2get(p0, k), dk = s2get(d, k), a = s2get(lo, k), b = s2get(hi, k);
        if (fabsf(dk) < 1e-12f) { if (o < a || o > b) return false; continue; }
        float ta = (a - o) / dk, tb = (b - o) / dk;
        if (ta > tb) { float t = ta; ta = tb; tb = t; }
        t0 = fmaxf(t0, ta); t1 = fminf(t1, tb);
        if (t1 <= t0) return false;
    }
    return true;
}
// Length of the segment p0 -> p1 inside crystal b
__device__ inline float s2segCrystal(const S2Box& b, S2V p0, S2V p1) {
    S2V q0 = s2toLocal(b, p0), q1 = s2toLocal(b, p1);
    float t0, t1;
    if (!s2clip(q0, q1, s2mul(b.h, -1.0f), b.h, t0, t1)) return 0.0f;
    return (t1 - t0) * s2len(s2sub(q1, q0));
}
// Distance from p (inside or on the surface of a box with half sizes h) to its exit along unit d
__device__ inline float s2exit(S2V p, S2V d, S2V h) {
    float t = 1e30f;
    for (int k = 0; k < 3; ++k) {
        float dk = s2get(d, k), o = s2get(p, k), hk = s2get(h, k);
        if (dk > 1e-12f) t = fminf(t, (hk - o) / dk);
        else if (dk < -1e-12f) t = fminf(t, (-hk - o) / dk);
    }
    return fmaxf(t, 0.0f);
}
// Length of the segment p0 -> p1 inside the cylinder with axis along y through (cx, cz), y in [y1, y2]
__device__ inline float s2segCylY(S2V p0, S2V p1, float cx, float cz, float y1, float y2, float R) {
    S2V d = s2sub(p1, p0);
    float L = s2len(d);
    if (L <= 0.0f) return 0.0f;
    d = s2mul(d, 1.0f / L);
    float t0 = 0.0f, t1 = L;
    if (fabsf(d.y) < 1e-12f) { if (p0.y < y1 || p0.y > y2) return 0.0f; }
    else {
        float ta = (y1 - p0.y) / d.y, tb = (y2 - p0.y) / d.y;
        if (ta > tb) { float t = ta; ta = tb; tb = t; }
        t0 = fmaxf(t0, ta); t1 = fminf(t1, tb);
        if (t1 <= t0) return 0.0f;
    }
    float ox = p0.x - cx, oz = p0.z - cz;
    float A = d.x * d.x + d.z * d.z, B = ox * d.x + oz * d.z, C = ox * ox + oz * oz - R * R;
    if (A < 1e-12f) { if (C > 0.0f) return 0.0f; }
    else {
        float disc = B * B - A * C;
        if (disc <= 0.0f) return 0.0f;
        float sq = sqrtf(disc);
        t0 = fmaxf(t0, (-B - sq) / A); t1 = fminf(t1, (-B + sq) / A);
        if (t1 <= t0) return 0.0f;
    }
    return t1 - t0;
}
// Solid angle of the rectangle [u0,u1] x [v0,v1] (coordinates from the foot of the perpendicular)
// seen from a point at height h above its plane
__device__ inline float s2rectOmega(float u0, float u1, float v0, float v1, float h) {
    float h2 = h * h;
    float a = atanf(u1 * v1 / (h * sqrtf(u1 * u1 + v1 * v1 + h2)));
    float b = atanf(u0 * v1 / (h * sqrtf(u0 * u0 + v1 * v1 + h2)));
    float c = atanf(u1 * v0 / (h * sqrtf(u1 * u1 + v0 * v0 + h2)));
    float d = atanf(u0 * v0 / (h * sqrtf(u0 * u0 + v0 * v0 + h2)));
    return fabsf(a - b - c + d);
}
// Klein-Nishina: shape of dsigma/dOmega (= diffComptonSection) and its integral over [0, pi] with sin(theta)
__device__ inline float s2knShape(float c, float a) {
    float f1 = a * (1.0f - c), f2 = 1.0f + c * c;
    return f2 / ((1.0f + f1) * (1.0f + f1)) * (1.0f + f1 * f1 / f2 / (1.0f + f1));
}
__device__ inline float s2knTotal(float a) {
    if (a < 0.01f) return 8.0f / 3.0f * (1.0f - 2.0f * a + 5.2f * a * a);
    float l = logf(1.0f + 2.0f * a), q = 1.0f + 2.0f * a;
    return 2.0f * ((1.0f + a) / (a * a) * (2.0f * (1.0f + a) / q - l / a) + l / (2.0f * a) - (1.0f + 3.0f * a) / (q * q));
}
// Probability that energy E is recorded inside [lo, hi] by a crystal with FWHM resolution R0 at E0
__device__ inline float s2window(float E, float lo, float hi, float R0, float E0) {
    if (R0 <= 0.0f) return (E >= lo && E <= hi) ? 1.0f : 0.0f;
    float s = R0 * sqrtf(E0 / E) * E * (1.41421356f / 2.35482f);
    return 0.5f * (erff((hi - E) / s) - erff((lo - E) / s));
}

struct S2Mu { float pe, c; };     // attenuation coefficients at E0 (1/mm)
__device__ inline float s2mu(S2Mu m, float r3, float rc) { return m.pe * r3 + m.c * rc; }   // r3 = (E0/E)^3, rc = KN(E)/KN(E0)

// Scatter energy-independent constants shared by all rays of a thread
struct S2Phys { float E0, a0, knNorm, knTot0Inv, lo, hi, R0; };

// Collimator layers (Param_Collimator), the spatial hash of the holes (as in PEGen) and the hole table
struct S2Col {
    int n, totalHoles;
    float hx[S2_MAXLAYERS], y1[S2_MAXLAYERS], y2[S2_MAXLAYERS], hz[S2_MAXLAYERS];
    S2Mu mu[S2_MAXLAYERS];
    int h0[S2_MAXLAYERS], nh[S2_MAXLAYERS];
    float fov2col, maxR;
};
struct S2Hash { const int* idx; const int* cnt; float ox, oz, cell; int nx, nz, mpc; };

__device__ inline int s2layerOf(const S2Col& L, int h) {
    for (int m = 0; m < L.n; ++m) if (h >= L.h0[m] && h < L.h0[m] + L.nh[m]) return m;
    return -1;
}

// Attenuation exponent of layer m (material + holes) along p0 -> p1, in parts added to tpe, tc, th:
// at energy E it is tpe * (E0/E)^3 + tc * KN(E)/KN(E0) + th (hole material, not scaled)
__device__ void s2layerParts(const float* col, const S2Col& L, const S2Hash& H, int m, S2V p0, S2V p1, float& tpe, float& tc, float& th) {
    float t0, t1;
    if (!s2clip(p0, p1, s2v(-L.hx[m], L.y1[m], -L.hz[m]), s2v(L.hx[m], L.y2[m], L.hz[m]), t0, t1)) return;
    S2V d = s2sub(p1, p0);
    S2V q0 = s2add(p0, s2mul(d, t0)), q1 = s2add(p0, s2mul(d, t1));
    float Ls = (t1 - t0) * s2len(d), Lh = 0.0f, tauHoles = 0.0f;
    if (L.nh[m] > 0) {
        float r = L.maxR;
        int cx0 = max(0, (int)floorf((fminf(q0.x, q1.x) - r - H.ox) / H.cell));
        int cx1 = min(H.nx - 1, (int)floorf((fmaxf(q0.x, q1.x) + r - H.ox) / H.cell));
        int cz0 = max(0, (int)floorf((fminf(q0.z, q1.z) - r - H.oz) / H.cell));
        int cz1 = min(H.nz - 1, (int)floorf((fmaxf(q0.z, q1.z) + r - H.oz) / H.cell));
        for (int cx = cx0; cx <= cx1; ++cx)
            for (int cz = cz0; cz <= cz1; ++cz) {
                int cid = cx * H.nz + cz;
                for (int k = 0; k < H.cnt[cid]; ++k) {
                    int h = H.idx[cid * H.mpc + k];
                    if (h < L.h0[m] || h >= L.h0[m] + L.nh[m]) continue;
                    const float* hp = col + 100 + 9 * h;
                    // count every hole once: only in the cell that holds its centre
                    if ((int)floorf((hp[0] - H.ox) / H.cell) != cx || (int)floorf((hp[3] - H.oz) / H.cell) != cz) continue;
                    float l = s2segCylY(q0, q1, hp[0], hp[3], hp[1] + L.fov2col, hp[2] + L.fov2col, hp[4]);
                    Lh += l;
                    tauHoles += hp[5] * l;
                }
            }
    }
    float Lm = fmaxf(Ls - Lh, 0.0f);
    tpe += L.mu[m].pe * Lm;
    tc += L.mu[m].c * Lm;
    th += tauHoles;
}
// All collimator layers, in parts as above
__device__ inline void s2colParts(const float* col, const S2Col& L, const S2Hash& H, S2V p0, S2V p1, float& tpe, float& tc, float& th) {
    tpe = tc = th = 0.0f;
    for (int m = 0; m < L.n; ++m) s2layerParts(col, L, H, m, p0, p1, tpe, tc, th);
}
__device__ inline float s2colTau(const float* col, const S2Col& L, const S2Hash& H, S2V p0, S2V p1, float r3, float rc) {
    float tpe, tc, th;
    s2colParts(col, L, H, p0, p1, tpe, tc, th);
    return tpe * r3 + tc * rc + th;
}
// Smallest distance from p to the wall of a hole of layer m (negative inside a hole), skipping hole `skip`
__device__ float s2wallDist(const float* col, const S2Col& L, const S2Hash& H, int m, S2V p, int skip, float reach) {
    float best = 1e30f, r = reach + L.maxR;
    int cx0 = max(0, (int)floorf((p.x - r - H.ox) / H.cell)), cx1 = min(H.nx - 1, (int)floorf((p.x + r - H.ox) / H.cell));
    int cz0 = max(0, (int)floorf((p.z - r - H.oz) / H.cell)), cz1 = min(H.nz - 1, (int)floorf((p.z + r - H.oz) / H.cell));
    for (int cx = cx0; cx <= cx1; ++cx)
        for (int cz = cz0; cz <= cz1; ++cz) {
            int cid = cx * H.nz + cz;
            for (int k = 0; k < H.cnt[cid]; ++k) {
                int h = H.idx[cid * H.mpc + k];
                if (h == skip || h < L.h0[m] || h >= L.h0[m] + L.nh[m]) continue;
                const float* hp = col + 100 + 9 * h;
                float dx = p.x - hp[0], dz = p.z - hp[3];
                best = fminf(best, sqrtf(dx * dx + dz * dz) - hp[4]);
            }
        }
    return best;
}

// Sum over the faces of crystal I visible from p of
//   (Klein-Nishina probability per solid angle) x (solid angle) x P(photo-absorption in I) x
//   P(energy window) x exp(-attenuation on the way)
// for a photon that arrived at p along din and scatters there.  A face is divided into cells of
// equal angular size seen from p (edges at hd*tan(angle)), at most S2_CELL_ANG rad and at least
// S2_CHORD_N per edge: scatter angle, energy, window and att.nearTau (crystals next to I) at the
// centre of every cell, exact solid angle of every cell, chord through I by s2cellAbsorb, window
// averaged over sub-points where the energies reach an edge of the window (s2windowEdge).  At
// 511 keV the lower edge of a wide window lies near 90 deg, i.e. towards the neighbouring
// crystals, and in the photopeak window all of the scatter comes from angles around its edge.  The collimator attenuation
// (att.farParts) is traced once per patch of cells (at most S2_PATCH_ANG rad) and scaled to the
// energy of every cell.
__device__ inline float s2cellEdge(float e0, float e1, float a0, float a1, int k, int n, float hd) {
    return k <= 0 ? e0 : (k >= n ? e1 : hd * tanf(a0 + (a1 - a0) * k / n));
}
// Mean photo-absorption probability (1 - exp(-mT * chord)) over the cell [qu0, qu1] x [qv0, qv1] of
// face A of I (local coordinates), rays from pl through the cell centre q along e.  The chord is
// 2 h_A / |e_A| except within the strip C |e_U| (C |e_V|) of the edge the rays run towards, where it
// drops linearly to 0; cells reaching such a strip are sampled on S2_CHORD_SUB^2 sub-points (with
// their own directions: at grazing incidence the chord also changes with the direction).
__device__ float s2cellAbsorb(S2V pl, S2V q, S2V e, const S2Box& I, int A, int U, int V,
                              float qu0, float qu1, float qv0, float qv1, float mT) {
    float C = 2.0f * s2get(I.h, A) / fmaxf(fabsf(s2get(e, A)), 1e-12f);
    float eu = s2get(e, U), ev = s2get(e, V), hu = s2get(I.h, U), hv = s2get(I.h, V);
    float wu = C * fabsf(eu), wv = C * fabsf(ev);                        // strip widths
    bool su = eu > 0.0f ? qu1 > hu - wu : (eu < 0.0f && qu0 < -hu + wu);
    bool sv = ev > 0.0f ? qv1 > hv - wv : (ev < 0.0f && qv0 < -hv + wv);
    if (!su && !sv) return 1.0f - expf(-mT * s2exit(q, e, I.h));
    float sum = 0.0f;
    for (int a = 0; a < S2_CHORD_SUB; ++a)
        for (int b = 0; b < S2_CHORD_SUB; ++b) {
            S2V r = q;
            s2put(r, U, qu0 + (qu1 - qu0) * (a + 0.5f) / S2_CHORD_SUB);
            s2put(r, V, qv0 + (qv1 - qv0) * (b + 0.5f) / S2_CHORD_SUB);
            S2V d = s2sub(r, pl);
            sum += 1.0f - expf(-mT * s2exit(r, s2mul(d, 1.0f / s2len(d)), I.h));
        }
    return sum / (S2_CHORD_SUB * S2_CHORD_SUB);
}
// Must the energy window be averaged over sub-points in the cells of the rectangle [u0, u1] x [v0, v1]
// of face A (local coordinates, ncell cells along its longer side)?  Yes if the energies scattered
// from p (direction dl) into it reach an edge zone of the window (4 sigma of the resolution) and
// change by more than sigma/4 across a cell.  The scatter angle is checked at the corners, the
// centre and the point the incoming direction points at.
__device__ bool s2windowEdge(S2V pl, S2V dl, const S2Phys& P, int A, int U, int V, float fa, float u0, float u1, float v0, float v1, int ncell) {
    if (P.R0 <= 0.0f) return true;
    float ctmin = 1.0f, ctmax = -1.0f;
    float t = (fa - s2get(pl, A)) / (fabsf(s2get(dl, A)) > 1e-12f ? s2get(dl, A) : 1e-12f);
    for (int k = 0; k < 6; ++k) {
        S2V q = pl;
        s2put(q, A, fa);
        float qu = k < 4 ? ((k & 1) ? u1 : u0) : 0.5f * (u0 + u1), qv = k < 4 ? ((k & 2) ? v1 : v0) : 0.5f * (v0 + v1);
        if (k == 5) {
            if (t <= 0.0f) continue;
            qu = fminf(fmaxf(s2get(pl, U) + t * s2get(dl, U), u0), u1);
            qv = fminf(fmaxf(s2get(pl, V) + t * s2get(dl, V), v0), v1);
        }
        s2put(q, U, qu);
        s2put(q, V, qv);
        S2V e = s2sub(q, pl);
        float ct = s2dot(dl, e) / s2len(e);
        ctmin = fminf(ctmin, ct);
        ctmax = fmaxf(ctmax, ct);
    }
    float Elo = P.E0 / (1.0f + P.a0 * (1.0f - ctmin)), Ehi = P.E0 / (1.0f + P.a0 * (1.0f - ctmax));
    float sl = P.R0 * sqrtf(P.E0 / P.lo) * P.lo / 2.35482f, sh = P.R0 * sqrtf(P.E0 / P.hi) * P.hi / 2.35482f;
    bool nlo = Ehi > P.lo - 4.0f * sl && Elo < P.lo + 4.0f * sl, nhi = Ehi > P.hi - 4.0f * sh && Elo < P.hi + 4.0f * sh;
    float sig = nlo ? (nhi ? fminf(sl, sh) : sl) : (nhi ? sh : -1.0f);
    return sig > 0.0f && Ehi - Elo > 0.25f * sig * ncell;
}
// Mean window probability over the cell [qu0, qu1] x [qv0, qv1] of face A (S2_WIN_SUB^2 sub-points)
__device__ float s2cellWindow(S2V pl, S2V dl, S2V q, const S2Phys& P, int U, int V, float qu0, float qu1, float qv0, float qv1) {
    float sum = 0.0f;
    for (int a = 0; a < S2_WIN_SUB; ++a)
        for (int b = 0; b < S2_WIN_SUB; ++b) {
            S2V r = q;
            s2put(r, U, qu0 + (qu1 - qu0) * (a + 0.5f) / S2_WIN_SUB);
            s2put(r, V, qv0 + (qv1 - qv0) * (b + 0.5f) / S2_WIN_SUB);
            S2V e = s2sub(r, pl);
            float ct = s2dot(dl, e) / s2len(e);
            sum += s2window(P.E0 / (1.0f + P.a0 * (1.0f - ct)), P.lo, P.hi, P.R0, P.E0);
        }
    return sum / (S2_WIN_SUB * S2_WIN_SUB);
}
template <class Att>
__device__ float s2toCrystal(S2V p, S2V din, const S2Phys& P, const S2Box& I, S2Mu muI, const Att& att) {
    S2V pl = s2toLocal(I, p), dl = s2dirToLocal(I, din);
    float sum = 0.0f;
    for (int A = 0; A < 3; ++A) {
        float pa = s2get(pl, A), ha = s2get(I.h, A);
        float sgn = pa > ha ? 1.0f : (pa < -ha ? -1.0f : 0.0f);
        if (sgn == 0.0f) continue;
        float hd = sgn * pa - ha;                                        // height above the face
        int U = (A + 1) % 3, V = (A + 2) % 3;
        float pu = s2get(pl, U), pv = s2get(pl, V), hu = s2get(I.h, U), hv = s2get(I.h, V);
        float eu0 = -hu - pu, eu1 = hu - pu, ev0 = -hv - pv, ev1 = hv - pv; // face edges relative to the foot of p
        float au0 = atanf(eu0 / hd), au1 = atanf(eu1 / hd), av0 = atanf(ev0 / hd), av1 = atanf(ev1 / hd);
        int nu = min(S2_PATCH_MAX, max(1, (int)ceilf((au1 - au0) / S2_PATCH_ANG)));
        int nv = min(S2_PATCH_MAX, max(1, (int)ceilf((av1 - av0) / S2_PATCH_ANG)));
        int mu_ = max((S2_CHORD_N + nu - 1) / nu, (int)ceilf((au1 - au0) / (nu * S2_CELL_ANG)));   // cells per patch
        int mv_ = max((S2_CHORD_N + nv - 1) / nv, (int)ceilf((av1 - av0) / (nv * S2_CELL_ANG)));
        int Nu = nu * mu_, Nv = nv * mv_;
        for (int iu = 0; iu < nu; ++iu)
            for (int iv = 0; iv < nv; ++iv) {
                float tpe = 0.0f, tc = 0.0f, th = 0.0f;                  // collimator attenuation of the patch (centre ray)
                if (att.hasFar) {
                    S2V c = pl;
                    s2put(c, A, sgn * ha);
                    s2put(c, U, pu + hd * tanf(au0 + (au1 - au0) * (iu + 0.5f) / nu));
                    s2put(c, V, pv + hd * tanf(av0 + (av1 - av0) * (iv + 0.5f) / nv));
                    S2V cw = s2toWorld(I, c);
                    att.farParts(p, cw, tpe, tc, th);
                }
                float pu0 = s2cellEdge(eu0, eu1, au0, au1, iu * mu_, Nu, hd), pu1 = s2cellEdge(eu0, eu1, au0, au1, (iu + 1) * mu_, Nu, hd);
                float pv0 = s2cellEdge(ev0, ev1, av0, av1, iv * mv_, Nv, hd), pv1 = s2cellEdge(ev0, ev1, av0, av1, (iv + 1) * mv_, Nv, hd);
                bool edgeW = s2windowEdge(pl, dl, P, A, U, V, sgn * ha, pu + pu0, pu + pu1, pv + pv0, pv + pv1, max(mu_, mv_));
                float u1 = pu0;
                for (int a = iu * mu_; a < (iu + 1) * mu_; ++a) {
                    float u0 = u1;
                    u1 = s2cellEdge(eu0, eu1, au0, au1, a + 1, Nu, hd);
                    float uc = hd * tanf(au0 + (au1 - au0) * (a + 0.5f) / Nu);
                    float v1 = s2cellEdge(ev0, ev1, av0, av1, iv * mv_, Nv, hd);
                    for (int b = iv * mv_; b < (iv + 1) * mv_; ++b) {
                        float v0 = v1;
                        v1 = s2cellEdge(ev0, ev1, av0, av1, b + 1, Nv, hd);
                        float vc = hd * tanf(av0 + (av1 - av0) * (b + 0.5f) / Nv);
                        S2V q = pl;                                      // cell centre on the face
                        s2put(q, A, sgn * ha);
                        s2put(q, U, pu + uc);
                        s2put(q, V, pv + vc);
                        S2V e = s2sub(q, pl);
                        e = s2mul(e, 1.0f / s2len(e));
                        float ct = s2dot(dl, e);
                        float Es = P.E0 / (1.0f + P.a0 * (1.0f - ct));
                        float w = edgeW ? s2cellWindow(pl, dl, q, P, U, V, pu + u0, pu + u1, pv + v0, pv + v1)
                                        : s2window(Es, P.lo, P.hi, P.R0, P.E0);
                        if (w < 1e-9f) continue;
                        float r = P.E0 / Es, r3 = r * r * r, rc = s2knTotal(Es / 511.0f) * P.knTot0Inv;
                        float mPE = muI.pe * r3, mT = mPE + muI.c * rc;
                        if (mT <= 0.0f) continue;
                        S2V qw = s2toWorld(I, q);
                        float tau = tpe * r3 + tc * rc + th + att.nearTau(p, qw, r3, rc);
                        sum += s2knShape(ct, P.a0) * P.knNorm * s2rectOmega(u0, u1, v0, v1, hd) * w * mPE / mT *
                               s2cellAbsorb(pl, q, e, I, A, U, V, pu + u0, pu + u1, pv + v0, pv + v1, mT) * expf(-tau);
                    }
                }
            }
    }
    return sum;
}

// Attenuation by the crystals in a list, along p -> q (bounding-sphere test first)
__device__ inline float s2crystalsTau(const float* xi, const float* det, const int* kl, int nk, S2V p, S2V q, float r3, float rc) {
    float tau = 0.0f;
    S2V d = s2sub(q, p);
    float L2 = s2dot(d, d);
    for (int n = 0; n < nk; ++n) {
        int k = kl[n];
        if (k < 0) break;
        S2Box K = s2box(xi, k);
        S2V e = s2sub(K.c, p);
        float t = fminf(fmaxf(s2dot(e, d) / L2, 0.0f), 1.0f);
        S2V f = s2sub(e, s2mul(d, t));
        if (s2dot(f, f) > K.R * K.R) continue;
        float l = s2segCrystal(K, p, q);
        if (l > 0.0f) { S2Mu mk = { det[12 * k + 8], det[12 * k + 9] }; tau += s2mu(mk, r3, rc) * l; }
    }
    return tau;
}
// Attenuation of crystal-scattered photons: rest of the scatter crystal J, crystals in between
struct S2AttXtal {
    const float* xi; const float* det;
    S2Box J; S2Mu muJ;
    const int* kl; int nk;
    enum { hasFar = 0 };
    __device__ void farParts(S2V, S2V, float&, float&, float&) const {}
    __device__ float nearTau(S2V p, S2V q, float r3, float rc) const {
        return s2mu(muJ, r3, rc) * s2segCrystal(J, p, q) + s2crystalsTau(xi, det, kl, nk, p, q, r3, rc);
    }
};
// Attenuation of collimator-scattered photons: collimator layers and holes (per patch), crystals
// next to the detector crystal (per cell)
struct S2AttCol {
    const float* col; const float* xi; const float* det;
    S2Col L; S2Hash H;
    const int* nb;                                                       // S2_MAXNB neighbours of I, -1 padded
    enum { hasFar = 1 };
    __device__ void farParts(S2V p, S2V q, float& tpe, float& tc, float& th) const { s2colParts(col, L, H, p, q, tpe, tc, th); }
    __device__ float nearTau(S2V p, S2V q, float r3, float rc) const { return s2crystalsTau(xi, det, nb, S2_MAXNB, p, q, r3, rc); }
};

// Voxel centre in the world frame (same as the kernels above)
__device__ inline S2V s2voxel(const float* img, long long col) {
    int nX = (int)floorf(img[0] + 0.001f), nY = (int)floorf(img[1] + 0.001f), nZ = (int)floorf(img[2] + 0.001f);
    int iz = (int)(col / ((long long)nY * nX));
    long long rem = col % ((long long)nY * nX);
    int iy = (int)(rem / nX), ix = (int)(rem % nX);
    float x = (ix - nX / 2.0f + 0.5f) * img[3] + img[8];
    float y = (iy - nY / 2.0f + 0.5f) * img[4] + img[9];
    float z = (iz - nZ / 2.0f + 0.5f) * img[5] + img[10];
    float ang = img[20] * img[7];
    return s2v(x * cosf(ang) - y * sinf(ang), x * sinf(ang) + y * cosf(ang), z);
}
// Physics constants and the energy window of detector crystal i
__device__ inline S2Phys s2phys(const float* det, const float* phy, int i) {
    S2Phys P;
    P.E0 = phy[7];
    P.a0 = P.E0 / 511.0f;
    float kt = s2knTotal(P.a0);
    P.knNorm = 1.0f / (2.0f * (float)M_PI * kt);
    P.knTot0Inv = 1.0f / kt;
    P.R0 = det[12 * i + 10];
    if ((int)floorf(phy[4] + 0.5f) > 0) { P.lo = phy[5]; P.hi = phy[6]; }
    else { P.lo = (1.0f - P.R0 / 2.0f) * P.E0; P.hi = (1.0f + P.R0 / 2.0f) * P.E0; }
    return P;
}

// ---------------------------------------------------------------------------------------------
// Crystal scatter: thread (row = detector crystal i, col = voxel v, slice = scatter crystal j)
// ---------------------------------------------------------------------------------------------
__global__ void crystalScatterV2(float* dst, const float* col, const float* xi, const float* det, const float* img,
                                 const float* phy, const float* pe, const unsigned int* bitmapCrystal, S2Col L, S2Hash H,
                                 const int* nbList, int nP, int nI, float peMin)
{
    long long i = blockIdx.y * (long long)blockDim.x + threadIdx.x;     // grid x: voxels, grid y: detectors
    long long v = blockIdx.x * (long long)blockDim.y + threadIdx.y;
    long long j = blockIdx.z * (long long)blockDim.z + threadIdx.z;
    if (i >= nP || v >= nI || j >= nP || i == j) return;
    if ((int)floorf(phy[0] + 0.5f) != 1 || det[12 * i + 7] <= 0.01f) return;
    float Pj = pe[(long long)nI * j + v];
    S2Mu muJ = { det[12 * j + 8], det[12 * j + 9] };
    if (Pj <= peMin || muJ.pe <= 0.0f || muJ.c <= 0.0f) return;
    float PC = Pj * muJ.c / muJ.pe;                                      // Compton first-interaction probability in j

    S2Phys P = s2phys(det, phy, (int)i);
    S2V vox = s2voxel(img, v);
    S2Box J = s2box(xi, (int)j), I = s2box(xi, (int)i);
    S2Mu muI = { det[12 * i + 8], det[12 * i + 9] };

    // crystals between j and i: flagged in the bitmap and overlapping the bounding box of both
    int kl[S2_MAXK];
    int nk = 0;
    S2V lo = s2v(fminf(J.c.x - J.a.x, I.c.x - I.a.x), fminf(J.c.y - J.a.y, I.c.y - I.a.y), fminf(J.c.z - J.a.z, I.c.z - I.a.z));
    S2V hi = s2v(fmaxf(J.c.x + J.a.x, I.c.x + I.a.x), fmaxf(J.c.y + J.a.y, I.c.y + I.a.y), fmaxf(J.c.z + J.a.z, I.c.z + I.a.z));
    for (int k = 0; k < nP && nk < S2_MAXK; ++k) {
        if (k == i || k == j) continue;
        const float* q = xi + S2_NXI * k;
        if (q[0] + q[9] < lo.x || q[0] - q[9] > hi.x || q[1] + q[10] < lo.y || q[1] - q[10] > hi.y ||
            q[2] + q[11] < lo.z || q[2] - q[11] > hi.z) continue;
        if (indexFrombitmap_crystal((int)j, (int)i, k, (unsigned int*)bitmapCrystal, nP, 32) == 0) continue;
        kl[nk++] = k;
    }
    S2AttXtal att = { xi, det, J, muJ, kl, nk };

    // scatter points: grid over j, weighted by the transmission of the incoming beam to each point
    // (collimator layers and holes, crystals next to j, path inside j) and 1/r^2; their sum is
    // normalised to the Compton probability PE[j,v] * mu_C/mu_PE from the PE matrix
    float D = s2len(s2sub(J.c, I.c));
    int nx = min(S2_XT_NXZ, max(S2_XT_MIN, (int)ceilf(2.0f * J.h.x / (S2_XT_GRID * D))));
    int ny = min(S2_XT_NY, max(1, (int)ceilf(2.0f * J.h.y / (S2_XT_GRID * D))));
    int nz = min(S2_XT_NXZ, max(S2_XT_MIN, (int)ceilf(2.0f * J.h.z / (S2_XT_GRID * D))));
    float muJ0 = muJ.pe + muJ.c;
    const int* nbJ = nbList + (long long)S2_MAXNB * j;
    S2V vl = s2toLocal(J, vox), mh = s2mul(J.h, -1.0f);
    float wsum = 0.0f, acc = 0.0f;
    for (int pass = 0; pass < 2; ++pass)
        for (int a = 0; a < nx; ++a)
            for (int b = 0; b < ny; ++b)
                for (int c = 0; c < nz; ++c) {
                    float gx, gz, wy;
                    float ux = s2node(nx, a, S2_XT_GAUSS, gx), uz = s2node(nz, c, S2_XT_GAUSS, gz), uy = s2node(ny, b, false, wy);
                    S2V sl = s2v(J.h.x * ux, J.h.y * uy, J.h.z * uz);
                    float t0, t1;
                    float depth = s2clip(vl, sl, mh, J.h, t0, t1) ? (t1 - t0) * s2len(s2sub(sl, vl)) : 0.0f;
                    S2V s = s2toWorld(J, sl);
                    S2V din = s2sub(s, vox);
                    float d2 = s2dot(din, din);
                    float w = gx * gz * expf(-muJ0 * depth - s2colTau(col, L, H, vox, s, 1.0f, 1.0f) -
                                             s2crystalsTau(xi, det, nbJ, S2_MAXNB, vox, s, 1.0f, 1.0f)) / d2;
                    if (pass == 0) { wsum += w; continue; }
                    if (w <= S2_PRUNE * wsum) continue;
                    acc += w * s2toCrystal(s, s2mul(din, 1.0f / sqrtf(d2)), P, I, muI, att);
                }
    if (wsum <= 0.0f) return;
    atomicAdd(&dst[i * nI + v], PC * acc / wsum);
}

// Collimator kernels: Compton interactions per emitted photon in the volume dV at r (layer m)
__device__ inline float s2colSource(S2V r, float dV, S2V vox, int m, const float* col, const S2Col& L, const S2Hash& H) {
    S2V d = s2sub(r, vox);
    return dV * L.mu[m].c * expf(-s2colTau(col, L, H, vox, r, 1.0f, 1.0f)) / (4.0f * (float)M_PI * s2dot(d, d));
}

// Shell around a hole: scatter point (iy, ip, id) and its volume. Along the hole most of the scatter
// that escapes comes from within ~1/mu of the front face (beam hitting the rim) and of the back face
// (scattered photon leaving through it): y nodes from three densities (exponential from each face,
// uniform).  Into the wall: quantiles of exp(-kappa*delta), kappa = attenuation per unit depth of the
// incoming and the scattered photon estimated at the wall surface, and of exp(-kappa/16*delta) for
// photons that reach the wall less obliquely (front-face rim).  Each node is weighted by
// 1 / (sum of the densities), so no part of the shell depends on one density alone.
struct S2Shell { float xh, zh, R, y1, y2, mu0, dmax; };
__device__ inline void s2shellPoint(const S2Shell& g, int iy, int ip, int id, S2V vox, S2V target, S2V& r, float& dV, float& dl) {
    float T = g.y2 - g.y1, Pf = 1.0f - expf(-g.mu0 * T);
    int comp = iy / S2_SH_NY;
    float qy = ((iy % S2_SH_NY) + 0.5f) / S2_SH_NY;
    float y = comp == 0 ? g.y1 - logf(1.0f - qy * Pf) / g.mu0 : (comp == 1 ? g.y2 + logf(1.0f - qy * Pf) / g.mu0 : g.y1 + qy * T);
    float dy = 1.0f / (S2_SH_NY * (g.mu0 * (expf(-g.mu0 * (y - g.y1)) + expf(-g.mu0 * (g.y2 - y))) / Pf + 1.0f / T));
    float dpsi = 2.0f * (float)M_PI / S2_SH_NPSI;
    float psi = (ip + 0.5f) * dpsi, cp = cosf(psi), sp = sinf(psi);
    S2V n = s2v(cp, 0.0f, sp), r0 = s2v(g.xh + g.R * cp, y, g.zh + g.R * sp);
    S2V a = s2sub(r0, vox), b = s2sub(target, r0);
    float ci = fmaxf(fabsf(s2dot(n, a)) / s2len(a), 0.05f), co = fmaxf(fabsf(s2dot(n, b)) / s2len(b), 0.05f);
    float k1 = fmaxf(g.mu0 / ci + g.mu0 / co, 1.0f / g.dmax), k2 = fmaxf(k1 / 16.0f, 1.0f / g.dmax);
    float P1 = 1.0f - expf(-k1 * g.dmax), P2 = 1.0f - expf(-k2 * g.dmax);
    float qd = ((id % S2_SH_ND) + 0.5f) / S2_SH_ND;
    dl = id < S2_SH_ND ? -logf(1.0f - qd * P1) / k1 : -logf(1.0f - qd * P2) / k2;
    float wd = 1.0f / (S2_SH_ND * (k1 * expf(-k1 * dl) / P1 + k2 * expf(-k2 * dl) / P2));
    r = s2v(g.xh + (g.R + dl) * cp, y, g.zh + (g.R + dl) * sp);
    dV = (g.R + dl) * dpsi * dy * wd;
}

// ---------------------------------------------------------------------------------------------
// Collimator scatter, shell around each hole: thread (row = detector crystal i, col = voxel v,
// slice = hole h - holeOffset).  The shell is r in [R, R + dmax] around the hole axis over the hole
// length; points closer to another hole belong to that hole.
// ---------------------------------------------------------------------------------------------
__global__ void collimatorScatterShellV2(float* dst, const float* col, const float* xi, const float* det, const float* img,
                                         const float* phy, S2Col L, S2Hash H, const int* nbList, int nP, int nI,
                                         int holeOffset, int nHoles)
{
    long long i = blockIdx.y * (long long)blockDim.x + threadIdx.x;     // grid x: voxels, grid y: detectors
    long long v = blockIdx.x * (long long)blockDim.y + threadIdx.y;
    long long hs = blockIdx.z * (long long)blockDim.z + threadIdx.z;
    if (i >= nP || v >= nI || hs >= nHoles) return;
    int h = holeOffset + (int)hs;
    int m = s2layerOf(L, h);
    if (m < 0 || L.mu[m].c <= 0.0f) return;
    if ((int)floorf(phy[0] + 0.5f) != 1 || det[12 * i + 7] <= 0.01f) return;

    S2Phys P = s2phys(det, phy, (int)i);
    S2V vox = s2voxel(img, v);
    S2Box I = s2box(xi, (int)i);
    S2Mu muI = { det[12 * i + 8], det[12 * i + 9] };
    S2AttCol att = { col, xi, det, L, H, nbList + (long long)S2_MAXNB * i };

    const float* hp = col + 100 + 9 * h;
    S2Shell g;
    g.xh = hp[0]; g.zh = hp[3]; g.R = hp[4];
    g.y1 = hp[1] + L.fov2col; g.y2 = hp[2] + L.fov2col;
    g.mu0 = L.mu[m].pe + L.mu[m].c;
    g.dmax = fminf(S2_SH_MFP / g.mu0, S2_SH_MAX);
    const int NY = 3 * S2_SH_NY, NP = S2_SH_NPSI, ND = 2 * S2_SH_ND;
    // pass 1: largest incoming weight; pass 2: scattered photons from the points that matter
    float wmax = 0.0f, acc = 0.0f;
    for (int pass = 0; pass < 2; ++pass)
        for (int iy = 0; iy < NY; ++iy)
            for (int ip = 0; ip < NP; ++ip)
                for (int id = 0; id < ND; ++id) {
                    S2V r;
                    float dV, dl;
                    s2shellPoint(g, iy, ip, id, vox, I.c, r, dV, dl);
                    if (fabsf(r.x) > L.hx[m] || fabsf(r.z) > L.hz[m]) continue;
                    if (s2wallDist(col, L, H, m, r, h, dl) < dl) continue;    // closer to another hole
                    float w = s2colSource(r, dV, vox, m, col, L, H);
                    if (pass == 0) { wmax = fmaxf(wmax, w); continue; }
                    if (w <= S2_PRUNE * wmax) continue;
                    S2V din = s2sub(r, vox);
                    acc += w * s2toCrystal(r, s2mul(din, 1.0f / s2len(din)), P, I, muI, att);
                }
    atomicAdd(&dst[i * nI + v], acc);
}

// ---------------------------------------------------------------------------------------------
// Collimator scatter, rest of the plate: thread (row = detector crystal i, col = voxel v, slice = layer m).
// Grid over the part of the layer from which the scattered photon can still reach the energy
// window; points inside a hole shell are left to collimatorScatterShellV2.
// ---------------------------------------------------------------------------------------------
__global__ void collimatorScatterBulkV2(float* dst, const float* col, const float* xi, const float* det, const float* img,
                                        const float* phy, S2Col L, S2Hash H, const int* nbList, int nP, int nI)
{
    long long i = blockIdx.y * (long long)blockDim.x + threadIdx.x;     // grid x: voxels, grid y: detectors
    long long v = blockIdx.x * (long long)blockDim.y + threadIdx.y;
    int m = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nP || v >= nI || m >= L.n) return;
    float T = L.y2[m] - L.y1[m], mu0 = L.mu[m].pe + L.mu[m].c;
    if (L.mu[m].c <= 0.0f || mu0 * T > S2_BK_SKIP_MUT) return;
    if ((int)floorf(phy[0] + 0.5f) != 1 || det[12 * i + 7] <= 0.01f) return;

    S2Phys P = s2phys(det, phy, (int)i);
    S2V vox = s2voxel(img, v);
    S2Box I = s2box(xi, (int)i);
    S2Mu muI = { det[12 * i + 8], det[12 * i + 9] };
    S2AttCol att = { col, xi, det, L, H, nbList + (long long)S2_MAXNB * i };

    // lateral region: around the crossing of voxel -> crystal with the layer, as far as the largest
    // scatter angle that can still give a count in the window (4 sigma below its lower edge)
    float ym = 0.5f * (L.y1[m] + L.y2[m]);
    float x0 = -L.hx[m], x1 = L.hx[m], z0 = -L.hz[m], z1 = L.hz[m];
    float s = P.R0 > 0.0f ? P.R0 * sqrtf(P.E0 / P.lo) * P.lo / 2.35482f : 0.0f;
    float Emin = P.lo - 4.0f * s, Eback = P.E0 / (1.0f + 2.0f * P.a0);
    if (Emin > Eback && I.c.y != vox.y) {
        float cmax = 1.0f - (P.E0 / Emin - 1.0f) / P.a0;
        float th = acosf(fminf(fmaxf(cmax, -1.0f), 1.0f));
        if (th < 1.2f) {
            float t = (ym - vox.y) / (I.c.y - vox.y);
            float px = vox.x + t * (I.c.x - vox.x), pz = vox.z + t * (I.c.z - vox.z);
            float aa = fabsf(ym - vox.y), bb = fabsf(I.c.y - ym);
            float rho = aa * bb / (aa + bb) * tanf(th) * 1.5f + I.R + 2.0f;
            x0 = fmaxf(x0, px - rho); x1 = fminf(x1, px + rho);
            z0 = fmaxf(z0, pz - rho); z1 = fminf(z1, pz + rho);
            if (x1 <= x0 || z1 <= z0) return;
        }
    }
    float dmax = fminf(S2_SH_MFP / mu0, S2_SH_MAX);
    float dx = (x1 - x0) / S2_BK_N, dz = (z1 - z0) / S2_BK_N;
    float wmax = 0.0f, acc = 0.0f;
    for (int pass = 0; pass < 2; ++pass)
        for (int a = 0; a < S2_BK_N; ++a)
            for (int c = 0; c < S2_BK_N; ++c) {
                float x = x0 + (a + 0.5f) * dx, z = z0 + (c + 0.5f) * dz, f = 1.0f;
                if (L.nh[m] > 0) {
                    // the part of the cell outside the holes and their shells (sub-points), at its centre
                    float sx = 0.0f, sz = 0.0f;
                    int n = 0;
                    for (int ka = 0; ka < S2_BK_SUB; ++ka)
                        for (int kc = 0; kc < S2_BK_SUB; ++kc) {
                            float qx = x0 + (a + (ka + 0.5f) / S2_BK_SUB) * dx, qz = z0 + (c + (kc + 0.5f) / S2_BK_SUB) * dz;
                            if (s2wallDist(col, L, H, m, s2v(qx, ym, qz), -1, dmax) < dmax) continue;   // hole or shell
                            sx += qx; sz += qz; ++n;
                        }
                    if (n == 0) continue;
                    f = (float)n / (S2_BK_SUB * S2_BK_SUB);
                    x = sx / n; z = sz / n;
                }
                for (int b = 0; b < S2_BK_NY; ++b) {
                    float wy, uy = s2node(S2_BK_NY, b, true, wy);
                    S2V r = s2v(x, 0.5f * (L.y1[m] + L.y2[m]) + 0.5f * T * uy, z);
                    float w = s2colSource(r, f * dx * dz * 0.5f * T * wy, vox, m, col, L, H);
                    if (pass == 0) { wmax = fmaxf(wmax, w); continue; }
                    if (w <= S2_PRUNE * wmax) continue;
                    S2V din = s2sub(r, vox);
                    acc += w * s2toCrystal(r, s2mul(din, 1.0f / s2len(din)), P, I, muI, att);
                }
            }
    atomicAdd(&dst[i * nI + v], acc);
}

// ---------------------------------------------------------------------------------------------
// Host-side preparation for the v2 kernels (plain C++, shared with tools/analytic_cpu)
// ---------------------------------------------------------------------------------------------
static void s2_crystalInfo(const float* det, float fov2col, std::vector<float>& xi)
{
    int nP = (int)floorf(det[0] + 0.0001f);
    xi.assign((size_t)nP * S2_NXI, 0.0f);
    for (int k = 0; k < nP; ++k) {
        float* p = &xi[(size_t)S2_NXI * k];
        float hx = 0.5f * det[12 * k + 4], hy = 0.5f * det[12 * k + 5], hz = 0.5f * det[12 * k + 6];
        float cr = cosf(det[12 * k + 11]), sr = sinf(det[12 * k + 11]);
        p[0] = det[12 * k + 1]; p[1] = det[12 * k + 2] + fov2col; p[2] = det[12 * k + 3];
        p[3] = hx; p[4] = hy; p[5] = hz; p[6] = cr; p[7] = sr;
        p[8] = sqrtf(hx * hx + hy * hy + hz * hz);
        p[9] = fabsf(cr) * hx + fabsf(sr) * hz; p[10] = hy; p[11] = fabsf(sr) * hx + fabsf(cr) * hz;
    }
}
// The S2_MAXNB nearest crystals of every crystal (candidates for shadowing it), -1 padded
static void s2_neighbours(const std::vector<float>& xi, int nP, std::vector<int>& nb)
{
    nb.assign((size_t)nP * S2_MAXNB, -1);
    std::vector<std::pair<float, int> > d;
    for (int i = 0; i < nP; ++i) {
        const float* a = &xi[(size_t)S2_NXI * i];
        d.clear();
        for (int k = 0; k < nP; ++k) {
            if (k == i) continue;
            const float* b = &xi[(size_t)S2_NXI * k];
            float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
            float dd = sqrtf(dx * dx + dy * dy + dz * dz);
            if (dd <= a[8] + b[8] + 4.0f * a[4]) d.push_back(std::make_pair(dd, k));
        }
        std::sort(d.begin(), d.end());
        for (int n = 0; n < (int)d.size() && n < S2_MAXNB; ++n) nb[(size_t)S2_MAXNB * i + n] = d[n].second;
    }
}
// Collimator layers (as in PEGen) and the hole hash grid (identical to PESysMatGen)
static S2Col s2_layers(const float* col, float fov2col)
{
    S2Col L;
    memset(&L, 0, sizeof(L));
    L.n = min((int)floorf(col[0] + 0.001f), S2_MAXLAYERS);
    L.fov2col = fov2col;
    int h0 = 0;
    for (int m = 0; m < L.n; ++m) {
        const float* b = col + (m + 1) * 10;
        L.nh[m] = (int)b[0]; L.h0[m] = h0; h0 += L.nh[m];
        L.hx[m] = b[1] / 2.0f; L.hz[m] = b[3] / 2.0f;
        L.y1[m] = -b[2] / 2.0f + fov2col + b[4]; L.y2[m] = b[2] / 2.0f + fov2col + b[4];
        L.mu[m].pe = b[6]; L.mu[m].c = b[7];
        if (L.mu[m].pe + L.mu[m].c <= 0.0f && b[5] > 0.0f) L.mu[m].pe = b[5];   // only a total coefficient given
    }
    L.totalHoles = h0;
    L.maxR = 0.0f;
    for (int h = 0; h < h0; ++h) L.maxR = fmaxf(L.maxR, col[100 + 9 * h + 4]);
    return L;
}
static void s2_holeHash(const float* col, const S2Col& L, std::vector<int>& idx, std::vector<int>& cnt, S2Hash& H)
{
    float colW = col[11], colH = col[13];
    H.cell = fmaxf(2.0f * L.maxR + 0.5f, 2.0f);
    H.ox = -colW / 2.0f - H.cell; H.oz = -colH / 2.0f - H.cell;
    H.nx = (int)ceilf((colW + 2 * H.cell) / H.cell) + 1;
    H.nz = (int)ceilf((colH + 2 * H.cell) / H.cell) + 1;
    for (int m = 1; m < L.n; ++m) {                                      // cover every layer
        H.nx = max(H.nx, (int)ceilf((2 * L.hx[m] + 2 * H.cell) / H.cell) + 1);
        H.nz = max(H.nz, (int)ceilf((2 * L.hz[m] + 2 * H.cell) / H.cell) + 1);
        H.ox = fminf(H.ox, -L.hx[m] - H.cell); H.oz = fminf(H.oz, -L.hz[m] - H.cell);
    }
    H.mpc = 16;
    idx.assign((size_t)H.nx * H.nz * H.mpc, 0);
    cnt.assign((size_t)H.nx * H.nz, 0);
    int dropped = 0;
    for (int h = 0; h < L.totalHoles; ++h) {
        float hx = col[100 + 9 * h], hz = col[103 + 9 * h], hr = col[104 + 9 * h];
        int cx0 = (int)floorf((hx - hr - H.ox) / H.cell), cx1 = (int)floorf((hx + hr - H.ox) / H.cell);
        int cz0 = (int)floorf((hz - hr - H.oz) / H.cell), cz1 = (int)floorf((hz + hr - H.oz) / H.cell);
        for (int cx = cx0; cx <= cx1; ++cx)
            for (int cz = cz0; cz <= cz1; ++cz) {
                if (cx < 0 || cx >= H.nx || cz < 0 || cz >= H.nz) continue;
                int cid = cx * H.nz + cz;
                if (cnt[cid] < H.mpc) idx[(size_t)cid * H.mpc + cnt[cid]++] = h;
                else ++dropped;
            }
    }
    if (dropped) printf("WARNING: %d hole entries dropped from the hole grid (more than %d per cell)\n", dropped, H.mpc);
    H.idx = idx.data(); H.cnt = cnt.data();
}
static float s2_peMin(const float* pe, size_t n)
{
    float mx = 0.0f;
    for (size_t k = 0; k < n; ++k) mx = fmaxf(mx, pe[k]);
    return S2_PE_REL * mx;
}


int scatter(float* parameter_Collimator, float* parameter_Detector, float* parameter_Image, float* parameter_Physics,float* PE_SysMat,const char* FnameGeoCrystal, const char* FnameGeoCollimator, float* dst, int cuda_id, float* dstCrystal, float* dstCollimator, int model)
{

	cout << "Get into scatter function" << endl;

	int numPSFImageVoxelX = (int)floor(parameter_Image[0] + 0.001f);
	int numPSFImageVoxelY = (int)floor(parameter_Image[1] + 0.001f);
	int numPSFImageVoxelZ = (int)floor(parameter_Image[2] + 0.001f);

	int numProjectionSingle = (int)floor(parameter_Detector[0]+0.0001f);
	int numImagebin = numPSFImageVoxelX * numPSFImageVoxelY * numPSFImageVoxelZ;
	int numRotation = (int)floor(parameter_Image[6] + 0.001f);
	
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
	}
	if (cuda_id>=deviceCount)
	{
		cout << "cuda_id > the number of GPUs on the host! Set Device to Device 0!" << endl;
	}
	else
	{
		cudaCheckError(cudaSetDevice(cuda_id));
		cout << "Set Device to Device " <<cuda_id<< endl;
	}

	cudaStream_t stream;
	cudaCheckError(cudaStreamCreate(&stream));
	//////////////////////////// Promary Compton Scatter between Crystals ////////////////////////////

	////------------------------- Geometry RelationShip between Crystals -------------------------////
	// Calculate geometry relationship between each crystal, if the crystal_k is cross the line of 
	// crystal_i and crystal_j, then deviceGeometryRelationShip_Crystal[i,j,k]=1,else =0.
	// The array deviceGeometryRelationShip_Crystal is a bitmap, the stored by bit, the index follows the below ruls:
	cout << "Start Calculate or IO Geometry RelationShip between Crystals "<< endl;
	long long unique_pairs = (long long)numProjectionSingle * (numProjectionSingle + 1) / 2;
	long long total_bits = static_cast<long long>(unique_pairs) * static_cast<long long>(numProjectionSingle);
	int bits_per_word = 32;
	long long array_size = (total_bits + bits_per_word - 1) / bits_per_word;

	cout << "Total unique pairs (i <= j): " << unique_pairs << endl;
	cout << "Total bits: " << total_bits << endl;
	cout << "Array size (unsigned int): " << array_size << endl;

	unsigned int* deviceGeometryRelationShip_Crystal2Crystal;
	unsigned int* hostGeometryRelationShip_Crystal2Crystal = new unsigned int[array_size]();


	float* h_parameter_Collimator;
	cudaCheckError(cudaMallocHost(&h_parameter_Collimator, sizeof(float) * 80000));
	memcpy(h_parameter_Collimator, parameter_Collimator, sizeof(float) * 80000);

	float* h_parameter_Detector;
	cudaCheckError(cudaMallocHost(&h_parameter_Detector, sizeof(float) * 80000));
	memcpy(h_parameter_Detector, parameter_Detector, sizeof(float) * 80000);

	float* h_parameter_Image;
	cudaCheckError(cudaMallocHost(&h_parameter_Image, sizeof(float) * 100));
	memcpy(h_parameter_Image, parameter_Image, sizeof(float) * 100);

	float* h_parameter_Physics;
	cudaCheckError(cudaMallocHost(&h_parameter_Physics, sizeof(float) * 100));
	memcpy(h_parameter_Physics, parameter_Physics, sizeof(float) * 100);

	float* h_PE_SysMat;
	cudaCheckError(cudaMallocHost(&h_PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin));
	memcpy(h_PE_SysMat, PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin);

	// Allocate memory on device
	float* deviceMatrix_crystal;
	cudaCheckError(cudaMalloc(&deviceMatrix_crystal, sizeof(float) * numProjectionSingle * numImagebin));
	cudaCheckError(cudaMemset(deviceMatrix_crystal, 0, sizeof(float) * numProjectionSingle * numImagebin));

	float* deviceMatrix_collimator;
	cudaCheckError(cudaMalloc(&deviceMatrix_collimator, sizeof(float) * numProjectionSingle * numImagebin));
	cudaCheckError(cudaMemset(deviceMatrix_collimator, 0, sizeof(float) * numProjectionSingle * numImagebin));

	float* devicePEMatrix;
	cudaCheckError(cudaMalloc(&devicePEMatrix, sizeof(float) * numProjectionSingle * numImagebin));
	cudaCheckError(cudaMemcpyAsync(devicePEMatrix, h_PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyHostToDevice, stream));

	float* deviceparameter_Collimator;
	cudaCheckError(cudaMalloc(&deviceparameter_Collimator, sizeof(float) * 80000));
	cudaCheckError(cudaMemcpyAsync(deviceparameter_Collimator, h_parameter_Collimator, sizeof(float) * 80000, cudaMemcpyHostToDevice, stream));

	float* deviceparameter_Detector;
	cudaCheckError(cudaMalloc(&deviceparameter_Detector, sizeof(float) * 80000));
	cudaCheckError(cudaMemcpyAsync(deviceparameter_Detector, h_parameter_Detector, sizeof(float) * 80000, cudaMemcpyHostToDevice, stream));

	float* deviceparameter_Image;
	cudaCheckError(cudaMalloc(&deviceparameter_Image, sizeof(float) * 100));
	cudaCheckError(cudaMemcpyAsync(deviceparameter_Image, h_parameter_Image, sizeof(float) * 100, cudaMemcpyHostToDevice, stream));

	float* deviceparameter_Physics;
	cudaCheckError(cudaMalloc(&deviceparameter_Physics, sizeof(float) * 100));
	cudaCheckError(cudaMemcpyAsync(deviceparameter_Physics, h_parameter_Physics, sizeof(float) * 100, cudaMemcpyHostToDevice, stream));


	int flagCalculateCrystalGeometryRelationShip = (int)floor(parameter_Physics[8]);
	if (flagCalculateCrystalGeometryRelationShip == 1)
	{
		cout << "Initializing Crystal Geometry Relationship Calculation with numProjectionSingle = " << numProjectionSingle << endl;
		cudaCheckError(cudaMalloc(&deviceGeometryRelationShip_Crystal2Crystal, array_size * sizeof(unsigned int)));
		cudaCheckError(cudaMemset(deviceGeometryRelationShip_Crystal2Crystal, 0, array_size * sizeof(unsigned int)));

		long long total_threads = static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);
		int threads_per_block = 256;
		long long int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

		cout << "Launching geometryRelationShip_Crystal2Crystal kernel with "
			<< blocks_per_grid << " blocks of " << threads_per_block << " threads each." << endl;
		
		geometryRelationShip_Crystal2Crystal << <blocks_per_grid, threads_per_block, 0, stream >> > (
			deviceGeometryRelationShip_Crystal2Crystal,
			deviceparameter_Detector);

		cudaCheckError(cudaStreamSynchronize(stream));
		cudaCheckError(cudaMemcpyAsync(hostGeometryRelationShip_Crystal2Crystal, deviceGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int) * array_size, cudaMemcpyDeviceToHost, stream));

		cudaCheckError(cudaStreamSynchronize(stream));

		cudaCheckError(cudaGetLastError());
		cudaCheckError(cudaDeviceSynchronize());
		cudaFree(deviceGeometryRelationShip_Crystal2Crystal);

		FILE* fp4;
		fp4 = fopen(FnameGeoCrystal, "wb+");
		if (fp4 == 0) { fprintf(stderr, "error: cannot create %s\n", FnameGeoCrystal); exit(EXIT_FAILURE); }
		if (fwrite(hostGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int), array_size, fp4) != (size_t)array_size) { fprintf(stderr, "error: writing %s failed\n", FnameGeoCrystal); exit(EXIT_FAILURE); }
		fclose(fp4);
		
		cout << "Kernel geometryRelationShip_Crystal2Crystal executed." << endl;

	}
	else 
	{		
		auto start_ioGeoCrystal = std::chrono::high_resolution_clock::now();
		FILE* fp4;
		fp4 = fopen(FnameGeoCrystal, "rb");
		if (fp4 == 0) { fprintf(stderr, "error: cannot open the geometry bitmap %s\n", FnameGeoCrystal); exit(EXIT_FAILURE); }
		size_t gotBits = fread(hostGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int), array_size, fp4);
		bool extraBits = fgetc(fp4) != EOF;
		fclose(fp4);
		if (gotBits != (size_t)array_size || extraBits) { fprintf(stderr, "error: %s does not match this geometry (%lld words expected); recompute it (Param_Physics[8] / [9] = 1)\n", FnameGeoCrystal, (long long)array_size); exit(EXIT_FAILURE); }
		auto end_ioGeoCrystal = std::chrono::high_resolution_clock::now();
		auto duration_ioGeoCrystal = std::chrono::duration_cast<std::chrono::milliseconds>(end_ioGeoCrystal - start_ioGeoCrystal);
		cout << "Time of io Crystal Geometry System Matrix: " << duration_ioGeoCrystal.count() << " ms" << endl;

	}
	
	cudaCheckError(cudaMalloc(&deviceGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int) * array_size));
	cudaCheckError(cudaMemcpyAsync(deviceGeometryRelationShip_Crystal2Crystal, hostGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int) * array_size, cudaMemcpyHostToDevice, stream));

	dim3 blockSize(16, 16, 1); 
	dim3 gridSize(
		(numImagebin + 15) / 16,                 // voxels on grid x (gridDim.y is limited to 65535)
		(numProjectionSingle + 15) / 16,
		(numProjectionSingle + 0) / 1
	);

	cout << "########################" << endl;
	cout << "numProjectionSingle = " << numProjectionSingle << endl;
	cout << "numImagebin = " << numImagebin << endl;
	cout << "gridSize.x = " << gridSize.x << endl;
	cout << "gridSize.y = " << gridSize.y << endl;
	cout << "gridSize.z = " << gridSize.z << endl;
	cout << "########################" << endl;

	auto start_crystalScatterSysMatCuda = std::chrono::high_resolution_clock::now();
	// scatter model v2: crystal table, collimator layers, hole grid and crystal neighbour lists
	float* deviceCrystalInfo = 0;
	int *deviceHashIdx = 0, *deviceHashCnt = 0, *deviceNeighbours = 0;
	std::vector<float> hostCrystalInfo;
	std::vector<int> hashIdx, hashCnt, neighbours;
	S2Col layers;
	S2Hash hash;
	if (model == 2)
	{
		s2_crystalInfo(h_parameter_Detector, h_parameter_Image[11], hostCrystalInfo);
		cudaCheckError(cudaMalloc(&deviceCrystalInfo, sizeof(float) * hostCrystalInfo.size()));
		cudaCheckError(cudaMemcpy(deviceCrystalInfo, hostCrystalInfo.data(), sizeof(float) * hostCrystalInfo.size(), cudaMemcpyHostToDevice));
		layers = s2_layers(h_parameter_Collimator, h_parameter_Image[11]);
		s2_holeHash(h_parameter_Collimator, layers, hashIdx, hashCnt, hash);
		s2_neighbours(hostCrystalInfo, numProjectionSingle, neighbours);
		cudaCheckError(cudaMalloc(&deviceHashIdx, sizeof(int) * hashIdx.size()));
		cudaCheckError(cudaMemcpy(deviceHashIdx, hashIdx.data(), sizeof(int) * hashIdx.size(), cudaMemcpyHostToDevice));
		cudaCheckError(cudaMalloc(&deviceHashCnt, sizeof(int) * hashCnt.size()));
		cudaCheckError(cudaMemcpy(deviceHashCnt, hashCnt.data(), sizeof(int) * hashCnt.size(), cudaMemcpyHostToDevice));
		cudaCheckError(cudaMalloc(&deviceNeighbours, sizeof(int) * neighbours.size()));
		cudaCheckError(cudaMemcpy(deviceNeighbours, neighbours.data(), sizeof(int) * neighbours.size(), cudaMemcpyHostToDevice));
		hash.idx = deviceHashIdx;
		hash.cnt = deviceHashCnt;
		float peMin = s2_peMin(h_PE_SysMat, (size_t)numProjectionSingle * numImagebin);
		cudaCheckError(cudaStreamSynchronize(stream));
		cout << "Kernel crystalScatterV2 Launched (scatter model v2)" << endl;
		crystalScatterV2 <<<gridSize, blockSize >>> (deviceMatrix_crystal, deviceparameter_Collimator, deviceCrystalInfo,
			deviceparameter_Detector, deviceparameter_Image, deviceparameter_Physics, devicePEMatrix,
			deviceGeometryRelationShip_Crystal2Crystal, layers, hash, deviceNeighbours, numProjectionSingle, numImagebin, peMin);
	}
	else
	{
	cout << "Kernel crystalScatterSysMatCuda Launched " << endl;
	crystalScatterSysMatCuda <<<gridSize, blockSize >>> (
		deviceMatrix_crystal,
		deviceparameter_Detector,
		deviceparameter_Image,
		deviceparameter_Physics,
		devicePEMatrix,
		deviceGeometryRelationShip_Crystal2Crystal,
		numProjectionSingle,
		numImagebin);
	}
	cudaCheckError(cudaGetLastError());
	
	float* h_Crystal_SysMat;
	cudaCheckError(cudaMallocHost(&h_Crystal_SysMat, sizeof(float) * numProjectionSingle * numImagebin));
	cudaCheckError(cudaMemcpyAsync(h_Crystal_SysMat, deviceMatrix_crystal, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyDeviceToHost, stream));
	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaStreamSynchronize(stream));
	
	/* // Write CrystalScatter_SysMat.sysmat for debug
	char Fnametmp[2048];
	sprintf(Fnametmp, "CrystalScatter_SysMat.sysmat");
	FILE* fptmp;
	fptmp = fopen(Fnametmp, "wb+");
	if (fptmp == 0) { puts("error"); exit(EXIT_FAILURE); }
	fwrite(h_Crystal_SysMat, sizeof(float), numProjectionSingle * numImagebin, fptmp);
	fclose(fptmp);
	cout << "Write Results of crystalScatterSysMatCuda function finished " << endl;
	*/


	auto end_crystalScatterSysMatCuda = std::chrono::high_resolution_clock::now();
	auto duration_crystalScatterSysMatCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end_crystalScatterSysMatCuda - start_crystalScatterSysMatCuda);
	cout << "Time of crystalScatterSysMatCuda function: " << duration_crystalScatterSysMatCuda.count()/1000.0/60.0 << " min" << endl;
	cudaCheckError(cudaStreamSynchronize(stream));
	cudaFreeHost(h_PE_SysMat);
	cudaFree(devicePEMatrix);
	cudaFree(deviceMatrix_crystal);
	cudaFree(deviceGeometryRelationShip_Crystal2Crystal);
	delete[] hostGeometryRelationShip_Crystal2Crystal;
	cout << "########################" << endl;

	///////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////// Promary Compton Scatter between Collimator and Detector Crystal /////////////////
	float* h_Collimator_SysMat;
	cudaCheckError(cudaMallocHost(&h_Collimator_SysMat, sizeof(float) * numProjectionSingle * numImagebin));
	unsigned int* deviceGeometryRelationShip_Collimator2Crystal = 0;
	unsigned int* hostGeometryRelationShip_Collimator2Crystal = 0;
	auto start_CollimatorScatterSysMatCuda = std::chrono::high_resolution_clock::now();
	if (model == 2)
	{
		dim3 blockSize_collimator(16, 16, 1);
		cout << "Kernel collimatorScatterShellV2 / collimatorScatterBulkV2 Launched (" << layers.n << " layers, "
			<< layers.totalHoles << " holes, scatter model v2)" << endl;
		for (int off = 0; off < layers.totalHoles; off += 65535)
		{
			int n = min(65535, layers.totalHoles - off);
			dim3 gridShell((numImagebin + 15) / 16, (numProjectionSingle + 15) / 16, n);
			collimatorScatterShellV2 <<<gridShell, blockSize_collimator >>> (deviceMatrix_collimator, deviceparameter_Collimator,
				deviceCrystalInfo, deviceparameter_Detector, deviceparameter_Image, deviceparameter_Physics, layers, hash,
				deviceNeighbours, numProjectionSingle, numImagebin, off, n);
		}
		if (layers.n > 0)
		{
			dim3 gridBulk((numImagebin + 15) / 16, (numProjectionSingle + 15) / 16, layers.n);
			collimatorScatterBulkV2 <<<gridBulk, blockSize_collimator >>> (deviceMatrix_collimator, deviceparameter_Collimator,
				deviceCrystalInfo, deviceparameter_Detector, deviceparameter_Image, deviceparameter_Physics, layers, hash,
				deviceNeighbours, numProjectionSingle, numImagebin);
		}
		cudaCheckError(cudaGetLastError());
		cudaCheckError(cudaMemcpy(h_Collimator_SysMat, deviceMatrix_collimator, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyDeviceToHost));
	}
	else
	{
	// legacy model (kept as it was, not re-indented)

	////-------------- Geometry RelationShip between Collimator Hole and Crystals -----------------////
	// Calculate geometry relationship between each crystal and collimator holes, if the crystal_k is cross the line of 
	// collimator_i and crystal_j, then deviceGeometryRelationShip[i,j,k]=1,else =0.
	// The array deviceGeometryRelationShip is a bitmap, the array is stored by bit, the index follows the belowing rules:
	int numCollimator_Holes = (int)floor(parameter_Collimator[10]+0.01f);
	total_bits = static_cast<long long>(numProjectionSingle)
		* static_cast<long long>(numProjectionSingle)
		* static_cast<long long>(numCollimator_Holes);
	array_size = (total_bits + bits_per_word - 1) / bits_per_word;


	cout << "Total bits: " << total_bits << endl;
	cout << "Array size (unsigned int): " << array_size << endl;

	hostGeometryRelationShip_Collimator2Crystal = new unsigned int[array_size]();

	int flagCalculateCollimatorGeometryRelationShip = (int)floor(parameter_Physics[9]+0.001f);
	if (flagCalculateCollimatorGeometryRelationShip == 1)
	{
		cout << "Initializing Collimator GeometryRelationship Calculation with numProjectionSingle = " << numProjectionSingle << " and numCollimatorHoles = " << numCollimator_Holes << endl;
		cudaCheckError(cudaMalloc(&deviceGeometryRelationShip_Collimator2Crystal, array_size * sizeof(unsigned int)));
		cudaCheckError(cudaMemset(deviceGeometryRelationShip_Collimator2Crystal, 0, array_size * sizeof(unsigned int)));

		long long total_threads = static_cast<long long>(numCollimator_Holes) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);
		int threads_per_block = 256;
		long long int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

		cout << "Launching geometryRelationShip_Collimator2Crystal kernel with "
			<< blocks_per_grid << " blocks of " << threads_per_block << " threads each." << endl;

		geometryRelationShip_Collimator2Crystal << <blocks_per_grid, threads_per_block, 0, stream >> > (
			deviceGeometryRelationShip_Collimator2Crystal,
			deviceparameter_Detector,
			deviceparameter_Collimator);

		cudaCheckError(cudaStreamSynchronize(stream));

		cudaCheckError(cudaMemcpyAsync(hostGeometryRelationShip_Collimator2Crystal, deviceGeometryRelationShip_Collimator2Crystal, sizeof(unsigned int) * array_size, cudaMemcpyDeviceToHost, stream));

		cudaCheckError(cudaStreamSynchronize(stream));

		cudaCheckError(cudaGetLastError());
		cudaCheckError(cudaDeviceSynchronize());
		cudaFree(deviceGeometryRelationShip_Collimator2Crystal);

		FILE* fp4;
		fp4 = fopen(FnameGeoCollimator, "wb+");
		if (fp4 == 0) { fprintf(stderr, "error: cannot create %s\n", FnameGeoCollimator); exit(EXIT_FAILURE); }
		if (fwrite(hostGeometryRelationShip_Collimator2Crystal, sizeof(unsigned int), array_size, fp4) != (size_t)array_size) { fprintf(stderr, "error: writing %s failed\n", FnameGeoCollimator); exit(EXIT_FAILURE); }
		fclose(fp4);

		cout << "Kernel geometryRelationShip_Collimator2Crystal executed." << endl;

	}
	else
	{
		auto start_ioGeoCollimator = std::chrono::high_resolution_clock::now();
		FILE* fp4;
		fp4 = fopen(FnameGeoCollimator, "rb");
		if (fp4 == 0) { fprintf(stderr, "error: cannot open the geometry bitmap %s\n", FnameGeoCollimator); exit(EXIT_FAILURE); }
		size_t gotBits = fread(hostGeometryRelationShip_Collimator2Crystal, sizeof(unsigned int), array_size, fp4);
		bool extraBits = fgetc(fp4) != EOF;
		fclose(fp4);
		if (gotBits != (size_t)array_size || extraBits) { fprintf(stderr, "error: %s does not match this geometry (%lld words expected); recompute it (Param_Physics[8] / [9] = 1)\n", FnameGeoCollimator, (long long)array_size); exit(EXIT_FAILURE); }
		auto end_ioGeoCollimator = std::chrono::high_resolution_clock::now();
		auto duration_ioGeoCollimator = std::chrono::duration_cast<std::chrono::milliseconds>(end_ioGeoCollimator - start_ioGeoCollimator);
		cout << "Time of io GeoCollimator System Matrix: " << duration_ioGeoCollimator.count() << " ms" << endl;

	}

	// Copy to device
	cudaCheckError(cudaMalloc(&deviceGeometryRelationShip_Collimator2Crystal, sizeof(unsigned int) * array_size));
	cudaCheckError(cudaMemcpyAsync(deviceGeometryRelationShip_Collimator2Crystal, hostGeometryRelationShip_Collimator2Crystal, sizeof(unsigned int) * array_size, cudaMemcpyHostToDevice, stream));
	cudaCheckError(cudaStreamSynchronize(stream));

	dim3 blockSize_collimator(16, 16, 1);
	dim3 gridSize_collimator(
		(numImagebin + 15) / 16,
		(numProjectionSingle + 15) / 16,
		(numCollimator_Holes + 0) / 1
	);

	cout << "########################" << endl;
	cout << "numProjectionSingle = " << numProjectionSingle << endl;
	cout << "numImagebin = " << numImagebin << endl;
	cout << "gridSize.x = " << gridSize_collimator.x << endl;
	cout << "gridSize.y = " << gridSize_collimator.y << endl;
	cout << "gridSize.z = " << gridSize_collimator.z << endl;
	cout << "########################" << endl;

	cout << "Kernel collimatorScatterSysMatCuda Launched " << endl;
	
	collimatorScatterSysMatCuda <<<gridSize_collimator, blockSize_collimator >>> (
		deviceMatrix_collimator,
		deviceparameter_Collimator,
		deviceparameter_Detector,
		deviceparameter_Image,
		deviceparameter_Physics,
		deviceGeometryRelationShip_Collimator2Crystal,
		numProjectionSingle,
		numImagebin);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "Kernel collimatorScatterSysMatCuda launch failed: " << cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
	cudaCheckError(cudaMemcpyAsync(h_Collimator_SysMat, deviceMatrix_collimator, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyDeviceToHost, stream));
	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaStreamSynchronize(stream));
	
	/* // Write CollimatorScatter_SysMat.sysmat for debug
	sprintf(Fnametmp, "CollimatorScatter_SysMat.sysmat");
	fptmp = fopen(Fnametmp, "wb+");
	if (fptmp == 0) { puts("error"); exit(EXIT_FAILURE); }
	fwrite(h_Collimator_SysMat, sizeof(float), numProjectionSingle * numImagebin, fptmp);
	fclose(fptmp);
	cout << "Write Results of CollimatorScatterSysMatCuda function finished " << endl;
	*/
	}
	auto end_CollimatorScatterSysMatCuda = std::chrono::high_resolution_clock::now();
	auto duration_CollimatorScatterSysMatCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end_CollimatorScatterSysMatCuda - start_CollimatorScatterSysMatCuda);
	cout << "Time of  collimatorScatterSysMatCuda function: " << duration_CollimatorScatterSysMatCuda.count() / 1000.0 / 60.0 << " min" << endl;

	cout << "########################" << endl;
	for (long long i = 0; i < (long long)numProjectionSingle * numImagebin; i++) // one rotation per call (dst already points at it)
	{
		dst[i] = h_Collimator_SysMat[i] + h_Crystal_SysMat[i];
	}
	if (dstCrystal) memcpy(dstCrystal, h_Crystal_SysMat, sizeof(float) * numProjectionSingle * numImagebin);
	if (dstCollimator) memcpy(dstCollimator, h_Collimator_SysMat, sizeof(float) * numProjectionSingle * numImagebin);
	// Release Sources
	cudaFreeHost(h_parameter_Collimator);
	cudaFreeHost(h_parameter_Detector);
	cudaFreeHost(h_parameter_Image);
	cudaFreeHost(h_parameter_Physics);
	cudaFreeHost(h_Collimator_SysMat);
	cudaFreeHost(h_Crystal_SysMat);

	cudaFree(deviceparameter_Collimator);
	cudaFree(deviceparameter_Detector);
	cudaFree(deviceparameter_Image);
	cudaFree(deviceparameter_Physics);
	cudaFree(deviceMatrix_collimator);
	
	cudaFree(deviceGeometryRelationShip_Collimator2Crystal);
	cudaFree(deviceCrystalInfo);
	cudaFree(deviceHashIdx);
	cudaFree(deviceHashCnt);
	cudaFree(deviceNeighbours);

	delete[] hostGeometryRelationShip_Collimator2Crystal;
	cudaStreamDestroy(stream);

	return numImagebin;
}
