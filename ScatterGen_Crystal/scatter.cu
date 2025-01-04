// Generate System Matrix on GPU with inter crystal primary compton scatter
// last modified: 2024/12/21

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <chrono> 

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

	float total_cross_section = 0.0f;
	int numSteps = (int)((theta_high - theta_low) / DeltaTheta);

	for (int i = 0; i <= numSteps; ++i) {
		float theta = theta_low + i * DeltaTheta; 
		total_cross_section += diffComptonSection(theta, E0) * sin(theta) * DeltaTheta;  
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

	long long int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < 0 || row > numProjectionSingle - 1) { return; }
	long long int col = blockIdx.y * blockDim.y + threadIdx.y;
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
	int PESysMat_index = numImagebin * id_Detector + ImageVoxel_index;
	float prob_Compton_othercrystal = devicePESysMat[PESysMat_index] * coeff_detector_compton / coeff_detector_pe;

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
						float yDetector_self = -thicknessDetectorCrystal / 2.0f + (float)(NumY + 0.5f) / (float)divideY * thicknessDetectorCrystal;

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
						float energy_resolution_detector_scatterphoton = energy_resolution_detector_targetPE * sqrt(scatterEnergy / target_PE_Energy);

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
						atomicAdd(&dst[dstIndex], prob_Compton_othercrystal * Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f - exp(-length_absorp * absorp_coeff_detector_total)) * absorp_coeff_detector_pe / absorp_coeff_detector_total);
					}
				}
			}
			//atomicAdd(&dst[dstIndex], prob_Compton_to_detectionCrystal);
		}

	}


}


__global__ void geometryRelationShip_Crystal2Crystal(unsigned int* dst_relation_crystal2crystal, float* deviceparameter_Detector)
{
	// clculate global index
	int numProjectionSingle = (int)deviceparameter_Detector[0];
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
	float thicknessDetector_j = deviceparameter_Detector[12 * j + 5];
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

	if (flagcross == 1)
	{
		atomicOr(&dst_relation_crystal2crystal[word_idx], 1U << bit_offset);
	}
}



int scatter(float* parameter_Detector, float* parameter_Image, float* parameter_Physics,float* PE_SysMat,const char* FnameGeo, float* dst, int cuda_id)
{

	cout << "Get into scatter function" << endl;
	float _float_numPSFImageVoxelX = parameter_Image[0];
	float _float_numPSFImageVoxelY = parameter_Image[1];
	float _float_numPSFImageVoxelZ = parameter_Image[2];

	int numPSFImageVoxelX = (int)floor(_float_numPSFImageVoxelX);
	int numPSFImageVoxelY = (int)floor(_float_numPSFImageVoxelY);
	int numPSFImageVoxelZ = (int)floor(_float_numPSFImageVoxelZ);

	int numProjectionSingle = (int)floor(parameter_Detector[0]+0.0001f);
	int numImagebin = numPSFImageVoxelX * numPSFImageVoxelY * numPSFImageVoxelZ;

	
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
		cudaSetDevice(cuda_id);
		cout << "Set Device to Device " <<cuda_id<< endl;
	}


	// Calculate geometry relationship between each crystal, if the crystal_k is cross the line of 
	// crystal_i and crystal_j, then deviceGeometryRelationShip[i,j,k]=1,else =0.
	// The array deviceGeometryRelationShip is a bitmap, the stored by bit, the index follows the below ruls:

	const int unique_pairs = numProjectionSingle * (numProjectionSingle + 1) / 2;
	const long long total_bits = static_cast<long long>(unique_pairs) * numProjectionSingle;
	const int bits_per_word = 32;
	const long long array_size = (total_bits + bits_per_word - 1) / bits_per_word;


	cout << "Total unique pairs (i <= j): " << unique_pairs << endl;
	cout << "Total bits: " << total_bits << endl;
	cout << "Array size (unsigned int): " << array_size << endl;

	unsigned int* deviceGeometryRelationShip_Crystal2Crystal;
	unsigned int* hostGeometryRelationShip_Crystal2Crystal = new unsigned int[array_size]();


	cudaStream_t stream;
	cudaStreamCreate(&stream);


	float* h_parameter_Detector;
	cudaMallocHost(&h_parameter_Detector, sizeof(float) * 80000);
	memcpy(h_parameter_Detector, parameter_Detector, sizeof(float) * 80000);

	float* h_parameter_Image;
	cudaMallocHost(&h_parameter_Image, sizeof(float) * 100);
	memcpy(h_parameter_Image, parameter_Image, sizeof(float) * 100);

	float* h_parameter_Physics;
	cudaMallocHost(&h_parameter_Physics, sizeof(float) * 100);
	memcpy(h_parameter_Physics, parameter_Physics, sizeof(float) * 100);

	float* h_PE_SysMat;
	cudaMallocHost(&h_PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin);
	memcpy(h_PE_SysMat, PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin);

	// 分配设备内存
	float* deviceMatrix;
	cudaMalloc(&deviceMatrix, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMemset(deviceMatrix, 0, sizeof(float) * numProjectionSingle * numImagebin);

	float* devicePEMatrix;
	cudaMalloc(&devicePEMatrix, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMemcpyAsync(devicePEMatrix, h_PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyHostToDevice, stream);


	float* deviceparameter_Detector;
	cudaMalloc(&deviceparameter_Detector, sizeof(float) * 80000);
	cudaMemcpyAsync(deviceparameter_Detector, h_parameter_Detector, sizeof(float) * 80000, cudaMemcpyHostToDevice, stream);

	float* deviceparameter_Image;
	cudaMalloc(&deviceparameter_Image, sizeof(float) * 100);
	cudaMemcpyAsync(deviceparameter_Image, h_parameter_Image, sizeof(float) * 100, cudaMemcpyHostToDevice, stream);

	float* deviceparameter_Physics;
	cudaMalloc(&deviceparameter_Physics, sizeof(float) * 100);
	cudaMemcpyAsync(deviceparameter_Physics, h_parameter_Physics, sizeof(float) * 100, cudaMemcpyHostToDevice, stream);




	int flagCalculateGeometryRelationShip = (int)floor(parameter_Physics[8]);
	if (flagCalculateGeometryRelationShip == 1)
	{
		cout << "Initializing GeometryRelationship Calculation with numProjectionSingle = " << numProjectionSingle << endl;
		cudaCheckError(cudaMalloc(&deviceGeometryRelationShip_Crystal2Crystal, array_size * sizeof(unsigned int)));
		cudaCheckError(cudaMemset(deviceGeometryRelationShip_Crystal2Crystal, 0, array_size * sizeof(unsigned int)));

		long long total_threads = static_cast<long long>(numProjectionSingle) * numProjectionSingle * numProjectionSingle;
		int threads_per_block = 256;
		long long int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

		cout << "Launching geometryRelationShip_Crystal2Crystal kernel with "
			<< blocks_per_grid << " blocks of " << threads_per_block << " threads each." << endl;
		
		geometryRelationShip_Crystal2Crystal << <blocks_per_grid, threads_per_block, 0, stream >> > (
			deviceGeometryRelationShip_Crystal2Crystal,
			deviceparameter_Detector);

		cudaStreamSynchronize(stream);

		cudaMemcpyAsync(hostGeometryRelationShip_Crystal2Crystal, deviceGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int) * array_size, cudaMemcpyDeviceToHost, stream);


		cudaStreamSynchronize(stream);

		cudaCheckError(cudaGetLastError());
		cudaCheckError(cudaDeviceSynchronize());
		cudaFree(deviceGeometryRelationShip_Crystal2Crystal);

		FILE* fp4;
		fp4 = fopen(FnameGeo, "wb+");
		if (fp4 == 0) { puts("error"); exit(0); }
		fwrite(hostGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int), array_size, fp4);
		fclose(fp4);
		
		cout << "Kernel geometryRelationShip_Crystal2Crystal executed." << endl;

	}
	else 
	{		
		auto start_ioGeo = std::chrono::high_resolution_clock::now();
		FILE* fp4;
		fp4 = fopen(FnameGeo, "rb");
		if (fp4 == 0) { puts("error"); exit(0); }
		fread(hostGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int), array_size, fp4);
		fclose(fp4);
		auto end_ioGeo = std::chrono::high_resolution_clock::now();
		auto duration_ioGeo = std::chrono::duration_cast<std::chrono::milliseconds>(end_ioGeo - start_ioGeo);
		cout << "Time of io Geo System Matrix: " << duration_ioGeo.count() << " ms" << endl;

	}
	

	cudaMalloc(&deviceGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int) * array_size);
	cudaMemcpyAsync(deviceGeometryRelationShip_Crystal2Crystal, hostGeometryRelationShip_Crystal2Crystal, sizeof(unsigned int) * array_size, cudaMemcpyHostToDevice, stream);

	dim3 blockSize(16, 16, 1); 
	dim3 gridSize(
		(numProjectionSingle + 15) / 16, 
		(numImagebin + 15) / 16,         
		(numProjectionSingle + 0) / 1
	);

	cout << "########################" << endl;
	cout << "numProjectionSingle = " << numProjectionSingle << endl;
	cout << "numImagebin = " << numImagebin << endl;
	cout << "gridSize.x = " << gridSize.x << endl;
	cout << "gridSize.y = " << gridSize.y << endl;
	cout << "gridSize.z = " << gridSize.z << endl;
	cout << "########################" << endl;

	auto start_scatterSysMatCuda = std::chrono::high_resolution_clock::now();
	cout << "Kernel scatterSysMatCuda Launched " << endl;
	crystalScatterSysMatCuda <<<gridSize, blockSize >>> (
		deviceMatrix,
		deviceparameter_Detector,
		deviceparameter_Image,
		deviceparameter_Physics,
		devicePEMatrix,
		deviceGeometryRelationShip_Crystal2Crystal,
		numProjectionSingle,
		numImagebin);

	cudaMemcpyAsync(dst, deviceMatrix, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyDeviceToHost, stream);
	auto end_scatterSysMatCuda = std::chrono::high_resolution_clock::now();
	auto duration_scatterSysMatCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end_scatterSysMatCuda - start_scatterSysMatCuda);
	cout << "Time of io scatterSysMatCuda function: " << duration_scatterSysMatCuda.count()/1000.0/60.0 << " min" << endl;


	cudaStreamSynchronize(stream);
	cout << "########################" << endl;



	cudaFreeHost(h_parameter_Detector);
	cudaFreeHost(h_parameter_Image);
	cudaFreeHost(h_parameter_Physics);
	cudaFreeHost(h_PE_SysMat);

	cudaFree(deviceparameter_Detector);
	cudaFree(deviceparameter_Image);
	cudaFree(deviceparameter_Physics);
	cudaFree(deviceMatrix);
	cudaFree(devicePEMatrix);
	cudaFree(deviceGeometryRelationShip_Crystal2Crystal);

	delete[] hostGeometryRelationShip_Crystal2Crystal;


	cudaStreamDestroy(stream);

	return numImagebin;
}
