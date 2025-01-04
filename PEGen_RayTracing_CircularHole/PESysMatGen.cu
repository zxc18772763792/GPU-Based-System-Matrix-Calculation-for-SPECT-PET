// 使用GPU生成PE系统矩阵
// �޸�ʱ��2024/11/24

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include "PESysMatGen.h"
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>


#define max(a,b) ((a>=b)?a:b) 
#define min(a,b) ((a<=b)?a:b)

using namespace std;

__device__ float length_box_ray(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x1_box, float y1_box, float z1_box, float x2_box, float y2_box, float z2_box)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Down box position: (x1_box, y1_box, z1_box)
	// Right-Up box position: (x2_box, y2_box, z2_box)
	float eps = 0.001f;

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps & fabs(z_out - z_in) < eps)
	{
		return 0.000f;
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
				return fabs(x2_box - x1_box);
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


// ����ú���������GPU��CPU������
__global__ void photodetectorCudaMe(float* dst,
	float* deviceparameter_Collimator,
	float* deviceparameter_Detector,
	float* deviceparameter_Image,
	int numProjectionSingle,
	int numImagebin)

{

	//解析参数
	float _float_numCollimatorLayer = deviceparameter_Collimator[0];
	float _float_numDetectorBins = deviceparameter_Detector[0];
	
	int numCollimatorLayer = (int)floor(_float_numCollimatorLayer + 0.000001f);
	int numDetectorbins = (int)floor(_float_numDetectorBins + 0.000001f);

	float _float_FOV2Collimator = deviceparameter_Image[11];
	// 准直器参数



	float _float_numCollimatorHoles[100];
	int numCollimatorHoles[100];
	//float _float_widthCollimatorLayers[100];
	//float _float_heightCollimatorLayers[100];
	float _float_thicknessCollimatorLayers[100];
	//float _float_coeffCollimatorLayers[100];

	int numCollimatorHoles_tot = 0;
	for (unsigned int id_CollimatorLayer = 0; id_CollimatorLayer < numCollimatorLayer; id_CollimatorLayer++)
	{
		_float_numCollimatorHoles[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0];
		numCollimatorHoles[id_CollimatorLayer] = (int)floor(_float_numCollimatorHoles[id_CollimatorLayer]);
		//_float_widthCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1];
		_float_thicknessCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2];
		//_float_heightCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3];
		//_float_coeffCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5];
		numCollimatorHoles_tot = numCollimatorHoles_tot + numCollimatorHoles[id_CollimatorLayer];

	}
	


	// 探测器参数
	float _float_numImageVoxelX = deviceparameter_Image[0];
	float _float_numImageVoxelY = deviceparameter_Image[1];
	float _float_numImageVoxelZ = deviceparameter_Image[2];
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

	int numImageVoxelX = (int)floor(_float_numImageVoxelX);
	int numImageVoxelY = (int)floor(_float_numImageVoxelY);
	int numImageVoxelZ = (int)floor(_float_numImageVoxelZ);



	//线程块编号 blockDim, 线程编号threadIdx
	//检验该线程是否在分配范围内

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < 0 || row > numProjectionSingle - 1) { return; }
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < 0 || col > numImagebin - 1) { return; }


	//PSF
	//int dstIndex = col * numProjectionSingle + row;
	//DRF
	int dstIndex = row * numImagebin + col;

	//定义探测器二维索引(zxy)
	unsigned int idxDetector = row;



	//定义像素二维索引
	int idxImageVoxelZ = col / (numImageVoxelY * numImageVoxelX);
	col = col % (numImageVoxelY * numImageVoxelX);
	int idxImageVoxelY = col / numImageVoxelX;
	int idxImageVoxelX = col % numImageVoxelX;

	//每个探测器划分为8*8*16个探测单元
	const unsigned int divideX = 8, divideY = 16, divideZ = 8;


	///////////////////////////////////////// Image Rotation Shift Parameters /////////////////////////////////////
	float xImage = (idxImageVoxelX - numImageVoxelX / 2.0f + 0.5f) * _float_widthImageVoxelX;
	float yImage = (idxImageVoxelY - numImageVoxelY / 2.0f + 0.5f) * _float_widthImageVoxelY;
	float zImage = (idxImageVoxelZ - numImageVoxelZ / 2.0f + 0.5f) * _float_widthImageVoxelZ;

	xImage = xImage + shiftFOVX_physics;
	yImage = yImage + shiftFOVY_physics;
	zImage = zImage + shiftFOVZ_physics;

	float xImage_rot = xImage * cos(RotationAngle) - yImage * sin(RotationAngle);
	float yImage_rot = xImage * sin(RotationAngle) + yImage * cos(RotationAngle);
	
	xImage = xImage_rot;
	yImage = yImage_rot;
	//int ImageVoxel_index = idxImageVoxelX + idxImageVoxelY * numImageVoxelX + idxImageVoxelZ * numImageVoxelY * numImageVoxelX;

	// All variables without a suffix are in the real-world physical coordinate system
	// All parameters with 'self' suffix are in the detector crystal coordinate system
	float xDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 1];
	float yDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 2] + _float_FOV2Collimator;
	float zDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 3];
	float rotationAngel_DetectorCrystal = deviceparameter_Detector[12 * idxDetector + 11];
	float widthDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 4];
	float heightDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 6];
	float thicknessDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 5];

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


	///////////////////////////////////////// CalCulation Cross Elements ///////////////////////////////////////
	float coeff_detector_total = deviceparameter_Detector[idxDetector * 12 + 7];
	float coeff_detector_pe = deviceparameter_Detector[idxDetector * 12 + 8];
	//float coeff_detector_compton = deviceparameter_Detector[idxDetector * 12 + 9];
	
	//计算探测器微元单位和编码孔在像素中心球面上的投影，缩小遍历范围
	unsigned int numcrossDetector = 0;
	unsigned int Cross_Detector[3000];
	for (unsigned int i = 0; i < 3000; ++i)
	{
		Cross_Detector[i] = 0;
	}

	unsigned int numcrossCollimatorHole[10];
	unsigned int Cross_CollimatorHole[1000][10];
	for (unsigned int id_CollimatorLayer = 0; id_CollimatorLayer < numCollimatorLayer; id_CollimatorLayer++)
	{
		numcrossCollimatorHole[id_CollimatorLayer] = 0;
		for (unsigned int i = 0; i < 1000; i++)
		{
			Cross_CollimatorHole[i][id_CollimatorLayer] = 0;
		}
	}


	float crit_self_radius = 0.0f;

	if (coeff_detector_total > 0.01f)
	{

		float distance = sqrt((yDetectorCrystalCenter - yImage) * (yDetectorCrystalCenter - yImage) + (xDetectorCrystalCenter - xImage) * (xDetectorCrystalCenter - xImage) + (zDetectorCrystalCenter - zImage) * (zDetectorCrystalCenter - zImage));

		crit_self_radius = sqrt(widthDetectorCrystal * widthDetectorCrystal + heightDetectorCrystal * heightDetectorCrystal + thicknessDetectorCrystal * thicknessDetectorCrystal) / distance/2.0f;

		float xDetector = (xDetectorCrystalCenter -xImage) / distance;
		float yDetector = (yDetectorCrystalCenter -yImage) / distance;
		float zDetector = (zDetectorCrystalCenter -zImage) / distance;


		// Ohter Crystal
		for (unsigned int id_Detector = 0; id_Detector < numDetectorbins; id_Detector++)
		{
			if (id_Detector != idxDetector)
			{
				float widthDetector_other = deviceparameter_Detector[12 * id_Detector + 4];
				float heightDetector_other = deviceparameter_Detector[12* id_Detector + 6];
				float thicknessDetector_other = deviceparameter_Detector[12 * id_Detector + 5];
				
				
				float crit_other_radius = sqrt(widthDetector_other * widthDetector_other + heightDetector_other * heightDetector_other + thicknessDetector_other * thicknessDetector_other);
				float xDetector_other = deviceparameter_Detector[id_Detector * 12 + 1];
				float yDetector_other = deviceparameter_Detector[id_Detector * 12 + 2]+ _float_FOV2Collimator;
				float zDetector_other = deviceparameter_Detector[id_Detector * 12 + 3];
				float distance_other = sqrt((xImage - xDetector_other) * (xImage - xDetector_other) + (yImage - yDetector_other) * (yImage - yDetector_other) + (zImage - zDetector_other) * (zImage - zDetector_other));
				
				crit_other_radius = crit_other_radius / distance_other/2.0f;

				xDetector_other = (xDetector_other-xImage) / distance_other;
				yDetector_other = (yDetector_other - yImage) / distance_other;
				zDetector_other = (zDetector_other-zImage) / distance_other;

				if (sqrt((zDetector_other - zDetector) * (zDetector_other - zDetector) + (yDetector_other - yDetector) * (yDetector_other - yDetector) + (xDetector_other - xDetector) * (xDetector_other - xDetector)) <= crit_other_radius + crit_self_radius)
				{
					Cross_Detector[numcrossDetector] = id_Detector;
					numcrossDetector = numcrossDetector + 1;
				}

			}
		}

		// Collimator Hole
		unsigned int id_Hole = 0;
		unsigned int numcrossCollimatorHole_inLayer = 0;
		for (unsigned int m = 0; m < numCollimatorLayer; m++)
		{	
			numcrossCollimatorHole_inLayer = 0;
			for (unsigned int idxCollimator_inlayer = 0; idxCollimator_inlayer < numCollimatorHoles[m]; idxCollimator_inlayer++)
			{

				float x_cylinder = deviceparameter_Collimator[id_Hole * 9 + 100];
				float y1_cylinder = deviceparameter_Collimator[id_Hole * 9 + 101] + _float_FOV2Collimator;
				float y2_cylinder = deviceparameter_Collimator[id_Hole * 9 + 102] + _float_FOV2Collimator;
				float z_cylinder = deviceparameter_Collimator[id_Hole * 9 + 103];
				float R_cylinder = deviceparameter_Collimator[id_Hole * 9 + 104];
				float y_cylinder=(y1_cylinder + y2_cylinder) / 2.0f;

				float crit_other_radius = sqrt(R_cylinder * R_cylinder * 4 + (y1_cylinder - y2_cylinder) * (y1_cylinder - y2_cylinder)) / 2.0f;

				float distance_other = sqrt((xImage - x_cylinder) * (xImage - x_cylinder) + (yImage - y_cylinder) * (yImage - y_cylinder) + (zImage - z_cylinder) * (zImage - z_cylinder));
				crit_other_radius = crit_other_radius / distance_other;

				x_cylinder = (x_cylinder-xImage) / distance_other;
				y_cylinder = (y_cylinder-yImage) / distance_other;
				z_cylinder = (z_cylinder-zImage) / distance_other;

				if (sqrt((z_cylinder - zDetector) * (z_cylinder - zDetector) + (y_cylinder - yDetector) * (y_cylinder - yDetector) + (x_cylinder - xDetector) * (x_cylinder - xDetector)) <= crit_other_radius + crit_self_radius)
				{
					Cross_CollimatorHole[numcrossCollimatorHole_inLayer][m] = id_Hole;
					numcrossCollimatorHole_inLayer = numcrossCollimatorHole_inLayer + 1;
					
				}
				id_Hole = id_Hole + 1;
			}
			numcrossCollimatorHole[m] = numcrossCollimatorHole_inLayer;
		}

	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	float absorption_factor = 0.000f;
	float solid_angle = 0.000f;
	float detection_eff_each = 1.0f;

	dst[dstIndex] = 0.0000000000f;
	if (coeff_detector_total > 0.01f)
	{
		for (unsigned int NumZ = 0; NumZ < divideZ; NumZ++)
		{
			for (unsigned int NumX = 0; NumX < divideX; NumX++)
			{
				for (unsigned int NumY = 0; NumY < divideY; NumY++)
				{
					detection_eff_each = 1.0f;
					
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

					float distancesq = (yDetector - yImage) * (yDetector - yImage) + (xDetector - xImage) * (xDetector - xImage) + (zDetector - zImage) * (zDetector - zImage);
					float COSangle = (yDetector - yImage) /sqrt(distancesq);
					
					/////////////////////////////////  Solid Angle ////////////////////////////////
					float solid_angle_x = (thicknessDetectorCrystal * heightDetectorCrystal / (float)divideZ / (float)divideY) / (4 * M_PI * distancesq) * fabs((xDetector - xImage) * cos(rotationAngel_DetectorCrystal) - (zDetector - zImage) * sin(rotationAngel_DetectorCrystal)) /sqrt(distancesq);
					float solid_angle_z = (widthDetectorCrystal * thicknessDetectorCrystal / (float)divideX / (float)divideY) / (4 * M_PI * distancesq) * fabs((xDetector - xImage) * sin(rotationAngel_DetectorCrystal) + (zDetector - zImage) * cos(rotationAngel_DetectorCrystal)) /sqrt(distancesq);
					float solid_angle_y = (widthDetectorCrystal * heightDetectorCrystal / (float)divideX / (float)divideZ) / (4 * M_PI * distancesq) * fabs(yDetector - yImage) /sqrt(distancesq);

					//solid_angle = solid_angle_x + solid_angle_y + solid_angle_z;
					

					solid_angle_x=max(solid_angle_x , solid_angle_y);
 					solid_angle=max(solid_angle_x,solid_angle_z);
					
					
					/////////////////////////////////  Attenuation ////////////////////////////////
					float attenuation_dist = 0.000f;

					// ����Ƿ����й��߶��Ǵӱ���崰���ν��룻���ǴӴ������ľ���Ϊ������
					float testlength = 999.0f;
					float x1_box, y1_box, z1_box, x2_box, y2_box, z2_box;
					for (unsigned int m = 0; m < numCollimatorLayer; m++)
					{
						x1_box = -deviceparameter_Collimator[(m + 1) * 10 + 1] / 2.0f;
						x2_box = deviceparameter_Collimator[(m + 1) * 10 + 1] / 2.0f;

						y1_box = -deviceparameter_Collimator[(m + 1) * 10 + 2] / 2.0f + _float_FOV2Collimator + deviceparameter_Collimator[(m + 1) * 10 + 4];
						y2_box = deviceparameter_Collimator[(m + 1) * 10 + 2] / 2.0f + _float_FOV2Collimator + deviceparameter_Collimator[(m + 1) * 10 + 4];

						z1_box = -deviceparameter_Collimator[(m + 1) * 10 + 3] / 2.0f;
						z2_box = deviceparameter_Collimator[(m + 1) * 10 + 3] / 2.0f;

						testlength = length_box_ray(xImage, yImage, zImage, xDetector, yDetector, zDetector, x1_box, y1_box, z1_box, x2_box, y2_box, z2_box);
						if (testlength < 0.1f)
							attenuation_dist = attenuation_dist + 100000.0f;
					}

					// Collimator with Cicurlar Hole
					unsigned int id_Hole = 0;
					for (unsigned int m = 0; m < numCollimatorLayer; m++)
					{
						float length = 0.00000f;
						float Length_in_Collimator = _float_thicknessCollimatorLayers[m] / COSangle;
						float coeff_total_collimator = deviceparameter_Collimator[(m + 1) * 10 + 5];
						for (unsigned int idxCollimator_inlayer = 0; idxCollimator_inlayer < numcrossCollimatorHole[m]; idxCollimator_inlayer++)
						{
							id_Hole = Cross_CollimatorHole[idxCollimator_inlayer][m];
							float x_cylinder = deviceparameter_Collimator[id_Hole * 9 + 100];
							float y1_cylinder = deviceparameter_Collimator[id_Hole * 9 + 101] + _float_FOV2Collimator;
							float y2_cylinder = deviceparameter_Collimator[id_Hole * 9 + 102] + _float_FOV2Collimator;
							float z_cylinder = deviceparameter_Collimator[id_Hole * 9 + 103];
							float R_cylinder = deviceparameter_Collimator[id_Hole * 9 + 104];
							float coeff_total_cylinder = deviceparameter_Collimator[id_Hole * 9 + 105];
							//float coeff_pe_cylinder = deviceparameter_Collimator[id_Hole * 9 + 106];
							//float coeff_compton_cylinder = deviceparameter_Collimator[id_Hole * 9 + 107];
							float tmp = 0.000000f;

							tmp = length_cylinder_ray(xImage, yImage, zImage, xDetector, yDetector, zDetector, x_cylinder, y1_cylinder, y2_cylinder, z_cylinder, R_cylinder);
							length = length + tmp;
							attenuation_dist = attenuation_dist + coeff_total_cylinder * tmp;
						
						}
						if ((Length_in_Collimator - length) >= 0.00001f)
						{
							attenuation_dist = attenuation_dist + coeff_total_collimator * (Length_in_Collimator - length);
						}
							
					}
					
					
					// Ohter Crystal
					for (unsigned int ii = 0; ii < numcrossDetector; ii++)
					{
						unsigned int id_Detector_att = Cross_Detector[ii];
						float length_att = 0;

						float x_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 1];
						float y_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 2] + _float_FOV2Collimator;
						float z_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 3];
						
						float width_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 4];
						float height_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 6];
						float thickness_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 5];
						float coeff_total_att = deviceparameter_Detector[12 * id_Detector_att + 7];
						float rotationAngel_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 11];

						float xImage_Att = (xImage - x_AttCrystalCenter) * cos(-rotationAngel_AttCrystal) - (zImage - z_AttCrystalCenter) * sin(-rotationAngel_AttCrystal);
						float yImage_Att = yImage - y_AttCrystalCenter;
						float zImage_Att = (xImage - x_AttCrystalCenter) * sin(-rotationAngel_AttCrystal) + (zImage - z_AttCrystalCenter) * cos(-rotationAngel_AttCrystal);

						float x1_Att = -0.5f * width_AttCrystal;
						float x2_Att = 0.5f * width_AttCrystal;

						float y1_Att = -0.5f * thickness_AttCrystal;
						float y2_Att = 0.5f * thickness_AttCrystal;

						float z1_Att = -0.5f * height_AttCrystal;
						float z2_Att = 0.5f * height_AttCrystal;

						float xDetector_Att = (xDetector - x_AttCrystalCenter) * cos(-rotationAngel_AttCrystal) - (zDetector - z_AttCrystalCenter) * sin(-rotationAngel_AttCrystal);
						float yDetector_Att = yDetector - y_AttCrystalCenter;
						float zDetector_Att = (xDetector - x_AttCrystalCenter) * sin(-rotationAngel_AttCrystal) + (zDetector - z_AttCrystalCenter) * cos(-rotationAngel_AttCrystal);

						length_att = length_box_ray(xImage_Att, yImage_Att, zImage_Att, xDetector_Att, yDetector_Att, zDetector_Att, x1_Att, y1_Att, z1_Att, x2_Att, y2_Att, z2_Att);
						attenuation_dist = attenuation_dist + length_att * coeff_total_att;

					}

					
					// Crystal Self
					float x1_detectorunit_self = ((float)NumX / (float)divideX - 0.5f) * widthDetectorCrystal;
					float x2_detectorunit_self = (((float)NumX + 1.0f) / (float)divideX - 0.5f) * widthDetectorCrystal;

					float y1_detectorunit_self = ((float)NumY / (float)divideY - 0.5f) * thicknessDetectorCrystal;
					float y2_detectorunit_self = (((float)NumY + 1.0f) / (float)divideY - 0.5f) * thicknessDetectorCrystal;

					float z1_detectorunit_self = ((float)NumZ / (float)divideZ - 0.5f) * heightDetectorCrystal;
					float z2_detectorunit_self = (((float)NumZ + 1.0f) / (float)divideZ - 0.5f) * heightDetectorCrystal;


					float dist_extend = 1000.0f;
					float length_crystalself_att1 = 0.0000f;
					float length_crystalself_att2 = 0.0000f;

					length_crystalself_att1 = length_box_ray_inside(xImage_self, yImage_self, zImage_self, xDetector_self, yDetector_self, zDetector_self, x1_detectorcrystal_self, y1_detectorcrystal_self, z1_detectorcrystal_self, x2_detectorcrystal_self, y2_detectorcrystal_self, z2_detectorcrystal_self);
					length_crystalself_att2 = length_box_ray_inside(xImage_self, yImage_self, zImage_self, xDetector_self, yDetector_self, zDetector_self, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);
					
					attenuation_dist = attenuation_dist + (length_crystalself_att1 - length_crystalself_att2) * coeff_detector_total;

					detection_eff_each = detection_eff_each * exp(-attenuation_dist);
					
					
					/////////////////////////////////  Absoption of scattered photons within the detector unit /////////////////////////////
					float x_tmp = xDetector_self + dist_extend * (xDetector_self - xImage_self) /sqrt(distancesq);
					float y_tmp = yDetector_self + dist_extend * (yDetector_self - yImage_self) /sqrt(distancesq);
					float z_tmp = zDetector_self + dist_extend * (zDetector_self - zImage_self) /sqrt(distancesq);
					float length_absorp = 0.000f;
					length_absorp = length_box_ray(xImage_self, yImage_self, zImage_self, x_tmp, y_tmp, z_tmp, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);
					absorption_factor = (1.0f - exp(-length_absorp * coeff_detector_total)) * coeff_detector_pe / coeff_detector_total;

					atomicAdd(&dst[dstIndex],detection_eff_each * solid_angle * absorption_factor);
				}
			}
		}
	}
}



int PESysMatGen(float* parameter_Collimator, float* parameter_Detector, float* parameter_Image, float* dst,int cuda_id)
{

	cout << "Get into PESysMatGen" << endl;
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
	cudaSetDevice(cuda_id);
	cout << "Set Device to Device " << cuda_id<< endl;

	float* deviceMatrix, * deviceparameter_Collimator, * deviceparameter_Detector, * deviceparameter_Image;
	cudaMalloc(&deviceMatrix, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMemset(deviceMatrix, 1, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMalloc(&deviceparameter_Collimator, sizeof(float) * 80000);
	cudaMemcpy(deviceparameter_Collimator, parameter_Collimator, sizeof(float) * 80000, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceparameter_Detector, sizeof(float) * 80000);
	cudaMemcpy(deviceparameter_Detector, parameter_Detector, sizeof(float) * 80000, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceparameter_Image, sizeof(float) * 100);
	cudaMemcpy(deviceparameter_Image, parameter_Image, sizeof(float) * 100, cudaMemcpyHostToDevice);

	/*每个网格（Grid）可以最多创建65535个线程块，每个线程块（Block）一般最多可以创建512个并行线程*/
	//取16*32线程块
	dim3 blockSize(16, 32);
	dim3 gridSize((numProjectionSingle + 15) / blockSize.x, (numImagebin + 31) / blockSize.y);
	cout << "########################" << endl;
	cout << "numProjectionSingle = " << numProjectionSingle << endl;
	cout << "numImagebin = " << numImagebin << endl;
	cout << "gridSize.x = " << gridSize.x << endl;
	cout << "gridSize.y = " << gridSize.y << endl;
	cout << "########################" << endl;

	photodetectorCudaMe <<<gridSize, blockSize >>> (
		deviceMatrix,
		deviceparameter_Collimator,
		deviceparameter_Detector,
		deviceparameter_Image,
		numProjectionSingle,
		numImagebin);
	//cudaThreadSynchronize();
	//float* test = new float[80000];
	cudaMemcpy(dst, deviceMatrix, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyDeviceToHost);
	//cudaMemcpy(test, deviceparameter_Collimator, sizeof(float) * 80000, cudaMemcpyDeviceToHost);

	cout << "########################" << endl;


	cudaFree(deviceparameter_Collimator);
	cudaFree(deviceparameter_Detector);
	cudaFree(deviceparameter_Image);
	cudaFree(deviceMatrix);


	return(numImagebin);
}
