
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Position
{
	float *m_x;
	float *m_y;

public:
	// Constructor.
	__host__ __device__ Position() : m_x(NULL), m_y(NULL) {}

	// Constructor.
	__host__ __device__ Position(float *x, float *y) : m_x(x), m_y(y) {}
		
	// Set the pointers.
	__host__ __device__ __forceinline__ void set(float *x, float *y)
	{
		m_x = x;
		m_y = y;
	}
};