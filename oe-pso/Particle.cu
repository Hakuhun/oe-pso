
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "location.cu"	

class Particle
{

public:
	Position *position;
	Position *localOptimum;

	// Constructor.
	__host__ __device__ Particle() : position(NULL), localOptimum(NULL) {}

	// Constructor.
	__host__ __device__ Particle(float *x, float *y) {
		Position loc = Position(x, y);
		position = &loc;
		localOptimum = NULL;
	}

};