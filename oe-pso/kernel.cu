
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <iostream>

#include "location.cu"	
#include "particle.cu"

#define N 50

using namespace std;

__device__ Particle dev_particles[N];
Particle particles[N];

__device__ Position dev_globalOptimum;

void initParticles() {
	srand(time(NULL));
	for (size_t i = 0; i < N; i++)
	{
		float x = rand();
		float y = rand();
		Position location = Position(&x, &y);
		particles[i].position = &location;

	}
} 
int main()
{
	//initialize particles with random positions (on host)
	initParticles();

	//copy particles from host to device
	cudaMemcpyToSymbol(particles, dev_particles, N * sizeof(Particle));

    return 0;
}

void checkError() {
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}
