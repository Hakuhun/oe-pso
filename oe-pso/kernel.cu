
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <iostream>

#define N 50

using namespace std;

class Managed {
public:
	void *operator new(size_t len) {
		void *ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void *ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

class Location : public Managed
{
public:
	float x;
	float y;

	Location(float x, float y) {
		this->x = x;
		this->y = y;
	}
};

class Particle : public Managed
{
public:
	Location * position;
	Location * localOptimum;
};

__device__ Particle dev_particles[N];
Particle host_particles[N];

void initParticles() {
	for (size_t i = 0; i < N; i++)
	{
		srand(time(NULL));
		host_particles[i] = Particle();
		host_particles[i].position = new Location(rand(), rand());
	}
}

void checkError() {
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}

int main()
{
	//initialize particles with random positions (on host)
	initParticles();

	//copy particles from host to device
	cudaMemcpyToSymbol(host_particles, dev_particles, N * sizeof(Particle));

	cout << "Atmasolva";

	cin.get();

    return 0;
}


