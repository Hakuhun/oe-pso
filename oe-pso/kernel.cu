
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

	Location Location::operator=(const Location &mng) {
		x = mng.x;
		y = mng.y;
		return *this;
	}

	Location Location::operator-(const Location &mng) {
		float x = this->x - mng.x;
		float y = this->y - mng.y;
		return Location(x, y);
	}

	Location Location::operator+(const Location &mng) {
		float x = x + mng.x;
		float y = y + mng.y;
		return Location(x, y);
	}

	Location Location::operator*(const int number) {
		float x = x * number;
		float y = y * number;
		return Location(x, y);
	}
};

class Particle : public Managed
{
public:

	Particle() {

	}

	//vector of the current location
	Location * position;
	//vector of the particle's local optimum
	Location * localOptimum;
	//vector of the position where the particle is heading to
	Location * direction;
	//
	Location * velocity;
};

__device__ Particle dev_particles[N];
Particle host_particles[N];

__device__ Location * globalOptimum = new Location(1,1);

//Innertial coefficent (innerciális együttható)
__device__ float w = 0.5;

//Acceleration coefficent (gyorsítási együttható)
__device__ float c1 = 0.2;

//Acceleration coefficent (gyorsítási együttható)
__device__ float c2 = 0.2;

__device__ void CalculateVelocity(Particle * particle) {
	//Calculate the velocity
	//Sets the direction to the previous velocity
	Location * previous_velocity = particle->velocity;
	
	int r1 = rand() % (1 - 0 + 1) + 0;
	int r2 = rand() % (1 - 0 + 1) + 0;

	particle->velocity = particle->direction * 2;

	//particle->velocity = new Location(x,y);
}

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


