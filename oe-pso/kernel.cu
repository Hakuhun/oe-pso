
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <iostream>

#define N 50
#define MIN 0
#define MAX 50
#define RANDOM(MIN, MAX) rand()%(MAX-MIN+1)+MIN

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

	Location() {}

	Location(float x, float y) {
		this->x = x;
		this->y = y;
	}

	Location operator=(const Location &mng) {
		x = mng.x;
		y = mng.y;
		return *this;
	}

	//Két vektor különbsége
	Location operator-(const Location &mng) {
		return Location(this->x - mng.x, this->y - mng.y);
	}

	//Két vektor összege
	Location operator+(const Location &mng) {
		Location result = Location(this->x + mng.x, this->y + mng.y);
		return result;
	}

	//számmal való szorzás
	Location operator*(const int &number) {
		return Location(this->x * number, this->y * number);
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

__device__ Location *dev_globalOptimum;

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
	
}

void initParticles() {
	int min = MIN, max = MAX;
	for (size_t i = 0; i < N; i++)
	{
		srand(time(NULL));
		host_particles[i] = Particle();
		host_particles[i].position = new Location(RANDOM(MIN, MAX), RANDOM(MIN, MAX));
		host_particles[i].localOptimum = new Location(RANDOM(MIN, MAX), RANDOM(MIN, MAX));
		host_particles[i].direction = new Location(RANDOM(MIN, MAX), RANDOM(MIN, MAX));
	}
}

void checkError() {
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout << stderr, cudaGetErrorName(cudaStatus);
	}
}

int main()
{
	//initialize particles with random positions (on host)
	initParticles();

	//copy particles from host to device
	cudaMemcpyToSymbol(host_particles, dev_particles, N * sizeof(Particle));
	checkError();
	//initalize global optimum variable
	Location * host_gOptimum = new Location(1, 1);
	cudaMemcpyToSymbol(host_gOptimum, dev_globalOptimum, N * sizeof(Location));
	checkError();

	cout << "Atmasolva";



	cin.get();

    return 0;
}


