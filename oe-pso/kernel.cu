
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "vector_functions.h"
#include <curand_kernel.h>
#include <ctime>
#include <helper_cuda.h>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <iostream>

#define N 1000
#define MIN2 0
#define MAX2 50
#define RANDOM(a, b) rand()%(MAX2-MIN2+1)+(MIN2)
#define fitness(x) DistanceCalculate()

using namespace std;

//class Managed {
//public:
//
//
//
//	void *operator new(size_t len) {
//		void *ptr;
//		cudaMallocManaged(&ptr, len);
//		cudaDeviceSynchronize();
//		return ptr;
//	}
//
//	void operator delete(void *ptr) {
//		cudaDeviceSynchronize();
//		cudaFree(ptr);
//	}
//};

class Particle /*: public Managed*/
{
public:
	__host__ __device__ Particle() {
		velocity = make_float2(0, 0);
	}

	//vector of the current location
	float2 position;
	//vector of the particle's local optimum
	float2 localOptimum;
	//vector of the position where the particle is heading to
	float2 direction;

	float2 velocity;
};

__device__ Particle dev_particles[N];
Particle host_particles[N];

__device__ float2 dev_globalOptimum;

int asd5[N];
__device__ int dasd[N];

//Innertial coefficent (innerciális együttható)
__constant__ float w = 0.5;

//Acceleration coefficent (gyorsítási együttható)
__constant__ float c1 = 0.2;

//Acceleration coefficent (gyorsítási együttható)
__constant__ float c2 = 0.2;

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ float2 operator*(float2 a, int b)
{
	return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float2 operator*(int b, float2 a)
{
	return make_float2(a.x * b, a.y * b);
}

//fitness 
__device__ double DistanceCalculate(float2 a, float2 b)
{
	float2 diff = a - b;
	return sqrt(pow(diff.x, 2) + pow(diff.y, 2));
}

__global__ void Evaluation() {

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Particle * particle = &dev_particles[index];

	if (DistanceCalculate(particle->direction, particle->localOptimum) <
		DistanceCalculate(particle->direction - particle->velocity, particle->localOptimum))
	{
		particle->localOptimum = particle->direction;

		if (DistanceCalculate(particle->direction, particle->localOptimum) <
			DistanceCalculate(particle->direction - particle->velocity, dev_globalOptimum))
		{
			dev_globalOptimum = particle->direction;
		}
	}
}

__device__ float cudaRand()
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);

	return curand_uniform_double(&state);
}

__global__ void CalculateVelocity() {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Particle * particle = &dev_particles[index];

	//Calculate the velocity
	particle->velocity = w * particle->velocity
		+ cudaRand() * c1 * (particle->localOptimum - particle->direction)
		+ cudaRand() * c2 * (dev_globalOptimum - particle->direction);
	printf("Value is: %d\n", particle->velocity.x);
}

__global__ void CalculateNewDirection() {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Particle * particle = &dev_particles[index];

	particle->direction = particle->direction + particle->velocity;
}

__global__ void checkParticles(){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Particle * particle = &dev_particles[index];
	printf("Value is: %d\n", particle->direction.x);
}

void initParticles() {
	for (size_t i = 0; i < N; i++)
	{
		srand(time(NULL));
		host_particles[i] = Particle();
		host_particles[i].position = make_float2(RANDOM(MIN2, MAX2), RANDOM(MIN2, MAX2));
		host_particles[i].localOptimum = make_float2(RANDOM(MIN2, MAX2), RANDOM(MIN2, MAX2));
		host_particles[i].direction = make_float2(RANDOM(MIN2, MAX2), RANDOM(MIN2, MAX2));
		asd5[i] = RANDOM(0, 100);
	}
}

void checkError() {
	cudaError_t cudaStatus = cudaGetLastError();	
	if (cudaStatus != cudaSuccess) {
		cout << "Hiba: " << cudaGetErrorName(cudaStatus) << '\n';
		cout << cudaGetErrorString(cudaStatus);
	}
}



int main()
{
	//initialize particles with random positions (on host)
	initParticles();

	//copy particles from host to device
	cudaMemcpyToSymbol(dev_particles, host_particles, N * sizeof(Particle));
	checkError();

	//initalize global optimum variable
	float2 host_gOptimum = make_float2(1, 1);
	cudaMemcpyToSymbol(&dev_globalOptimum, &host_gOptimum, N * sizeof(float2));

	checkError();
	cudaMemcpyToSymbol(dasd, asd5, N * sizeof(int));

	checkParticles << <1, N >> > ();
	checkError();

	Evaluation << <1, N >> > ();
	checkError();

	int i = 0;
	
	while (i < 1000)
	{
		CalculateVelocity << <1, N >> > ();
		checkError();
		CalculateNewDirection << <1, N >> > ();
		checkError();
		Evaluation << <1, N >> > ();
		checkError();
		i++;
	}
	cout << "Vege";
	cin.get();

    return 0;
}