//Do NOT MODIFY THIS FILE

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"
#include "bmm.h"

// ===========================> Functions Prototype <===============================
void fill(float* data, int size);
double calc_mse(float* data1, float* data2, int size);
void cpuKernel_yx(const float* const a, const float* const b, float* c, const int m, const int n, const int y, const int x);
void cpuKernel_y(const float* const a, const float* const b, float* c, const int m, const int n, const int y);
void cpuKernel(const float* const a, const float* const b, float* c, const int m, const int n);
void gpuKernel(const float* const a, const float* const b, float* c, const int m, const int n, double* gpu_kernel_time);
// =================================================================================

int main(int argc, char** argv) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);
	
	// get parameter from command line to build Matrix dimension
	// check for 10<=m<=13, because m>=14 do not fit in the memory of our GPU, i.e., 1GB.
	int m = atoi(argv[1]);
    int n = (1 << m);
	
	// allocate memory in CPU for calculation
	float* a;
	float* b;
	float* c_serial;
	float* c;
	a        = (float*)malloc(n*n * sizeof(float));
	b        = (float*)malloc(n*n * sizeof(float));
	c_serial = (float*)malloc(n*n * sizeof(float));
	c        = (float*)malloc(n*n * sizeof(float));
	
	// fill a, b matrices with random values between -16.0f and 16.0f
	srand(0);
	fill(a, n*n);
	fill(b, n*n);

	// time measurement for CPU calculations
	clock_t t0 = clock(); 
	if (m<=10) {
		cpuKernel (a, b, c_serial, m, n);
	} else {
		cpuKernel_y (a, b, c_serial, m, n, 0);   // first row
		cpuKernel_y (a, b, c_serial, m, n, n-1); // last row
	}
	clock_t t1 = clock(); 
		
	// time measurement for GPU calculations
	clock_t t2 = clock(); 
	gpuKernel (a, b, c, m, n);
    clock_t t3 = clock(); 
		
	// check correctness of GPU calculations against CPU
	double mse = 0.0;
	if (m<=10) {
		mse += calc_mse( c_serial, c, n*n );
	} else {
		mse += calc_mse( c_serial          , c          , n ); // first row
		mse += calc_mse( c_serial + n*(n-1), c + n*(n-1), n ); // last row
	}

	printf("m=%d n=%d CPU=%g ms GPU=%g ms mse=%g\n",
	m, n, (t1-t0)/1000.0, (t3-t2)/1000.0, mse);
		
	// free allocated memory for later use
	free(a);
	free(b);
	free(c_serial);
	free(c);
   
	return 0;
}
//-----------------------------------------------------------------------------
void fill(float* data, int size) {
    for (int i=0; i<size; ++i)
        data[i] = (float) (rand() % 17 - 8);
}

double calc_mse (float* data1, float* data2, int size) {
	double mse = 0.0;
	int i; for (i=0; i<size; i++) {
		double e = data1[i]-data2[i];
		e = e * e;
		mse += e;
	}
	return mse;
}
//-----------------------------------------------------------------------------
void cpuKernel_yx(const float* const a, const float* const b, float* c, const int m, const int n, 
                  const int y, const int x) { // one element: y,x
	mem2d(c,m,y,x)=0.0f;
    for(int k=0; k<n; k++) {
		mem2d(c,m,y,x) += mem2d(a,m,y,k) * mem2d(b,m,k,x);
	}
}
void cpuKernel_y(const float* const a, const float* const b, float* c, const int m, const int n,
                 const int y) { // one row: y
    for(int x=0; x<n; x++) {
		cpuKernel_yx(a,b,c,m,n,y,x);
	}
}
void cpuKernel(const float* const a, const float* const b, float* c, const int m, const int n) { // entire matrix
    for(int y=0; y<n; y++)
    for(int x=0; x<n; x++) {
		cpuKernel_yx(a,b,c,m,n,y,x);
	}
}

