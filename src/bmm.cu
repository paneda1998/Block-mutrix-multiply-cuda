
//IN THE NAME OF GOD
//creatrd by alireza baneshi
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"
#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY is used to set number of threads in a CUDA block 
#define TILE_WIDTH 32
#define TILEY 32
#define TILEX 32



dim3 getDimGrid(const int m, const int n) {
	if(TILEX>TILEY){

dim3 dimGrid(n/(TILEX),n/(TILEX));

	return dimGrid;

}
else{
	dim3 dimGrid(n/(TILEY),n/(TILEY));

	return dimGrid;
}
}

dim3 getDimBlock(const int m, const int n) {
	if(TILEX>TILEY){
dim3 dimBlock(TILEX,TILEX);
	return dimBlock;

}
else{
	dim3 dimBlock(TILEY,TILEY);
	return dimBlock;
}
}



__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {
	 
	if(TILEX==TILEY)						{

    int row = by *TILEY + ty;	 int col = bx * TILEY + tx;
	float tmpVal = 0.0f;
	    __shared__ float bds[TILEX][TILEY];
	    __shared__ float ads[TILEY][TILEX];

    for(int i = 0;i < n / (TILEX);i++){
        ads[ty][tx] = ad[(row * n) + (i * TILEX) + tx];
        bds[ty][tx] = bd[col + (((i * TILEX) + ty) * n)];
        __syncthreads();
for(int k = 0;k < TILEX;k++){
            tmpVal += ads[ty][k] * bds[k][tx];
        }
	__syncthreads();
    }
	// cd[row][col] = ?
	mem2d(cd,m,row,col) = tmpVal;
									}
	else if(TILEY>TILEX)						        {
int row = by *TILEY + ty;	 int col = bx * TILEY + tx;
	float tmpVal = 0.0f;

    __shared__ float bds[TILEX][TILEY];
    __shared__ float ads[TILEY][TILEX];

    for(int i = 0;i < n / (TILEX);i++){
	if(tx<TILEX)
        ads[ty][tx] = ad[(row * n) + (i * TILEX) + tx];
	if(ty<TILEX)
        bds[ty][tx] = bd[col + (((i * TILEX) + ty) * n)];

for(int k = 0;k < TILEX;k++){
        __syncthreads();
            tmpVal += ads[ty][k] * bds[k][tx];
        }
	__syncthreads();
    }
	// cd[row][col] = ?
	mem2d(cd,m,row,col) = tmpVal;



									}
	else if(TILEX>TILEY)						        {
const int tiley = TILEX;
const int tilex = TILEY;

int row = by *tiley + ty;	 int col = bx * tiley + tx;
	float tmpVal = 0.0f;

    __shared__ float bds[tilex][tiley];
    __shared__ float ads[tiley][tilex];

    for(int i = 0;i < n / (tilex);i++){
	if(tx<tilex)
        ads[ty][tx] = ad[(row * n) + (i * tilex) + tx];
	if(ty<tilex)
        bds[ty][tx] = bd[col + (((i * tilex) + ty) * n)];

for(int k = 0;k < tilex;k++){
        __syncthreads();
            tmpVal += ads[ty][k] * bds[k][tx];
        }
	__syncthreads();
    }
	// cd[row][col] = ?
	mem2d(cd,m,row,col) = tmpVal;



									}



}






//-----------------------------------------------------------------------------
void gpuKernel(const float* const  a, const float* const b, float* c, const int m, const int n) {
	
	
	float* ad;
        float* bd;
        float* cd;
        float* ad1;
        float* bd1;
	float r =n/2;
        float r1 =m-1;
        float* cd2;
        bd1 = (float*)malloc(n*(n/4) * sizeof(float));
	ad1 = (float*)malloc(n*(n/4) * sizeof(float));
	cd2 = (float*)malloc(n*(n/4) * sizeof(float));
	
if(m<14){
   dim3 dimGrid = getDimGrid(m,n); 
	dim3 dimBlock = getDimBlock(m,n); 




    HANDLE_ERROR(cudaMalloc((void**)&ad, n*(n) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&bd, n*(n) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&cd, n*(n) * sizeof(float)));

HANDLE_ERROR(cudaMemcpy(ad, a, n*(n) * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(bd, b, n*(n) * sizeof(float), cudaMemcpyHostToDevice));
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, m, n);
    HANDLE_ERROR(cudaMemcpy(c, cd, n*(n) * sizeof(float), cudaMemcpyDeviceToHost));
        


                HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(bd));
                HANDLE_ERROR(cudaFree(cd));



}
else
{


	for (int i = 0 ; i< n/2 ;i++){

		for (int k = 0; k<(n/2) ;k++){

			bd1[k+(i*n/2)] = b[k+(i*n)];
			ad1[k+(i*n/2)] = a[k+(i*n)];
			}

	      }
    HANDLE_ERROR(cudaMalloc((void**)&ad, n*(n/4) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&bd, n*(n/4) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&cd, n*(n/4) * sizeof(float)));


     
    HANDLE_ERROR(cudaMemcpy(ad, ad1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(bd, bd1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
dim3 dimGrid = getDimGrid(r1,r); 
	dim3 dimBlock = getDimBlock(r1,r); 
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, r1, r);
//std::cout<<1;
    HANDLE_ERROR(cudaMemcpy(cd2, cd, n*(n/4) * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0 ; i< n/2 ;i++)
		for (int k = 0; k<(n/2) ;k++)
			c[k+n*i] = cd2[k+(n*i/2)];
/////////////////////////////////////////////////////////////////////////////////////////////////////////


for (int i = 0 ; i< n/2 ;i++){

		for (int k = 0; k<(n/2) ;k++){
			bd1[k+(i*n/2)] = b[k+(i*n)+(n*n/2)];
			ad1[k+(i*n/2)] = a[k+(i*n)+(n/2)];
			}

	      }


    HANDLE_ERROR(cudaMemcpy(ad, ad1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(bd, bd1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, r1, r);
    HANDLE_ERROR(cudaMemcpy(cd2, cd, n*(n/4) * sizeof(float), cudaMemcpyDeviceToHost));
         for (int i = 0 ; i< n/2 ;i++)
		for (int k = 0; k<(n/2) ;k++)
			c[k+n*i] += cd2[k+(n*i/2)];
/////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////


for (int i = 0 ; i< n/2 ;i++){

		for (int k = 0; k<(n/2) ;k++){

			bd1[k+(i*n/2)] = b[k+(i*n)+(n/2)+(n*n/2)];
			}

	      }


    HANDLE_ERROR(cudaMemcpy(bd, bd1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, r1, r);
    HANDLE_ERROR(cudaMemcpy(cd2, cd, n*(n/4) * sizeof(float), cudaMemcpyDeviceToHost));
         for (int i = 0 ; i< n/2 ;i++)
		for (int k = 0; k<(n/2) ;k++)
			c[k+n*i+n/2] = cd2[k+(n*i/2)];
/////////////////////////////////////////////////////////////////////////////////////////////////////////



for (int i = 0 ; i< n/2 ;i++){

		for (int k = 0; k<(n/2) ;k++){


			bd1[k+(i*n/2)] = b[k+(i*n)+(n/2)];
			ad1[k+(i*n/2)] = a[k+(i*n)];
			}

	      }



    HANDLE_ERROR(cudaMemcpy(ad, ad1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(bd, bd1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, r1, r);
    HANDLE_ERROR(cudaMemcpy(cd2, cd, n*(n/4) * sizeof(float), cudaMemcpyDeviceToHost));
         for (int i = 0 ; i< n/2 ;i++)
		for (int k = 0; k<(n/2) ;k++)
			c[k+n*i+n/2] += cd2[k+(n*i/2)];
/////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////


for (int i = 0 ; i< n/2 ;i++){

		for (int k = 0; k<(n/2) ;k++){

			
			ad1[k+(i*n/2)] = a[k+(i*n)+(n*n/2)];

		}

	      }


    HANDLE_ERROR(cudaMemcpy(ad, ad1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, r1, r);
    HANDLE_ERROR(cudaMemcpy(cd2, cd, n*(n/4) * sizeof(float), cudaMemcpyDeviceToHost));
         for (int i = 0 ; i< n/2 ;i++)
		for (int k = 0; k<(n/2) ;k++)
			c[k+n*i+n/2+(n*n/2)] = cd2[k+(n*i/2)];
/////////////////////////////////////////////////////////////////////////////////////////////////////////


for (int i = 0 ; i< n/2 ;i++){

		for (int k = 0; k<(n/2) ;k++){

			bd1[k+(i*n/2)] = b[k+(i*n)+(n/2)+(n*n/2)];

			ad1[k+(i*n/2)] = a[k+(i*n)+(n/2)+(n*n/2)];

		}

	      }



    HANDLE_ERROR(cudaMemcpy(ad, ad1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(bd, bd1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, r1, r);
    HANDLE_ERROR(cudaMemcpy(cd2, cd, n*(n/4) * sizeof(float), cudaMemcpyDeviceToHost));
         for (int i = 0 ; i< n/2 ;i++)
		for (int k = 0; k<(n/2) ;k++)
			c[k+n*i+n/2+(n*n/2)] += cd2[k+(n*i/2)];
/////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////



for (int i = 0 ; i< n/2 ;i++){

		for (int k = 0; k<(n/2) ;k++){

			bd1[k+(i*n/2)] = b[k+(i*n)+(n*n/2)];
			}

	      }



    HANDLE_ERROR(cudaMemcpy(bd, bd1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, r1, r);
    HANDLE_ERROR(cudaMemcpy(cd2, cd, n*(n/4) * sizeof(float), cudaMemcpyDeviceToHost));
         for (int i = 0 ; i< n/2 ;i++)
		for (int k = 0; k<(n/2) ;k++)
			c[k+n*i+(n*n/2)] = cd2[k+(n*i/2)];
/////////////////////////////////////////////////////////////////////////////////////////////////////////



for (int i = 0 ; i< n/2 ;i++){

		for (int k = 0; k<(n/2) ;k++){

			bd1[k+(i*n/2)] = b[k+(i*n)];			
			ad1[k+(i*n/2)] = a[k+(i*n)+(n*n/2)];


		}

	      }


    HANDLE_ERROR(cudaMemcpy(ad, ad1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(bd, bd1, n*(n/4) * sizeof(float), cudaMemcpyHostToDevice));
    kernelFunc<<< dimGrid,dimBlock >>>(ad, bd, cd, r1, r);
    HANDLE_ERROR(cudaMemcpy(cd2, cd, n*(n/4) * sizeof(float), cudaMemcpyDeviceToHost));
         for (int i = 0 ; i< n/2 ;i++)
		for (int k = 0; k<(n/2) ;k++)
			c[k+n*i+(n*n/2)] += cd2[k+(n*i/2)];



                HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(bd));
                HANDLE_ERROR(cudaFree(cd));

	//GpuTimer timer;

    //timer.Start();*/
}
}

