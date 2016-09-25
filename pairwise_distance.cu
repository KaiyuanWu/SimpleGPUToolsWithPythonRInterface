#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <time.h>
#include "pairwise_distance.h"
const int CUDA_NUM_THREADS = 512;
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
 
__global__ void pow2_kernel(const int n, const float* x, float* x2) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
	x2[i] = x[i]*x[i];
  }
}

void gpu_gemm(cublasHandle_t handle, const int nrows, const int ncols, const float* x,
    float* out) {
	float alpha=1.f;
	float beta =0.f;
  	cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T, nrows, ncols, &alpha, x, ncols,  &beta, out, nrows);
}

void gpu_syr2k(cublasHandle_t handle, const int nrows, const float* diag, const float* ones, float* dist){
	float alpha = 1.f;
        float beta = -2.f;
	cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, nrows, 1, &alpha, diag, 1, ones, 1, &beta, dist, nrows);
}

int pairwise_distance_gpu2(float* x, int xnrows, int xncols, float* y, int ynrows, int yncols, float* dist){
        cublasStatus_t stat;
        cublasHandle_t handle;
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("CUBLAS initialization failed\n");
                return EXIT_FAILURE;
        }
	float *dev_x, *dev_y, *dev_x2, *dev_y2, *dev_dist, *dev_num_samples,*dev_xnorm, *dev_num_features, *dev_ynorm;
	int max_num_samples = max(xnrows, ynrows);
	int max_num_features = max(xncols, yncols);
	cudaError_t cudaStat1 = cudaMalloc ((void**)&dev_x, xnrows*xncols*sizeof(float));
	cudaError_t cudaStat2 = cudaMalloc ((void**)&dev_y, ynrows*yncols*sizeof(float));
	cudaError_t cudaStat3 = cudaMalloc ((void**)&dev_dist, xnrows*ynrows*sizeof(float));
	cudaError_t cudaStat4 = cudaMalloc ((void**)&dev_num_samples, max_num_samples*sizeof(float));
	cudaError_t cudaStat5 = cudaMalloc ((void**)&dev_num_features, max_num_features*sizeof(float));
	cudaError_t cudaStat6 = cudaMalloc ((void**)&dev_xnorm, xnrows*sizeof(float));
	cudaError_t cudaStat7 = cudaMalloc ((void**)&dev_ynorm, ynrows*sizeof(float));
	cudaError_t cudaStat8 = cudaMalloc ((void**)&dev_x2, xnrows*xncols*sizeof(float));
	cudaError_t cudaStat9 = cudaMalloc ((void**)&dev_y2, ynrows*yncols*sizeof(float));

	if(cudaStat1 != cudaSuccess||
  	   cudaStat2 != cudaSuccess ||	
	   cudaStat3 != cudaSuccess ||	
	   cudaStat4 != cudaSuccess ||	
 	   cudaStat5 != cudaSuccess ||	
	   cudaStat6 != cudaSuccess ||	
	   cudaStat7 != cudaSuccess ||
	   cudaStat8 != cudaSuccess ||
	   cudaStat9 != cudaSuccess){
		printf ("device memory allocation failed x\n"); 
                return EXIT_FAILURE; 
	}
	
	int maxn = max(max_num_samples, max_num_features);
	float* temp = new float[maxn];
	for(int i = 0; i < maxn; i++){
		temp[i] = 1;
	}
	cudaStat1 = cudaMemcpy(dev_x, x, xnrows*xncols*sizeof(float), cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(dev_y, y, ynrows*yncols*sizeof(float), cudaMemcpyHostToDevice);
	cudaStat4 = cudaMemcpy(dev_num_samples, temp, max_num_samples*sizeof(float), cudaMemcpyHostToDevice);
	cudaStat5 = cudaMemcpy(dev_num_features, temp, max_num_features*sizeof(float), cudaMemcpyHostToDevice);

	if(cudaStat1 != cudaSuccess||                                 
           cudaStat2 != cudaSuccess ||
           cudaStat4 != cudaSuccess ||
           cudaStat5 != cudaSuccess ){
                printf ("device memory allocation failed x\n");
                return EXIT_FAILURE;
        }	
	
	//calculate x.^2
	pow2_kernel<<<GET_BLOCKS(xnrows*xncols), CUDA_NUM_THREADS>>>(xnrows*xncols, dev_x, dev_x2);		
	//calculate y.^2
	pow2_kernel<<<GET_BLOCKS(ynrows*yncols), CUDA_NUM_THREADS>>>(ynrows*yncols, dev_y, dev_y2);		
	//calcuate xnorm
	float alpha = 1.f;
	float beta = 0.f;
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, xnrows,  xncols, &alpha, dev_num_features, 1, dev_x2, xncols,  &beta, dev_xnorm, 1);
	if( stat != CUBLAS_STATUS_SUCCESS){
		printf("fail to xnorm!\n");
		return EXIT_FAILURE;
	}
	//calculate ynorm
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, ynrows,  yncols, &alpha, dev_num_features, 1, dev_y2, yncols,  &beta, dev_ynorm, 1);
        if( stat != CUBLAS_STATUS_SUCCESS){
                printf("fail to ynorm!\n");
                return EXIT_FAILURE;
        }
	//calculate dist = xnorm*I 	
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, xnrows, ynrows, 1, &alpha, dev_xnorm, xnrows, dev_num_samples, 1,  &beta, dev_dist, xnrows);
        if( stat != CUBLAS_STATUS_SUCCESS){
                printf("fail to dist = xnorm*I!\n");
                return EXIT_FAILURE;
        }
	//calculate dist += I*ynorm
	beta = 1.f;
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, xnrows, ynrows, 1, &alpha, dev_num_samples, xnrows, dev_ynorm, 1,  &beta, dev_dist, xnrows);
        if( stat != CUBLAS_STATUS_SUCCESS){
                printf("fail to dist += I*ynorm!\n");
                return EXIT_FAILURE;
        } 
	alpha = -2.f;
	beta = 1.f;
	stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, xnrows, ynrows, xncols, &alpha, dev_x, xncols, dev_y, yncols,  &beta, dev_dist, xnrows);
        if( stat != CUBLAS_STATUS_SUCCESS){
                printf("fail to dist += x*yT!\n");
                return EXIT_FAILURE;
        }
	
	cudaStat1 = cudaMemcpy (dist, dev_dist, xnrows*ynrows*sizeof(float),cudaMemcpyDeviceToHost);
        if (cudaStat1 != cudaSuccess) {
                printf ("device memory copy failed dist \n");
                return EXIT_FAILURE;
        }
	// *dev_x, *dev_y, *dev_x2, *dev_y2, *dev_dist, *dev_num_samples,*dev_xnorm, *dev_num_features, *dev_ynorm
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_x2);
	cudaFree(dev_y2);
	cudaFree(dev_dist);
	cudaFree(dev_num_samples);
	cudaFree(dev_num_features);
	cudaFree(dev_xnorm);
	cudaFree(dev_ynorm);
	delete[] temp;
        cublasDestroy(handle);
        return true;
		
}
int pairwise_distance_gpu1(float* x, int nrows, int ncols, float* dist){
    	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle); 
	if (stat != CUBLAS_STATUS_SUCCESS) { 
		printf ("CUBLAS initialization failed\n"); 
		return EXIT_FAILURE; 
	}
	float *dev_x, *dev_dist, *dev_ones, *diag, *dev_diag;
	diag = new float[nrows];
	
	cudaError_t cudaStat1 = cudaMalloc ((void**)&dev_x, nrows*ncols*sizeof(float));
	cudaError_t cudaStat2 = cudaMalloc ((void**)&dev_ones, nrows*sizeof(float));
	cudaError_t cudaStat3 = cudaMalloc ((void**)&dev_diag, nrows*sizeof(float));;
	cudaError_t cudaStat4 = cudaMalloc ((void**)&dev_dist, nrows*nrows*sizeof(float));
	if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess || cudaStat4 != cudaSuccess) { 
		printf ("device memory allocation failed x\n"); 
		return EXIT_FAILURE; 
	}


	cudaError_t cudaStat = cudaMemcpy (dev_x, x, ncols*nrows*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess) { 
                printf ("device memory copy failed x\n"); 
                return EXIT_FAILURE; 
        }
	gpu_gemm(handle, nrows, ncols, dev_x, dev_dist);
	
	cudaStat = cudaMemcpy (dist, dev_dist, nrows*nrows*sizeof(float),cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
                printf ("device memory copy failed dist \n");
                return EXIT_FAILURE;
        }
	for(int i = 0; i < nrows; i++)
		diag[i] = dist[i*nrows+i];
	cudaStat = cudaMemcpy (dev_diag, diag, nrows*sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) { 
                printf ("device memory copy failed diag\n"); 
                return EXIT_FAILURE; 
        }  
	for(int i = 0; i < nrows; i++)
                diag[i] = 1;
	cudaStat = cudaMemcpy (dev_ones, diag, nrows*sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) {
                printf ("device memory copy failed diag\n");
                return EXIT_FAILURE;
        }
	gpu_syr2k(handle, nrows, dev_diag, dev_ones, dev_dist);	
	
	cudaStat = cudaMemcpy (dist, dev_dist, nrows*nrows*sizeof(float),cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) {
                printf ("device memory copy failed dist \n");
                return EXIT_FAILURE;
        }
	for(int i = 0; i < nrows; i++){
		for(int j = i+1; j < nrows; j++)
			dist[i*nrows+j] = dist[j*nrows + i];
	}

	cudaFree (dev_x);
	cudaFree (dev_dist);
	cudaFree (dev_diag);
	cudaFree (dev_ones);
	delete[] diag;
	cublasDestroy(handle);
	return true;
}
