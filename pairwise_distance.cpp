#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <time.h>
#include "pairwise_distance.h"
void gpu_gemm(cublasHandle_t handle, const int nrows, const int ncols, const float* x,
    float* out) {
	float alpha=1.f;
	float beta =0.f;
  	cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T, nrows, ncols, &alpha, x, ncols,  &beta, out, nrows);
}

void gpu_gemm(cublasHandle_t handle, const int nrows, const int ncols, const double* x,
    double* out) {
        double alpha=1.;
        double beta =0.;
        cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T, nrows, ncols, &alpha, x, ncols,  &beta, out, nrows);
}


void gpu_dot(cublasHandle_t handle, const int nrows, const int ncols, const double* x,
    double* out) {
	double alpha = 1.;
	double beta = 0.;
	cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T, nrows, ncols, &alpha, x, ncols,  &beta, out, nrows);
}


void gpu_syr2k(cublasHandle_t handle, const int nrows, const float* diag, const float* ones, float* dist){
	float alpha = 1.f;
        float beta = -2.f;
	cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, nrows, 1, &alpha, diag, 1, ones, 1, &beta, dist, nrows);
}

void gpu_syr2k(cublasHandle_t handle, const int nrows, const double* diag, const double* ones, double* dist){
        double alpha = 1.;
        double beta = -2;
        cublasDsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, nrows, 1, &alpha, diag, 1, ones, 1, &beta, dist, nrows);
}

int pairwise_distance_gpu(double* x, int nrows, int ncols, double* dist){
	cudaError_t cudaStat;    
    	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle); 
	if (stat != CUBLAS_STATUS_SUCCESS) { 
		printf ("CUBLAS initialization failed\n"); 
		return EXIT_FAILURE; 
	}
	double* dev_x;
	double* dev_dist;
	double* dev_ones;
	double* diag;
	double* dev_diag;
	diag = new double[nrows];
	
	cudaStat = cudaMalloc ((void**)&dev_x, nrows*ncols*sizeof(double));
	if (cudaStat != cudaSuccess) { 
		printf ("device memory allocation failed x\n"); 
		return EXIT_FAILURE; 
	}

	cudaStat = cudaMalloc ((void**)&dev_ones, nrows*sizeof(double));
        if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed ones\n");
                return EXIT_FAILURE;
        }
	cudaStat = cudaMalloc ((void**)&dev_diag, nrows*sizeof(double));
        if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed diag\n");
                return EXIT_FAILURE;
        }

        cudaStat = cudaMalloc ((void**)&dev_dist, nrows*nrows*sizeof(double));   
        if (cudaStat != cudaSuccess) { 
                printf ("device memory allocation failed dist\n"); 
                return EXIT_FAILURE; 
        }

	cudaStat = cudaMemcpy (dev_x, x, ncols*nrows*sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess) { 
                printf ("device memory copy failed x\n"); 
                return EXIT_FAILURE; 
        }
	gpu_gemm(handle, nrows, ncols, dev_x, dev_dist);
	
	cudaStat = cudaMemcpy (dist, dev_dist, nrows*nrows*sizeof(double),cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
                printf ("device memory copy failed dist \n");
                return EXIT_FAILURE;
        }
	for(int i = 0; i < nrows; i++)
		diag[i] = dist[i*nrows+i];
	cudaStat = cudaMemcpy (dev_diag, diag, nrows*sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) { 
                printf ("device memory copy failed diag\n"); 
                return EXIT_FAILURE; 
        }  
	for(int i = 0; i < nrows; i++)
                diag[i] = 1;
	cudaStat = cudaMemcpy (dev_ones, diag, nrows*sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) {
                printf ("device memory copy failed diag\n");
                return EXIT_FAILURE;
        }
	gpu_syr2k(handle, nrows, dev_diag, dev_ones, dev_dist);	
	
	cudaStat = cudaMemcpy (dist, dev_dist, nrows*nrows*sizeof(double),cudaMemcpyDeviceToHost);
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

int pairwise_distance_gpu(float* x, int nrows, int ncols, float* dist){
	cudaError_t cudaStat;    
    	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle); 
	if (stat != CUBLAS_STATUS_SUCCESS) { 
		printf ("CUBLAS initialization failed\n"); 
		return EXIT_FAILURE; 
	}
	float* dev_x;
	float* dev_dist;
	float* dev_ones;
	float* diag;
	float* dev_diag;
	diag = new float[nrows];
	
	cudaStat = cudaMalloc ((void**)&dev_x, nrows*ncols*sizeof(float));
	if (cudaStat != cudaSuccess) { 
		printf ("device memory allocation failed x\n"); 
		return EXIT_FAILURE; 
	}

	cudaStat = cudaMalloc ((void**)&dev_ones, nrows*sizeof(float));
        if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed ones\n");
                return EXIT_FAILURE;
        }
	cudaStat = cudaMalloc ((void**)&dev_diag, nrows*sizeof(float));
        if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed diag\n");
                return EXIT_FAILURE;
        }

        cudaStat = cudaMalloc ((void**)&dev_dist, nrows*nrows*sizeof(float));   
        if (cudaStat != cudaSuccess) { 
                printf ("device memory allocation failed dist\n"); 
                return EXIT_FAILURE; 
        }

	cudaStat = cudaMemcpy (dev_x, x, ncols*nrows*sizeof(float), cudaMemcpyHostToDevice);
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
