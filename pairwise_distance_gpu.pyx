import numpy as np
cimport numpy as np
cdef extern from "pairwise_distance.h":
	int pairwise_distance_gpu(float* x, int nrows, int ncols, float* dist)
	int pairwise_distance_gpu(double* x, int nrows, int ncols, double* dist)
def pairwise_dist_gpu(np.ndarray[np.double_t,ndim=2, mode="c"] X, 
		      np.ndarray[np.double_t,ndim=2, mode = "c"] dist):
	assert X.shape[0] == dist.shape[0] == dist.shape[1]
	cdef int nrows
	cdef int ncols
	nrows = X.shape[0]
	ncols = X.shape[1]
	return pairwise_distance_gpu(<double*> X.data, nrows, ncols, <double*> dist.data)

def pairwise_dist_gpu(np.ndarray[np.float32_t,ndim=2, mode="c"] X,
                      np.ndarray[np.float32_t,ndim=2, mode = "c"] dist):
	assert X.shape[0] == dist.shape[0] == dist.shape[1]
	cdef int nrows
	cdef int ncols
	nrows = X.shape[0]
	ncols = X.shape[1]
	return pairwise_distance_gpu(<float*> X.data, nrows, ncols, <float*> dist.data)
