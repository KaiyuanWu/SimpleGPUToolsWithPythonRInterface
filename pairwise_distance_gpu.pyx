import numpy as np
cimport numpy as np
cdef extern from "pairwise_distance.h":
	int pairwise_distance_gpu1(float* x, int nrows, int ncols, float* dist)
	int pairwise_distance_gpu2(float* x, int xnrows, int xncols, float* y, int ynrows, int yncols, float* dist)
def pairwise_dist_gpu1(np.ndarray[np.float32_t,ndim=2, mode="c"] X,
                      np.ndarray[np.float32_t,ndim=2, mode = "c"] dist):
	assert X.shape[0] == dist.shape[0] == dist.shape[1]
	cdef int nrows
	cdef int ncols
	nrows = X.shape[0]
	ncols = X.shape[1]
	return pairwise_distance_gpu1(<float*> X.data, nrows, ncols, <float*> dist.data)

def pairwise_dist_gpu2(np.ndarray[np.float32_t,ndim=2, mode="c"] X,np.ndarray[np.float32_t,ndim=2, mode="c"] Y,
                      np.ndarray[np.float32_t,ndim=2, mode = "c"] dist):
	assert X.shape[0] == dist.shape[0] 
	assert X.shape[1] == Y.shape[1] 
	assert Y.shape[0] == dist.shape[1] 
	cdef int xnrows
	cdef int xncols
	cdef int ynrows
	cdef int yncols
	xnrows = X.shape[0]
	xncols = X.shape[1]
	ynrows = Y.shape[0]
	yncols = Y.shape[1]
	return pairwise_distance_gpu2( <float*> Y.data, ynrows, yncols, <float*> X.data, xnrows, xncols, <float*> dist.data)
