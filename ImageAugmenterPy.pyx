import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "img_aug.h":
	int augment_img(unsigned char* img_data, int c, int h, int w, unsigned char* aug_img_data, int c1, int h1, int w1, 
	bool rand_crop,
        int crop_x_start,
        int crop_y_start,
        float max_rotate_angle,
        float max_aspect_ratio,
        float max_shear_ratio,
        int max_crop_size,
        int min_crop_size,
        float max_random_scale,
        float min_random_scale,
        float max_img_size,
        float min_img_size,
        float random_h,
        float random_s,
        float random_l,
        float rotate,
        int fill_value,
        int inter_method,
        int pad,
        bool rand_mirror,
	int nlandmarks,
	float* landmarks_x,
	float* landmarks_y)

def augment_img_process(np.ndarray[np.uint8_t,ndim=3, mode="c"] img, nlandmarks = 0, 
	np.ndarray[np.float32_t, ndim=1, mode="c"] landmarks_x=None, np.ndarray[np.float32_t, ndim=1, mode="c"] landmarks_y=None, 
	args={}):
	cdef bool rand_crop = False
	cdef int crop_x_start = -1
	cdef int crop_y_start = -1
	cdef float max_rotate_angle = 0
	cdef float max_aspect_ratio = 0
	cdef float max_shear_ratio = 0
	cdef int max_crop_size = -1
	cdef int min_crop_size = -1
	cdef float max_random_scale = 1
	cdef float min_random_scale = 1
	cdef float max_img_size = 1.0e10
	cdef float min_img_size = 0.0
	cdef float random_h = 0
	cdef float random_s = 0
	cdef float random_l = 0
	cdef float rotate = -1
	cdef int fill_value = 255
	cdef int inter_method = 1
	cdef int pad = 0
	cdef bool rand_mirror = False
	cdef int c
	cdef int h
	cdef int w
	cdef int c1
	cdef int h1
	cdef int w1
	cdef int nlandmarks_

	print(type(nlandmarks))	
	nlandmarks_  = nlandmarks

	if 'data_shape' not in args:
		return -1
	else:
		c1 = args['data_shape'][0]
		h1 = args['data_shape'][1]
		w1 = args['data_shape'][2]
	h = img.shape[0]
	w = img.shape[1]
	c = img.shape[2]

	if 'rand_crop' in args:
		rand_crop = args['rand_crop']
	if 'crop_x_start' in args:
		crop_x_start = args['crop_x_start'] 
	if 'crop_y_start' in args:
		crop_y_start = args['crop_y_start']
	if 'max_rotate_angle' in args:
		max_rotate_angle = args['max_rotate_angle']
	if 'max_aspect_ratio' in args:
		max_aspect_ratio = args['max_aspect_ratio']
	if 'max_shear_ratio' in args:
		max_shear_ratio = args['max_shear_ratio']
	if 'max_crop_size' in args:
		max_crop_size = args['max_crop_size']
	if 'min_crop_size' in args:
		min_crop_size = args['min_crop_size']
	if 'max_random_scale' in args:
		max_random_scale = args['max_random_scale']
	if 'min_random_scale' in args:
		min_random_scale = args['min_random_scale']
	if 'max_img_size' in args:
		max_img_size = args['max_img_size']
	if 'min_img_size' in args:
		min_img_size = args['min_img_size']
	if 'random_h' in args:
		random_h = args['random_h']
	if 'random_s' in args:
		random_s = args['random_s']
	if 'random_l' in args:
		random_l = args['random_l']
	if 'rotate' in args:
		rotate = args['rotate']
	if 'fill_value' in args:
		fill_value = args['fill_value']
	if 'inter_method' in args:
		inter_method = args['inter_method']
	if 'pad' in args:
		pad = args['pad']
	if 'rand_mirror' in args:
		rand_mirror = args['rand_mirror']

	
	cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] aug_img
	aug_img = np.ascontiguousarray(np.zeros((h1, w1,c1)), dtype=np.uint8)
	if nlandmarks > 0:
		augment_img(<unsigned char*>img.data, c, h, w, <unsigned char*> aug_img.data, c1, h1, w1, 
		rand_crop, crop_x_start, crop_y_start,  max_rotate_angle,
		max_aspect_ratio, max_shear_ratio, max_crop_size, min_crop_size, max_random_scale,
		min_random_scale, max_img_size, min_img_size, random_h, random_s,
		random_l, rotate, fill_value, inter_method, pad, rand_mirror, nlandmarks_, NULL, NULL)
	else:
		augment_img(<unsigned char*>img.data, c, h, w, <unsigned char*> aug_img.data, c1, h1, w1, 
                rand_crop, crop_x_start, crop_y_start,  max_rotate_angle,
                max_aspect_ratio, max_shear_ratio, max_crop_size, min_crop_size, max_random_scale,
                min_random_scale, max_img_size, min_img_size, random_h, random_s,
                random_l, rotate, fill_value, inter_method, pad, rand_mirror, nlandmarks_,
		<float*> landmarks_x.data, <float*> landmarks_y.data)	
	
	return  aug_img
