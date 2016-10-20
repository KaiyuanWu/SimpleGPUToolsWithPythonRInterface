# SimpleToolsWithPythonRInterface

1. PairWise Distance Calculation

  (a) Build Step
  
    Check the setup.py, make sure the include path and lib path of cuda&cublas are correct

    nvcc -ccbin=c++ -Xcompiler -fPIC -I/usr/local/cuda-7.5/include  -c pairwise_distance.cu  -o pairwise_distance.o     
    python pdg_setup.py build_ext --inplace
    
  (b) Test
  
    import pairwise_distance_gpu as pdg
    
    x = np.random.randn(100,3000).astype('float32')
    
    y = np.random.randn(100,2000).astype('float32')
    
    dist = np.zeros((100,100)).astype('float32')
    
    pdg.pairwise_dist_gpu1(x,dist)
    
    print(x)
    
    print(dist)
    
    pdg.pairwise_dist_gpu2(x,y,dist)
    
    print(x,y)
    
    print(dist)


2. Image Augment

  (a) Build Step
    
     g++ -fPIC -std=c++11 -c img_aug.cpp -o img_aug.o
     
     python ImageAugmenterPySetup.py  build_ext --inplace
     
  (b) Test
  
  Test code piece
  
  import cv2
  
  import ImageAugmenterPy
  
  img = cv2.imread('/tmp/test/test.jpg')
  
  args = {'data_shape':(3, 100,100)}
  
  aug_img = ImageAugmenterPy.augment_img_process(img,args=args)
  
  print(aug_img.shape)
  
  cv2.imshow("aug_img", aug_img)
  
  cv2.waitKey()
     
