# SimpleGPUToolsWithPythonRInterface

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
