# SimpleGPUToolsWithPythonRInterfce
1. PairWise Distance Calculation
  (a) Build Step
    Check the setup.py, make sure the include path and lib path of cuda&cublas are correct
    python setup.py build_ext --inplace
  (b) Test
    import pairwise_distance_gpu as pdg
    x = np.random.randn(100,3000)
    y = np.zeros((100,100))
    pdg.pairwise_dist_gpu(x,y)
    print(x)
    print(y)
    
