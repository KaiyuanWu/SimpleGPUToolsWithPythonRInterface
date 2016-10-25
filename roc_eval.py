def roc_eval(featureA_array, featureC_array, labelA = None, labelC = None, do_norm = False):
        nexamplesA = len(featureA_array)
        nexamplesC = len(featureC_array)
        if do_norm:
                featureA_norm = (featureA_array*featureA_array).sum(axis = 1)**0.5
                featureC_norm = (featureC_array*featureC_array).sum(axis = 1)**0.5
                featureA_array /= featureA_norm.reshape(-1,1)
                featureC_array /= featureC_norm.reshape(-1,1)
        score = np.zeros((nexamplesA, nexamplesC), dtype = 'float32')
        pdg.pairwise_dist_gpu2(featureA_array.astype('float32'), featureC_array.astype('float32'), score)
        #fpr = np.arange(0,1.00001, .00001)
        fpr = np.exp(np.arange(np.log(0.00001),0,0.2))
        if labelA == None or labelC == None:
                assert nexamplesA == nexamplesC
                labelA = np.arange(nexamplesA)
                labelC = np.arange(nexamplesC)
        neg_score = score[labelA.reshape(-1,1) != labelC.reshape(1,-1)]
        pos_score = score[labelA.reshape(-1,1) == labelC.reshape(1,-1)]
        neg_score = np.sort(neg_score)
        thresholds = neg_score[(fpr*neg_score.shape[0]).astype('int32')]
        tpr = np.mean(pos_score < thresholds.reshape(-1,1), axis = 1)
        return fpr, tpr, thresholds
