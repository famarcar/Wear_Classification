
# coding: utf-8

# ### Clasificación desgaste de materiales
# 
# ##### Fabio Martínez
# ##### Camilo Gonzalez
# ##### Cristian Viafara
# 
# ##### Universidad Industrial de Santander

# Primero: leer los datos de severo y moderado ubicados en dos diferentes carpetas. 

# In[1]:


import numpy as np
import os
import sys
import tarfile
from skimage import io
#import matplotlib.pyplot as plt

path_severo = '/home/fmartinezc/main/datasets/2018-Classified-images-wear-wom/SEVERO/'
path_moderado = '/home/fmartinezc/main/datasets/2018-Classified-images-wear-wom/MODERADO/'
images_moderado = os.listdir(path_moderado)
images_severo = os.listdir(path_severo)

print(len(images_severo))
print(len(images_moderado))
print(images_severo)




# Hog descriptor computation
#X, y = Hog_wear(moderate, severe, orient_iter, iter_pix_cell, iter_cell_block)
print(len(images_severo))
print(len(images_moderado))

def Hog_wear(orient_iter, iter_pix_cell, iter_cell_block, p_moderado=images_moderado, p_severo=images_severo):
    from skimage import io
 #   import matplotlib.pyplot as plt
    import numpy as np
    from skimage import color
    from skimage.feature import hog
    #get_ipython().run_line_magic('matplotlib', 'inline')

    X=[] # descriptor

    #moderado
    for image_path in p_moderado:
        image_file = os.path.join(path_moderado, image_path)
        image = color.rgb2gray(io.imread(image_file))
        fd = hog(image, orientations=orient_iter, pixels_per_cell=(iter_pix_cell, iter_pix_cell),
                 cells_per_block=(iter_cell_block, iter_cell_block), block_norm="L2-Hys")
        X.append(fd)
    #X = np.vstack(X,fd) 

    #severo
    for image_path in p_severo:
        image_file = os.path.join(path_severo, image_path)
        image = color.rgb2gray(io.imread(image_file))
        fd = hog(image, orientations=orient_iter, pixels_per_cell=(iter_pix_cell, iter_pix_cell),
                 cells_per_block=(iter_cell_block, iter_cell_block), block_norm="L2-Hys")
        X.append(fd)
    
    X = np.r_[X]
    y=[]

    lab_mod = np.zeros(len(images_moderado)).astype(int).transpose()
    lab_sev = np.ones(len(images_severo)).astype(int).transpose()
    y = np.concatenate( (lab_mod, lab_sev), axis=0)
    y = np.r_[y]
    #print("X.shape: ", X.shape, " y.shape: ", y.shape)
    #print(type(X))
    return X, y


# In[9]:


X, y = Hog_wear(8, 16, 3)
print("X.shape: ", X.shape, " y.shape: ", y.shape)




# # Computing with different hog histograms and cell parameters to select best parameters by using a k-fold fix parameter of 12 . 

# In[ ]:


from skimage import io
from skimage import color
from skimage.feature import hog

#import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import *



param_orient = [4,8,16,32]
param_pix_cell = [8, 16, 32, 64]
param_cell_block = [1, 2, 4]
estimators   = [ SVC()]
est_files   = ['SVC'] #'GaussianNB', GaussianNB(),
nfolds_range = range(2,len(images_severo)+len(images_moderado))


for  num_est, est_i in enumerate(estimators):
    print(est_i, num_est)
    file = open(est_files[num_est]+'.txt', 'w')
    for orient_iter in param_orient:
        for iter_pix_cell in param_pix_cell:
            for iter_cell_block in param_cell_block:            
                # Hog descriptor computation
                print('orient_iter: ', orient_iter, ' iter_pix_cell: ', iter_pix_cell,
                      ' iter_cell_block: ', iter_cell_block)
                X, y = Hog_wear(orient_iter, iter_pix_cell, iter_cell_block)
                print("X.shape: ", X.shape, " y.shape: ", y.shape)
                file.write("%s " % 'o'+str(orient_iter)+'-c'+str(iter_pix_cell)+
                               '-b'+str(iter_cell_block))
                
                    
                for folds in nfolds_range:
                    print('folds: ', folds)
                    s = cross_val_score(est_i, X, y, cv=KFold(folds, shuffle=True),
                                    scoring=make_scorer(accuracy_score))
                    print("accuracy: %.2f (+/- %.2f)"%(np.mean(s), np.std(s)))
                    file.write("%i %.2f %.2f "%(folds, np.mean(s), np.std(s)))
                file.write("\n")
                               

