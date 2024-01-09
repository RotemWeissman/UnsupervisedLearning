# Import Packages

# General Tools
import numpy as np
import scipy as sp

# Machine Learning

# Computer Vision

# Statistics
from scipy.stats import multivariate_normal as MVN

# Miscellaneous
import os
import math
from platform import python_version
import random
import time
import urllib.request

# Typing
from typing import Callable, List, Tuple, Union

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Jupyter
from IPython import get_ipython
from IPython.display import Image, display
from ipywidgets import Dropdown, FloatSlider, interact, IntSlider, Layout

def squareDistance(mX: np.ndarray, mC: np.ndarray) -> np.ndarray:
    '''
    sqaure distance calculation, Euclidean based
    Args:
    mX      - Input data with shape N x d.
    mC      - Centroids with shape K x d.
    Output:
    mD      - Distance matrix between each data point to every centroid
    '''
    mD = []
    
    for c in mC: 
        d_square = np.sum((mX - c)**2, axis=1)
        mD.append(d_square)
    
    mD = np.array(mD)

    return mD

#===========================Fill This===========================#
def InitKMeans(mX: np.ndarray, K: int, initMethod: int = 0, seedNum: int = 123) -> np.ndarray:
    '''
    K-Means algorithm initialization.
    Args:
        mX          - Input data with shape N x d.
        K           - Number of clusters.
        initMethod  - Initialization method: 0 - Random, 1 - K-Means++.
        seedNum     - Seed number used.
    Output:
        mC          - The initial centroids with shape K x d.
    Remarks:
        - Given the same parameters, including the `seedNum` the algorithm must be reproducible.
    '''
    np.random.seed(seedNum)

    if initMethod == 0: # random centroids
        mC = mX[np.random.choice(mX.shape[0], size=K, replace=False)]
    
    elif initMethod == 1: # k-means++
        mC = mX[np.random.choice(mX.shape[0], size=1, replace=False)] # first centroid in random
        
        for i in range(1,K): # until K centroids are selected
            mD_square = squareDistance(mX, mC)
            min_d = np.min(mD_square, axis=0)
            weights = ( min_d / np.sum(min_d))
            newC = mX[np.random.choice(mX.shape[0], p=weights, size=1, replace=False)]
            mC = np.vstack((mC, newC))

    return mC
#===============================================================#

#===========================Fill This===========================#
def CalcKMeansObj(mX: np.ndarray, mC: np.ndarray) -> float:
    '''
    K-Means algorithm.
    Args:
        mX          - The data with shape N x d.
        mC          - The centroids with shape K x d.
    Output:
        objVal      - The value of the objective function of the KMeans.
    Remarks:
        - The objective function uses the squared euclidean distance.
    '''
    
    mD_square = squareDistance(mX,mC)
    
    objVal = mD_square.min(axis=0).sum()

    return objVal
#===============================================================#

#===========================Fill This===========================#
def KMeans(mX: np.ndarray, mC: np.ndarray, numIter: int = 1000, stopThr: float = 0) -> np.ndarray:
    '''
    K-Means algorithm.
    Args:
        mX          - Input data with shape N x d.
        mC          - The initial centroids with shape K x d.
        numIter     - Number of iterations.
        stopThr     - Stopping threshold.
    Output:
        mC          - The final centroids with shape K x d.
        vL          - The labels (0, 1, .., K - 1) per sample with shape (N, )
        lO          - The objective value function per iterations (List).
    Remarks:
        - The maximum number of iterations must be `numIter`.
        - If the objective value of the algorithm doesn't improve by at least `stopThr` the iterations should stop.
    '''

    lO = [CalcKMeansObj(mX,mC)]
    vL = []

    for i in range(numIter):
        # assume fixed centroids, find best clusters:
        
        #calculate distances:
        mD = squareDistance(mX,mC)
        # assign clusters
        vL = mD.argmin(axis=0)

        # assume fixed clusters, find best centroids:
        new_mC = []
        for j in range(mC.shape[0]):
            cluster_points = mX[np.where(vL == j)[0]]
            centroid_j = cluster_points.mean(axis=0)
            new_mC.append(centroid_j)
        new_mC = np.asarray(new_mC).astype(np.float)
        mC = new_mC
        inertia = CalcKMeansObj(mX,mC)
        lO.append(inertia)
        if (lO[-2] - lO[-1]) <= stopThr:
            break

    return mC, vL, lO

#===============================================================#

#===========================Fill This===========================#
def InitGmm(mX: np.ndarray, K: int, seedNum: int = 123) -> np.ndarray:
    '''
    GMM algorithm initialization.
    Args:
        mX          - Input data with shape N x d.
        K           - Number of clusters.
        seedNum     - Seed number used.
    Output:
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
    Remarks:
        - Given the same parameters, including the `seedNum` the algorithm must be reproducible.
        - mμ Should be initialized by the K-Means++ algorithm.
    '''
    np.random.seed(seedNum)
    
    # Initialize mμ with k-means++ algorithm:
    mμ = mX[np.random.choice(mX.shape[0], size=1, replace=False)]
        
    for i in range(1,K):
        # calculate square distances
        mD_square = []
        for μ in mμ:
            d_square = np.sum((mX - μ)**2, axis=1)
            mD_square.append(d_square)
        mD_square = np.array(mD_square)
        # choose next centroid
        min_d = np.min(mD_square, axis=0)
        weights = ( min_d / np.sum(min_d))
        new_mμ = mX[np.random.choice(mX.shape[0], p=weights, size=1, replace=False)]
        mμ = np.vstack((mμ, new_mμ))

    # Initialize tΣ:
    tΣ = np.empty((mX.shape[1], mX.shape[1], K))
    for k in range(K):
        tΣ[:,:,k] = np.diag(np.var(mX, axis=0))
    
    # Initialize vW:
    vW = np.random.uniform(size=K)
    vW = vW/vW.sum()

    return mμ, tΣ, vW
#===============================================================#

#===========================Fill This===========================#
def CalcGmmObj(mX: np.ndarray, mμ: np.ndarray, tΣ: np.ndarray, vW: np.ndarray) -> float:
    '''
    GMM algorithm objective function.
    Args:
        mX          - The data with shape N x d.
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
    Output:
        objVal      - The value of the objective function of the GMM.
    Remarks:
        - A
    '''

    objVal = 0
    for i in range(len(vW)):
        objVal += np.log(np.sum(vW[i] * MVN.pdf(x=mX, mean=mμ[i], cov=tΣ[:,:,i])))
    return objVal
#===============================================================#

#===========================Fill This===========================#
def GMM(mX: np.ndarray, mμ: np.ndarray, tΣ: np.ndarray, vW: np.ndarray, numIter: int = 1000, stopThr: float = 1e-5) -> np.ndarray:
    '''
    GMM algorithm.
    Args:
        mX          - Input data with shape N x d.
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
        numIter     - Number of iterations.
        stopThr     - Stopping threshold.
    Output:
        mμ          - The final mean vectors with shape K x d.
        tΣ          - The final covariance matrices with shape (d x d x K).
        vW          - The final weights of the GMM with shape K.
        vL          - The labels (0, 1, .., K - 1) per sample with shape (N, )
        lO          - The objective function value per iterations (List).
    Remarks:
        - The maximum number of iterations must be `numIter`.
        - If the objective value of the algorithm doesn't improve by at least `stopThr` the iterations should stop.
    '''

    lO  = [CalcGmmObj(mX, mμ, tΣ, vW)]

    for i in range(numIter):
        # Expectation step:
        mP = np.empty((mX.shape[0], len(vW)))
        for j in range(len(vW)):
            mP[:,j] = vW[j] * MVN.pdf(x=mX, mean=mμ[j], cov=tΣ[:,:,j])
        mP /= mP.sum(axis=1, keepdims=True)
        
        # Maximization step:
        for j in range(len(vW)):
            Nk = np.sum(mP[:,j])
            vW[j] = Nk / mX.shape[0]
            mμ[j] = (mP[:,j] @ mX) / Nk
            tΣ[:,:,j] = mP[:,j] * (mX - mμ[j]).T @ (mX - mμ[j]) / Nk

        lO.append(CalcGmmObj(mX,mμ,tΣ,vW))

        if np.abs(lO[-2] - lO[-1]) <= stopThr:
            break

    vL = np.argmax(mP, axis=1)

    return mμ, tΣ, vW, vL, lO
#===============================================================#