# Import Packages
import numpy as np
import scipy as sp

class CMDS():
    def __init__(self, d: int = 2):
        '''
        Constructing the object.
        Args:
            d - Number of dimensions of the encoder output.
        '''
        #===========================Fill This===========================#
        # 1. Keep the model parameters.

        self.d = d
        self.mEigenvalues = None
        self.mEigenvectors = None
        self.mDxx = None
        #===============================================================#
        
    def fit(self, mDxx: np.ndarray):
        '''
        Fitting model parameters to the input.
        Args:
            mDxx - Input data (Distance matrix) with shape Nx x Nx.
        Output:
            self
        '''
        #===========================Fill This===========================#
        # 1. Build the model encoder.

        centered_mDxx = mDxx - mDxx.mean(axis=1).reshape(-1,1) # center columns
        centered_mDxx -= centered_mDxx.mean(axis=0).reshape(1,-1) # center rows

        mK = -0.5 * centered_mDxx

        vEigenvalues, mEigenvectors = sp.sparse.linalg.eigsh(mK, k=self.d, which='LM', return_eigenvectors=True)

        sorted_indices = np.argsort(vEigenvalues)[::-1]
        self.mEigenvalues = np.diag(vEigenvalues[sorted_indices])
        
        self.mEigenvectors = mEigenvectors[:,sorted_indices]
        self.mDxx = mDxx


        #===============================================================# 
        return self
    
    def transform(self, mDxy: np.ndarray) -> np.ndarray:
        '''
        Applies (Out of sample) encoding.
        Args:
            mDxy - Input data (Distance matrix) with shape Nx x Ny.
        Output:
            mZ - Low dimensional representation (embeddings) with shape Ny x d.
        '''
        #===========================Fill This===========================#
        # 1. Encode data using the model encoder.
        centered_mDxy = mDxy - self.mDxx.mean(axis=1).reshape(-1,1) # center columns
        centered_mDxy -= centered_mDxy.mean(axis=0).reshape(1,-1) # center rows

        mKxy = -0.5 * centered_mDxy

        mZ = np.linalg.inv(np.sqrt(self.mEigenvalues)) @ self.mEigenvectors.T @ mKxy
        
        #===============================================================#

        return mZ.T
    
    def fit_transform(self, mDxx: np.ndarray) -> np.ndarray:
        '''
        Applies encoding on the input.
        Args:
            mDxx - Input data (Distance matrix) with shape Nx x Nx.
        Output:
            mZ - Low dimensional representation (embeddings) with shape Nx x d.
        '''
        #===========================Fill This===========================#
        # 1. Encode data using the model encoder.

        self.fit(mDxx)
        mZ = np.sqrt(self.mEigenvalues) @ self.mEigenvectors.T
        #===============================================================#

        return mZ.T
