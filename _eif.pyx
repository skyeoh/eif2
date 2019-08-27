# Cython wrapper for Extended Isolation Forest

# distutils: language = C++
# distutils: sources  = eif.cxx
# cython: language_level = 3

import cython
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "eif.hxx":
    cdef cppclass iForest:
        iForest (int, int, int, int)
        void CheckExtensionLevel ()
        void fit (double*, int, int)
        void predict (double*, double*, int)

cdef class PyiForest:
    cdef int size_Xfit
    cdef int dim
    cdef iForest* thisptr

    def __cinit__ (self, int ntrees, int sample, int limit=0, int exlevel=0):
        self.thisptr = new iForest (ntrees, sample, limit, exlevel)

    def __dealloc__ (self):
        del self.thisptr

    def CheckExtensionLevel (self):
        self.thisptr.CheckExtensionLevel ()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit (self, np.ndarray[double, ndim=2, mode="c"] Xfit not None):
        self.size_Xfit = Xfit.shape[0]
        self.dim = Xfit.shape[1]
        self.thisptr.fit (<double*> np.PyArray_DATA(Xfit), self.size_Xfit, self.dim)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict (self, np.ndarray[double, ndim=2, mode="c"] Xpred=None):
        cdef np.ndarray[double, ndim=1, mode="c"] S
        if Xpred is None:
            S = np.empty(self.size_Xfit, dtype=np.float64, order='C')
            self.thisptr.predict (<double*> np.PyArray_DATA(S), NULL, 0)
        else:
            S = np.empty(Xpred.shape[0], dtype=np.float64, order='C')
            self.thisptr.predict (<double*> np.PyArray_DATA(S), <double*> np.PyArray_DATA(Xpred), Xpred.shape[0])
        return S
