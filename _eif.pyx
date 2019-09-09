# Cython wrapper for Extended Isolation Forest

# distutils: language = C++
# distutils: sources  = eif.cxx
# cython: language_level = 3

import cython
import numpy as np
cimport numpy as np

cimport __eif

np.import_array()

cdef class iForest:
    cdef int size_Xfit
    cdef int dim
    cdef __eif.iForest* thisptr

    def __cinit__ (self, int ntrees, int sample, int limit=0, int exlevel=0, int seed=-1):
        self.thisptr = new __eif.iForest (ntrees, sample, limit, exlevel, seed)

    def __dealloc__ (self):
        del self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit (self, np.ndarray[double, ndim=2] Xfit not None):
        if not Xfit.flags['C_CONTIGUOUS']:
            Xfit = Xfit.copy(order='C')
        self.size_Xfit = Xfit.shape[0]
        self.dim = Xfit.shape[1]
        self.thisptr.fit (<double*> np.PyArray_DATA(Xfit), self.size_Xfit, self.dim)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict (self, np.ndarray[double, ndim=2] Xpred=None):
        cdef np.ndarray[double, ndim=1, mode="c"] S
        if Xpred is None:
            S = np.empty(self.size_Xfit, dtype=np.float64, order='C')
            self.thisptr.predict (<double*> np.PyArray_DATA(S), NULL, 0)
        else:
            if not Xpred.flags['C_CONTIGUOUS']:
                Xpred = Xpred.copy(order='C')
            S = np.empty(Xpred.shape[0], dtype=np.float64, order='C')
            self.thisptr.predict (<double*> np.PyArray_DATA(S), <double*> np.PyArray_DATA(Xpred), Xpred.shape[0])
        return S

    def OutputTreeNodes (self, int tree_index):
        self.thisptr.OutputTreeNodes (tree_index)
