cdef extern from "eif.hxx":
    cdef cppclass iForest:
        iForest (int, int, int, int, int)
        void fit (double*, int, int)
        void predict (double*, double*, int)
        void OutputTreeNodes (int)
