# Extended Isolation Forest

This is an implementation for the Extended Isolation Forest method, which is described in this [paper](https://arxiv.org/pdf/1811.02141.pdf). It is an improvement on the original algorithm Isolation Forest, which is described (among other places) in this [paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf), for detecting anomalies and outliers from a data point distribution.

## Build and Installation
For regular build:


    python setup.py build_ext --inplace


For build with OpenMP:


    python setup.py build_ext --inplace --enable-openmp


or


    python setup.py --enable-openmp build_ext --inplace


For installation:


    python setup.py install


## Requirements

- numpy
- C/C++ compiler
- Cython
- [optional] OpenMP (this should come along with the GNU or Intel C/C++ compilers)
