#!/bin/sh
# Shell script to remove files generated from Cython build
ls --color
rm *.o *.so *.cpp
rm -r build
ls --color
