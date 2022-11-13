#!/bin/bash
cd ..
cd Website/backend
if [ -e a.out ]
then
	echo "Executable Found : Skipping Compiling"
else
	echo -n "Compiling CUDA Code ... "
	cd ../../CUDA
	nvcc GeneticAlgorithm.cu
	mv a.out ../Website/backend
	echo "Done"
fi
