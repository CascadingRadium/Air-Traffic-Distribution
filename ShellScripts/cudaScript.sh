#!/bin/bash
cd ..
cd Website/backend
if [ $1 == 0 ]
then
	echo "Latest Executable Found ... Skipping Compiling"
else
	echo -n "Outdated Executable Found ... Compiling CUDA Code ... "
	cd ../../CUDA
	nvcc GeneticAlgorithm.cu
	>OldVersionCheck.txt
	echo 0 >> OldVersionCheck.txt
	mv a.out ../Website/backend
	echo "Done"
fi
