#!/bin/bash
TEST_DAY=2013-08-16
if [ $# != 1 ]
then
	echo "Enter number of flights to test" 
	echo "Enter -1 to test all the flights for" $TEST_DAY
else
	cd ShellScripts
	chmod +x backendScript.sh
	chmod +x frontendScript.sh
	chmod +x cudaScript.sh
	echo -n "Starting Backend ... "
	gnome-terminal --geometry 50x10+10+10 -- /bin/bash -c './backendScript.sh'
	echo "Done"
	input="../CUDA/OldVersionCheck.txt"
	while read -r line
	do
		./cudaScript.sh $line
	done < "$input"
	echo -n "Generating the metrics for real data ... "
	python3 Metrics.py $TEST_DAY $1
	echo "Done"
	echo -n "Starting Frontend ... "
	gnome-terminal --geometry 50x10+10+1000 -- /bin/bash -c './frontendScript.sh'
	echo "Done"
	echo ""
	echo "Please upload the file /ShellScripts/$TEST_DAY-WebsiteInput.txt to the website and hit the generate solution button"
	echo "Please execute the command" 
	echo ""
	echo "python3 ShellScripts/CompareMetric.py $TEST_DAY $1"
	echo ""
	echo "after the solution is generated"
fi
