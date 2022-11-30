#!/bin/bash
TEST_DAY=2013-08-19
if [ $# != 1 ]
then
	echo "Enter number of flights to test" 
	echo "Enter -1 to test all the flights for" $TEST_DAY
else
	cd ShellScripts
	chmod +x backendScript.sh
	chmod +x frontendScript.sh
	chmod +x cudaScript.sh
	chmod +x firstTimeSetup.sh
	FreshRepoCheck="FreshRepo.txt"
	while read -r line
	do
		if [ $line == 1 ]
		then 
			echo -n "Performing first time setup ... "
			gnome-terminal --disable-factory --geometry 50x10+1000+10 -- /bin/bash -c './firstTimeSetup.sh'
			FirstTime=$!
			wait $FirstTime
			echo "Done"
		fi
	done < "$FreshRepoCheck"
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
	if [ $1 == -1 ]
	then
		echo "Please upload the file in the location below to the website and hit the generate solution button"
		echo ""
		echo "/Dataset/$TEST_DAY/$TEST_DAY-WebsiteInputForFullDay.txt"
		echo ""
	else
		echo "Please upload the file in the location below to the website and hit the generate solution button"
		echo ""
		echo "/ShellScripts/$TEST_DAY-WebsiteInput.txt"
		echo ""
	fi
	echo "Please execute the command below after the solution is generated" 
	echo ""
	echo "python3 ShellScripts/CompareMetric.py $TEST_DAY $1"
	echo ""
fi
