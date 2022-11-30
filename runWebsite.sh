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
		gnome-terminal --disable-factory --geometry 50x10+10000+10 -- /bin/bash -c './firstTimeSetup.sh'
		FirstTime=$!
		wait $FirstTime
		echo "Done"
	fi
done < "$FreshRepoCheck"

echo -n "Starting Backend ... "
gnome-terminal --geometry 50x10+10+10 -- /bin/bash -c './backendScript.sh'
echo "Done"

CudaCheck="../CUDA/OldVersionCheck.txt"
while read -r line
do
	./cudaScript.sh $line
done < "$CudaCheck"
echo -n "Starting Frontend ... "
gnome-terminal --geometry 50x10+10+1000 -- /bin/bash -c './frontendScript.sh'
echo "Done"
