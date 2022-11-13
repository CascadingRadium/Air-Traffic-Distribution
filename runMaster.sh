#!/bin/bash
cd ShellScripts
chmod +x backendScript.sh
chmod +x frontendScript.sh
chmod +x cudaScript.sh
echo -n "Starting Backend ... "
gnome-terminal --geometry 50x10+10+10 -- /bin/bash -c './backendScript.sh'
echo "Done"
echo -n "Starting Frontend ... "
gnome-terminal --geometry 50x10+10+1000 -- /bin/bash -c './frontendScript.sh'
echo "Done"
./cudaScript.sh
