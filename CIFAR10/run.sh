##TODO: run.sh (command: sh run.sh) file to manage servers and clients.

echo "Starting server"
python server.py & sleep 10

for i in 'seq 0 19'; do #To change number of clients change the 'seq 0 19'
    echo "Starting client $i"
    python client.py --partition=${i} &
done 

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

