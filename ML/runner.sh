while(true)
do 
    ./runTest.py  
    if [ $? -ne 0 ] 
    then
        break 
    else
        echo "all good"
    fi
done
