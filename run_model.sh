#!/bin/bash

echo $1 1>&2
#if [-f Model/$2]
#	echo "Model/$2 - Found"
#else
mkdir Model/$2
chmod 777 Model/$2
#fi
#if [-f Model/$2/$2.txt]
#	echo "Model/$2/$2.txt - Found"
#else
touch Model/$2/$2.txt
chmod 777 Model/$2/$2.txt
#fi
source ../DP_env/bin/activate
nice -n 10 python $1
