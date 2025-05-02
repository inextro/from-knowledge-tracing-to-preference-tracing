#!/bin/bash

num_topics=(10 20 30 40 50)
ratios=(0.1 0.2 0.3 0.4)

for num_topic in "${num_topics[@]}"
do
    for ratio in "${ratios[@]}"
    do
        echo "Running script with num_topics=${num_topic} and ratio=${ratio}"
        python3 bkt.py -n ${num_topic} -r ${ratio}
    done
done