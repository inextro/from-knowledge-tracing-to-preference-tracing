#!/bin/bash

# num_topics=(20 30 40)
num_topic=40
emb_dims=(32 64)

for emb_dim in "${emb_dims[@]}"
do
    echo "Running script with num_topics=${num_topic} and emb_dim=${emb_dim}"
    python3 baseline_performance.py -n ${num_topic} -e ${emb_dim}
done