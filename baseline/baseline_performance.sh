#!/bin/bash

num_topics=(20 30 40)
emb_dims=(32 64)

for num_topic in "${num_topics[@]}"
do
    for emb_dim in "${emb_dims[@]}"
    do
        echo "Running script with num_topics=${num_topic} and emb_dim=${emb_dim}"
        uv run baseline_performance.py -n ${num_topic} -e ${emb_dim}
    done
done