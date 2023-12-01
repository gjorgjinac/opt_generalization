#!/bin/bash

dimension=10


for benchmark in random bbob affine m4
do

 for sample_count_dimension_factor in 50 100
    do
        for fold in {0..9}
        do
        tsp python N2_get_transformer_embeddings.py --dimension $dimension --sample_count_dimension_factor $sample_count_dimension_factor --benchmark $benchmark --problem_classification 1 --fold $fold
        done
    done
done