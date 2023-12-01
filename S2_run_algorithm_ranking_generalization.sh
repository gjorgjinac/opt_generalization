#!/bin/bash

dimension=3
algorithms='DE-GA-ES-PSO'
for train_benchmark in random m4 bbob affine
do
    for sample_count_dimension_factor in 50 100
    do
    
        tsp python N3_algorithm_ranking_generalization.py --train_benchmark $train_benchmark --dimension $dimension --sample_count_dimension_factor $sample_count_dimension_factor --model_name rf --algorithms $algorithms
        tsp python N3_algorithm_ranking_generalization_same_benchmark.py --train_benchmark $train_benchmark --dimension $dimension --sample_count_dimension_factor $sample_count_dimension_factor --model_name rf --algorithms $algorithms
    done
done