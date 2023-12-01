#!/bin/bash

dimension=10


for benchmark in bbob affine m4 random
do
    for seed in {100..1001..100}
    do
        tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms DE --benchmark $benchmark
        tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms GA --benchmark $benchmark
        tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms ES --benchmark $benchmark
        tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms PSO --benchmark $benchmark
    done
done