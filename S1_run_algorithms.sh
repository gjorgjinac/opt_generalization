#!/bin/bash

dimension=10


for benchmark in bbob affine
do
    for seed in {100..1001..100}
    do
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms PSO1 --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms PSO2 --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms PSO3 --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms PSO4 --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms PSO8 --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms PSO10 --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms DE --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms GA --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms ES --benchmark $benchmark
        #tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms PSO --benchmark $benchmark
        tsp python N1_run_algorithms.py --dimension $dimension --seed $seed --algorithms CMAES --benchmark $benchmark
    done
done