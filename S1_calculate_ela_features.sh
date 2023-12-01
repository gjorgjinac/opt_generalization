#!/bin/bash

dimension=10


for benchmark in random bbob affine m4
do

 for sample_count_dimension_factor in 50 100
    do
        tsp Rscript N1_ela_feature_calculation.R "${sample_count_dimension_factor}d_samples/${benchmark}_${dimension}d_scaled.csv"
    done
done