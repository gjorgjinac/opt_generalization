# opt_generalization

This repository contains the code for the paper "A Cross-Benchmark Examination of Feature-based Algorithm Selector Generalization in Single-Objective Numerical Optimization".

The code is meant to evaluate an Algorithm Selection (AS) model, trained on one optimization benchmark and tested on another one. This involves training a Random Forest (rf) model that takes in as input problem landscape features and predicts the scores of several algorithms. Four benchmark suites are implemented: BBOB, AFFINE, RANDOM and ZIGZAG. Please note that we use M4 and ZIGZAG to refer to the same benchmark.
To conduct the complete evaluation, several steps need to be performed:

1) Generating problem instances - generating the problem instances from all benchmarks and evaluating them on a fixed sample. This can be done using script S0_sample.sh.
The properties of the generated problem instances can be modified by changing the parameters in the config.py file:
- The number of instances per problem class to use when creating the AFFINE problem instances can be set through the property affine_max_instance_id in config.py
- The values of the alpha parameter used to combine problem instances when creating the AFFINE problem instances can be set through the property affine_alphas in config.py
- The number of instances per problem class to use from the BBOB benchmark can be set through the property bbob_max_instance_id in config.py
- The number of ZIGZAG problem isntances to be generated can be set through the property m4_problem_count in config.py
- The number of RANDOM problem isntances to be generated can be set through the property random_problem_count in config.py

2) Calculating ELA features - can be done using the script S1_calculate_ela_features.sh

3) Calculating TransOpt features - can be done using the script S1_calculate_transformer_features.sh. Assumes a transformer model for problem classification has already been trained and is saved in a file f'results_downstream/problem_classification_stats/dim_{dimension}_instances_999_samples_{sample_count_dimension_factor}_fold_{fold}_n_heads_1_n_layers_1_d_model_30_d_k_None_d_v_None_aggregations_all/trained_model_dict.pt'. If this is not the case, a model should be trained using the N1_train_transformer_model.py script

4) Algorithm Execution - running all optimization algorithms on problems from all benchmarks. This can be done using the script S1_run_algorithms.sh

5) Training RF model for algorithm selection - can be done using the script S2_run_algorithm_ranking_generalization.sh.

The jypter notebooks serve for analysis of the results.
