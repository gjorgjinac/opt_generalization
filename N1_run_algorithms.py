import os
import random
import pandas as pd
import numpy as np
import opfunu
import numpy as np
from pymoo.operators.sampling.lhs import LHS
from pymoo.problems import get_problem
from pymoo.optimize import minimize
import pandas as pd
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.core.problem import Problem
import sys
from algorithms_init import *
from problem_classes import *
import pandas as pd
from tqdm import tqdm
import itertools
from config import *
from N_m4_function_utils import *
import argparse
import traceback
def run_algorithms_on_problems(dimension,benchmark_name, problems, algorithm_names, seed, maximum_generations=50):
    population_size = 10*dimension
    sampling = LHS()
    set_random_seed(seed)
    algorithms=get_algorithms(population_size, seed, sampling)
    for algorithm_name in algorithm_names:
        file_path=f'{data_dir}/algorithm_runs/{benchmark_name}_{algorithm_name}_dim_{dimension}_seed_{seed}.csv'
        '''if os.path.isfile(file_path):
            print('Already exists ', file_path)
            continue'''
        best_solutions_df = pd.DataFrame()
        algorithm=algorithms[algorithm_name]
        all_populations_df=pd.DataFrame()
        print(algorithm_name)
        problem_id=0
        for problem in tqdm(problems):
            problem_id+=1
            try:
                algorithm.termination = DefaultSingleObjectiveTermination()
                res = minimize(problem, termination=MaximumGenerationTermination(maximum_generations),
                                           algorithm=algorithm, save_history=True,
                                           seed=seed,
                                           verbose=False)

                all_ys=[]
                best_ys=[]
                for iteration_index, iteration in enumerate(res.history):

                    for population_individual in iteration.pop:
                        all_ys+=[population_individual.F]

                    iteration_min=np.array(all_ys).min()
                    best_ys+=[(iteration_index, iteration_min)]
                best_ys_df=pd.DataFrame(best_ys, columns=['iteration','best_y'])
                best_ys_df['algorithm_name'] = algorithm_name
                
                f=problem.name
                if 'bbob' in problem.name:
                    f=problem.name.split('-')[1]+'_'+problem.name.split('-')[2]
                best_ys_df['f']=f
                best_ys_df['seed']=seed
                best_solutions_df=pd.concat([best_solutions_df, best_ys_df])
            except Exception as e:
                print("exception")
                print(e)
                continue
            
            if problem_id%1000==0:
                best_solutions_df.to_csv(file_path) 
        best_solutions_df.to_csv(file_path)


parser = argparse.ArgumentParser(description="A script that runs the algorithms on all benchmark functions")


parser.add_argument("--dimension", type=int, help="The problem dimension")
parser.add_argument("--seed", type=int, help="The random seed to be set")
parser.add_argument("--algorithms", type=str, help="A dash-separated list of algorithms")
parser.add_argument("--benchmark", type=str, help="Benchmark name")
args = parser.parse_args()

algorithm_names=args.algorithms.split('-')
dimension=args.dimension
seed=args.seed
benchmark=args.benchmark
problems=get_problems(benchmark,dimension)

run_algorithms_on_problems(dimension=dimension, benchmark_name=benchmark, problems=problems, algorithm_names=algorithm_names, seed=seed, maximum_generations=maximum_generations_to_run_algorithms)