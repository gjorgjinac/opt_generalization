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
from affine_problems import *
import pandas as pd
from bbob_custom import *
from tqdm import tqdm
import itertools


def run_algorithms_on_problems(benchmark_name, problems, algorithm_names, seed, maximum_generations=50):
    population_size = 10*dimension
    sampling = LHS()
    set_random_seed(seed)
    algorithms=get_algorithms(population_size, seed, sampling)
    for algorithm_name in algorithm_names:
        file_path=f'algorithm_run_data/{benchmark_name}_{algorithm_name}_dim_{dimension}_seed_{seed}.csv'
        if os.path.isfile(file_path):
            print('Already exists ', file_path)
            continue
        best_solutions_df = pd.DataFrame()
        algorithm=algorithms[algorithm_name]
        all_populations_df=pd.DataFrame()
        
        for problem in problems:
            algorithm.termination = DefaultSingleObjectiveTermination()
            res = minimize(new_problem, termination=MaximumGenerationTermination(maximum_generations),
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
            best_ys_df['f']=problem.name
            best_ys_df['seed']=seed
            best_solutions_df=pd.concat([best_solutions_df, best_ys_df])
        except Exception as e:
            print(e)
            continue

            best_solutions_df.to_csv(file_path)     

def get_affine_problems(dimension, alphas, max_instance_id):
    problems=[]
    for problem_id_1, problem_id_2 in tqdm(list(itertools.product(list(range(1, 25)), list(range(1, 25))))):
        for instance_id in range(1,max_instance_id+1):
            for alpha in alphas:
                problems+= [AffineProblem(problem_id_1, problem_id_2, alpha, dimension, instance_id, instance_id)]

    return problems

