import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from modulesRandFunc import generate_exp2fun as genExp2fun
from modulesRandFunc import generate_tree as genTree
from modulesRandFunc import generate_tree2exp as genTree2exp
import numpy as np
import tensorflow as tf
import sys
import warnings
import pandas as pd
from config import *
import os
from tqdm import tqdm
import itertools
from pymoo.problems import get_problem
import numpy as np
from problem_classes import *
from N_m4_function_utils import *

def get_samples_for_dimension(dimension,sample_count,sample_range=None):
    sample_dir='samples' if sample_range is None else f"samples_range_{sample_range}"
    samples_df=pd.read_csv(f'{data_dir}/lhs_{sample_dir}/{dimension}d_{sample_count}.csv', index_col=0)
    print(samples_df.shape)
    return samples_df


def generate_random_functions(dimension, n_functions,name, tree_min=2, tree_max=16):
    random_functions_file_name=f'{data_dir}/random_functions/{name}.csv'
    if os.path.isfile(random_functions_file_name):
        return list(pd.read_csv(random_functions_file_name)['function'].values)
    functions=set()
    while len(functions)<n_functions:
        tree = genTree.generate_tree(tree_min,tree_max)
        exp = genTree2exp.generate_tree2exp(tree)
        fun = genExp2fun.generate_exp2fun(
            exp, dimension, 1
        )
        functions.add(fun)
    functions=list(functions)
    
    df=pd.DataFrame(functions).reset_index()
    df.columns=['function_id','function']
    df.to_csv(random_functions_file_name)
    return functions
    
def sample_random_functions(functions, samples_df, name, sample_count_dimension_factor, sample_range=None):
    all_samples=pd.DataFrame()
    array_x=samples_df.values
    sample_dir='samples' if sample_range is None else f"samples_range_{sample_range}"
    for index, f in enumerate(functions):
        try:
            array_y=eval(f)
            sample_f=samples_df.copy()
            sample_f['y']=array_y
            sample_f['f']=f
            all_samples=pd.concat([all_samples,sample_f])
        except Exception as e:
            continue
            
    all_samples.set_index('f').to_csv(f'{data_dir}/{sample_dir}/{sample_count_dimension_factor}d_samples/{name}.csv')
    
def generate_and_sample_random_functions(dimension, sample_count_dimension_factor,sample_range, n_functions,  tree_min=2, tree_max=16):
    name=f'random_{dimension}d'
    functions=generate_random_functions(dimension, n_functions, name, tree_min, tree_max)
    samples_df=get_samples_for_dimension(dimension, sample_count=sample_count_dimension_factor,sample_range=sample_range)
    sample_random_functions(functions, samples_df, name,sample_count_dimension_factor, sample_range=sample_range)
    
    

def sample_affine_problems(dimension,sample_count_dimension_factor, alphas, max_instance_id, sample_range=None):
    name=f'affine_new_{dimension}d'
    sample_dir='samples' if sample_range is None else f"samples_range_{sample_range}"
    samples_df=get_samples_for_dimension(dimension, sample_count=sample_count_dimension_factor)
    problems_list = list(range(1, 25))
    all_samples=pd.DataFrame()
    for problem_id1, problem_id2 in tqdm(list(itertools.product(problems_list, problems_list))):
        if problem_id1!=problem_id2:
            for instance_id in range(1,max_instance_id+1):

                for alpha in [0.05,0.95]:
                    new_problem = AffineProblem(problem_id1, problem_id2, alpha, dimension, instance_id, instance_id)
                    ys=new_problem._evaluate(samples_df.values,{})['F']
                    new_problem_sample_df=samples_df.copy()
                    new_problem_sample_df['y']=ys
                    new_problem_sample_df['f']=new_problem.name
                    all_samples=pd.concat([all_samples,new_problem_sample_df])
                
    all_samples=all_samples.set_index('f')
    all_samples.to_csv(f'{data_dir}/{sample_dir}/{sample_count_dimension_factor}d_samples/{name}.csv')
            
        
        


def sample_M4_problems(dimension,sample_count_dimension_factor, n_functions, sample_range=None):
    name=f'm4_{dimension}d'
    sample_dir='samples' if sample_range is None else f"samples_range_{sample_range}"
    samples_df=get_samples_for_dimension(dimension,sample_count_dimension_factor)
    all_samples=pd.DataFrame()
    if not os.path.isfile(f'{data_dir}/random_functions/m4_{dimension}d.csv'):
        print('generating new M4 problems')
        function_ids=init_m4_functions(n_functions,dimension)
        save_m4_functions_parameters(function_ids, dimension)
    else:
        print('using already generated M4 problems')
        function_data=pd.read_csv(f'{data_dir}/random_functions/m4_{dimension}d.csv',index_col=0)
        function_ids = [(get_function_from_parameters(row),row['function_id']) for index, row in function_data.iterrows()]
        
    
    for function, function_id in function_ids:
        problem = M4Problem(function=function,name=function_id, dimension=dimension)
        ys=problem._evaluate(samples_df.values,{})['F']
        problem_sample_df=samples_df.copy()
        problem_sample_df['y']=ys
        problem_sample_df['f']=problem.name
        all_samples=pd.concat([all_samples,problem_sample_df])
                
    all_samples=all_samples.set_index('f')
    
    all_samples.to_csv(f'{data_dir}/{sample_dir}/{sample_count_dimension_factor}d_samples/{name}.csv')
    
    
def sample_bbob_problems(dimension,sample_count_dimension_factor, sample_range=None):
    name=f'bbob_{dimension}d'
    sample_dir='samples' if sample_range is None else f"samples_range_{sample_range}"
    samples_df=get_samples_for_dimension(dimension,sample_count_dimension_factor)
    all_samples=pd.DataFrame()
    for problem_id in tqdm(range(1, 25)):
        print(problem_id)
        for instance_id in range(1, bbob_max_instance_id+1):
            problem_name = f'bbob-{problem_id}-{instance_id}'
            problem =BBOBProblem(problem_name, n_var=dimension)

            sample_set_df = samples_df.copy()
            sample_set_df['y'] = problem.evaluate(samples_df.values)
            sample_set_df['f'] = f'{problem_id}_{instance_id}'
            all_samples =pd.concat([ all_samples,sample_set_df])
    all_samples=all_samples.set_index('f')
    all_samples.to_csv(f'{data_dir}/{sample_dir}/{sample_count_dimension_factor}d_samples/bbob_{dimension}d.csv')

    
def sample_cec_problems(dimension,sample_count_dimension_factor, sample_range=None):
    name=f'bbob_{dimension}d'
    sample_dir='samples' if sample_range is None else f"samples_range_{sample_range}"
    samples_df=get_samples_for_dimension(dimension,sample_count_dimension_factor)
    all_samples=pd.DataFrame()
    for problem in tqdm(get_cec_problems(dimension)):
        sample_set_df = samples_df.copy()
        sample_set_df['y'] = problem.evaluate(samples_df.values)
        sample_set_df['f'] = problem.name
        all_samples =pd.concat([ all_samples,sample_set_df])
    all_samples=all_samples.set_index('f')
    all_samples.to_csv(f'{data_dir}/{sample_dir}/{sample_count_dimension_factor}d_samples/cec_{dimension}d.csv')
