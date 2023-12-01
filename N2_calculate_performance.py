import pandas as pd
import numpy as np
from config import *

import argparse

def get_all_runs(benchmark, benchmark_ids,algorithms, dimension):
    all_runs_df=pd.DataFrame()
    for algorithm_name in algorithms:
        for seed in np.arange(100,1001,100):
            df=pd.read_csv(f'{data_dir}/algorithm_runs/{benchmark}_{algorithm_name}_dim_{dimension}_seed_{seed}.csv', index_col=0)
            all_runs_df=pd.concat([all_runs_df,df])
    return all_runs_df.set_index(benchmark_ids)


def get_best_algorithm_per_instance(dimension, runs_df,id_columns,benchmark, drop_instances_with_multiple_best_algorithms,algorithm_portfolio_name, budgets=[10,30,50]):
    all_budget_results=pd.DataFrame()
    for budget in budgets:
        best_y_per_algorithm_per_run=runs_df.query('iteration<@budget').groupby(id_columns+['algorithm_name','seed']).min()['best_y'].to_frame().reset_index()
        best_y_in_each_run=best_y_per_algorithm_per_run.groupby(id_columns+['seed']).min()['best_y'].reset_index()
        best_algorithm_per_seed=best_y_in_each_run.merge(best_y_per_algorithm_per_run)
        count_best_algorithm_per_instance=best_algorithm_per_seed.groupby(id_columns+['algorithm_name']).count()['best_y'].reset_index()
        best_count_per_instance=count_best_algorithm_per_instance.groupby(id_columns)['best_y'].max().reset_index()
        best_algorithms_per_instance=count_best_algorithm_per_instance.merge(best_count_per_instance)


        if drop_instances_with_multiple_best_algorithms:
            instances_with_multiple_best_algorithms=best_algorithms_per_instance.groupby(id_columns).count().query('best_y>1')
            best_algorithms_per_instance=best_algorithms_per_instance.set_index(id_columns)
            best_algorithms_per_instance=best_algorithms_per_instance.drop(instances_with_multiple_best_algorithms.index)
                                                   
        best_algorithms_per_instance['budget']=budget
        all_budget_results=pd.concat([all_budget_results, best_algorithms_per_instance])
        
    if 'f' in all_budget_results.columns:
        all_budget_results=all_budget_results.set_index('f')
    all_budget_results.to_csv(f'{data_dir}/algorithm_performance/{benchmark}_{dimension}d_{algorithm_portfolio_name}_best_algorithm_per_instance.csv')
    return all_budget_results


def rank_algorithms(dimension,runs_df,id_columns,benchmark, algorithm_portfolio_name, budgets=[10,30,50]):
    all_budget_results=pd.DataFrame()
    algorithm_count=runs_df['algorithm_name'].drop_duplicates().shape[0]
    print(algorithm_count)
    for budget in budgets:
        t=runs_df.query('iteration<@budget').reset_index().groupby(id_columns+['algorithm_name']).median().sort_values(id_columns+['best_y']).reset_index()
        t['algorithm_rank']=[x%algorithm_count for x in t.index]
        t['budget']=budget
        all_budget_results= pd.concat([ all_budget_results, t])
    all_budget_results=all_budget_results.set_index('f')
    all_budget_results.to_csv(f'{data_dir}/algorithm_performance/{benchmark}_{dimension}d_{algorithm_portfolio_name}_ranks.csv')
    return all_budget_results
        
    


def columns_normalize_algorithm_score_new(dimension,runs_df,id_columns,benchmark,algorithm_portfolio_name,  budgets=[10,30,50]):
    all_budget_results=pd.DataFrame()
    all_budget_results_not_aggregated=pd.DataFrame()
    for budget in budgets:
        best_solution_in_algorithm_run=runs_df.query('iteration<@budget').reset_index().groupby(id_columns+['algorithm_name','seed']).min().drop(columns=['iteration'])
        best_solution_in_run=best_solution_in_algorithm_run.reset_index().groupby(['f','seed']).min().rename(columns={'best_y':'best_y_among_all'}).drop(columns=['algorithm_name'])
        worst_solution_in_run=best_solution_in_algorithm_run.reset_index().groupby(['f','seed']).max().rename(columns={'best_y':'worst_y_among_all'}).drop(columns=['algorithm_name'])
        t=best_solution_in_algorithm_run.merge(best_solution_in_run, left_on=['f','seed'],right_index=True).merge(worst_solution_in_run, left_on=['f','seed'],right_index=True)
        t['algorithm_rank']=t.apply(lambda x: (x['best_y']-x['best_y_among_all'])/(x['worst_y_among_all']-x['best_y_among_all']), axis=1)
        t_copy=t.copy()
        t_copy['budget']=budget
        all_budget_results_not_aggregated=pd.concat([all_budget_results_not_aggregated,t_copy])
        t=t.reset_index().groupby(['f','algorithm_name']).median()
        t['budget']=budget
        all_budget_results= pd.concat([ all_budget_results, t])
    print(all_budget_results.shape)
    all_budget_results=all_budget_results.replace([np.inf,-np.inf], np.nan).dropna()
    print(all_budget_results.shape)
    all_budget_results=all_budget_results.reset_index().set_index('f')
    all_budget_results.to_csv(f'{data_dir}/algorithm_performance/{benchmark}_{dimension}d_{algorithm_portfolio_name}_column_normalized_score.csv')
    
    
    all_budget_results_not_aggregated=all_budget_results_not_aggregated.replace([np.inf,-np.inf], np.nan).dropna()
    print(all_budget_results_not_aggregated.shape)
    all_budget_results_not_aggregated=all_budget_results_not_aggregated.reset_index().set_index('f')
    all_budget_results_not_aggregated.to_csv(f'{data_dir}/algorithm_performance/{benchmark}_{dimension}d_{algorithm_portfolio_name}_column_normalized_score_not_aggregated.csv')
    return all_budget_results

parser = argparse.ArgumentParser(description="A script that calculates best algorithm and algorithm ranks for a given algorithm portfolio")
parser.add_argument("--algorithms", type=str, help="A dash-separated list of algorithms")
parser.add_argument("--dimension", type=int, help="A dash-separated list of algorithms")

args = parser.parse_args()
dimension=args.dimension
algorithms=args.algorithms.split('-')
id_columns=['f']
benchmarks=['bbob','affine','random','m4']
for benchmark in ['affine','bbob']:
    print(benchmark)
    runs_df=get_all_runs(benchmark,id_columns,algorithms, dimension)
    columns_normalize_algorithm_score_new(dimension,runs_df, id_columns, benchmark, args.algorithms)
    #columns_normalize_algorithm_score(dimension,runs_df, id_columns, benchmark, args.algorithms)
    get_best_algorithm_per_instance(dimension,runs_df, id_columns, benchmark, True, args.algorithms)
    rank_algorithms(dimension,runs_df, id_columns, benchmark, args.algorithms)