from pymoo.core.problem import Problem
import numpy as np
import ioh
from config import *
from tqdm import tqdm
import itertools
import pandas as pd
from N_m4_function_utils import *
import opfunu
class AffineProblem(Problem):
    
     def __init__(self, problem_id_1, problem_id_2, alpha, dim, instance_id_1, instance_id_2):

        super().__init__(n_var=dim, n_obj=1, n_constr=0, xl=-5, xu=5)
        self.name=f'p{problem_id_1}_p{problem_id_2}_i{instance_id_1}_i{instance_id_2}_a{alpha}'
        self.problem_id_1=problem_id_1
        self.problem_id_2=problem_id_2
        self.instance_id_1=instance_id_1
        self.instance_id_2=instance_id_2
        self.dim=dim

        self.alpha=alpha
    
     def _evaluate(self, x_list, out, *args, **kwargs):
            
        f1 = ioh.get_problem(self.problem_id_1, self.instance_id_1, self.dim)
        f2 = ioh.get_problem(self.problem_id_2, self.instance_id_2, self.dim)
        o1=f1.optimum.y
        o2=f2.optimum.y
        out['F'] =[np.exp(self.alpha * np.log(np.clip(f1(x) - o1, 1e-12, 1e12)) + (1-self.alpha) * np.log(np.clip(f2(x - f1.optimum.x + f2.optimum.x) - o2, 1e-12, 1e12))) for x in x_list]
        return out
    
    

class CECProblem(Problem):

    def __init__(self,f,dimension):
        super().__init__(n_var=dimension, n_obj=1, xl=-100, xu=100)
        self.f=f
        self.dimension=dimension
        self.name=f
        
    def _evaluate(self, x, out, *args, **kwargs):
        try:
            f_dim=self.f(ndim=self.dimension, m_group=int(self.dimension/5))
        except TypeError:
            f_dim=self.f(ndim=self.dimension)
        try:
            out["F"] = [f_dim.evaluate(xx) for xx in x]
        except Exception as e:
            print("EXCEPTION")
            print(e)

            

class RandomProblem(Problem):

    def __init__(self,f,dimension):
        super().__init__(n_var=dimension, n_obj=1, xl=-5, xu=5)
        self.f=f
        self.dimension=dimension
        self.name=f
        
        
    def _evaluate(self, x, out, *args, **kwargs):
        array_x = np.array(x)
        f_value = eval(self.f)
        out["F"] = f_value
        return out

        
        
        
def get_bbob(name, n_var=10, **kwargs):
    try:
        import cocoex as ex
    except:
        raise Exception("COCO test suite not found. \nInstallation Guide: https://github.com/numbbo/coco")

    args = name.split("-")
    suite = args[0]
    n_instance = int(args[-1])
    n_function = int(args[-2].replace("f", ""))

    assert 1 <= n_function <= 24, f"BBOB has 24 different functions to be chosen. {n_function} is out of range."

    suite_filter_options = f"function_indices: {n_function} " \
                           f"instance_indices: {n_instance} " \
                           f"dimensions: {n_var}"

    problems = ex.Suite(suite, "instances: 1-999", suite_filter_options)
    assert len(problems) == 1, "COCO problem not found."

    coco = problems.next_problem()

    return n_function, n_instance, coco

class BBOBProblem(Problem):

    def __init__(self, name, n_var, pf_from_file=True, **kwargs):
        self.function, self.instance, self.object = get_bbob(name, n_var)
        self.name = name
        self.pf_from_file = pf_from_file

        coco = self.object
        n_var, n_obj, n_ieq_constr = coco.number_of_variables, coco.number_of_objectives, coco.number_of_constraints
        xl, xu = coco.lower_bounds, coco.upper_bounds

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_ieq_constr=n_ieq_constr,
                         xl=xl,
                         xu=xu,
                         **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        if self.n_obj == 1:
            fname = '._bbob_problem_best_parameter.txt'

            self.object._best_parameter(what="print")
            ps = np.loadtxt(fname)
            os.remove(fname)

            return ps

    def _calc_pareto_front(self, *args, **kwargs):
        if self.pf_from_file:
            return Remote.get_instance().load("pymoo", "pf", "bbob.pf", to="json")[str(self.function)][str(self.instance)]
        else:
            ps = self.pareto_set()
            if ps is not None:
                return self.evaluate(ps)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.object(x) for x in X])
        return out

    def __getstate__(self):
        d = self.__dict__.copy()
        d["object"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.object = get_bbob(self.name, self.n_var)
        
        
class M4Problem(Problem):
    def __init__(self, function,name,dimension):
        super().__init__(n_var=dimension, n_obj=1, n_ieq_constr=0, xl=-5.0, xu=5.0)
        self.function = function
        self.name=name

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self.function(np.array(x))
        return out

    
def get_cec_problems(dimension):
    cec_problems=[]
    for year in [2005,2008,2010, 2013, 2014, 2015, 2017, 2019, 2020, 2021, 2022]:
        year_functions=opfunu.get_functions_based_classname(str(year))

        for f in year_functions:
            cec_problems+=[CECProblem(f,dimension=dimension)]
    return cec_problems
    

def get_affine_problems(dimension, alphas=None):
    problems=[]
    for problem_id_1, problem_id_2 in tqdm(list(itertools.product(list(range(1, 25)), list(range(1, 25))))):
        if problem_id_1!=problem_id_2:
            for instance_id in range(1,affine_max_instance_id+1):
                for alpha in affine_alphas if alphas is None else alphas:
                    problems+= [AffineProblem(problem_id_1, problem_id_2, alpha, dimension, instance_id, instance_id)]

    return problems

def get_random_problems(dimension):
    functions=pd.read_csv(f'{data_dir}/random_functions/random_{dimension}d.csv', index_col=0)
    return [RandomProblem(f,dimension) for f in functions['function'].values]


def get_m4_problems(dimension):
    functions=pd.read_csv(f'{data_dir}/random_functions/m4_{dimension}d.csv', index_col=0)
    return [M4Problem(function=get_function_from_parameters(row), name=row['function_id'],dimension=dimension) for row in functions.to_dict(orient='records')]

def get_bbob_problems(dimension):
    return [BBOBProblem(f'bbob-{problem_id}-{instance_id}', n_var=dimension)  for problem_id in range(1, 25) for instance_id in range(1,bbob_max_instance_id+1)]

def get_problems(benchmark, dimension):
    if benchmark=='bbob':
        return get_bbob_problems(dimension)
    if benchmark=='random':
        return get_random_problems(dimension)
    if benchmark=='affine':
        return get_affine_problems(dimension)
    if benchmark=='affine_new':
        return get_affine_problems(dimension, [0.05,0.95])
    if benchmark=='m4':
        return get_m4_problems(dimension)
    if benchmark=='cec':
        return get_cec_problems(dimension)

    
def get_all_problems(dimension):
    all_problems=[]
    for benchmark in all_benchmarks:
        all_problems+=get_problems(benchmark, dimension)
    all_problems={f.name if 'bbob' not in f.name else f.name.replace('bbob-','').replace('-','_') : f for f in all_problems}
    return all_problems