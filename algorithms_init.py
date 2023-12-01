from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch


from pymoo.algorithms.soo.nonconvex.ga import *
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.spx import SPX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling

import numpy as np
import random

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    

def get_algorithms(population_size, seed, sampling):
    cmaes = CMAES(popsize=population_size, seed=seed)
    pso = PSO(popsize=population_size, seed=seed, sampling=sampling)
    pso1 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='zero', adaptive=True)
    pso2 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='zero', adaptive=False)
    pso3 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='random', adaptive=True)
    pso4 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='random', adaptive=False)
    pso5 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='random', adaptive=False, w=0.7)
    pso6 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='zero', adaptive=False, w=0.7)
    pso7 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='zero', adaptive=False, w=0.7,c1=1,c2=2)
    pso8 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='zero', adaptive=False, w=0.2,c1=1,c2=1)
    pso9 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='random', adaptive=False, w=0.7,c1=1,c2=2)
    pso10 = PSO(popsize=population_size, seed=seed, sampling=sampling, initial_velocity='random', adaptive=False, w=0.2,c1=1,c2=1)


    de = DE(popsize=population_size, seed=seed, sampling=sampling)
    es = ES(popsize=population_size, seed=seed, sampling=sampling)
    ga = GA(popsize=population_size,sampling=sampling, seed=seed)
    
    de1 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.5, variant="DE/rand/1/bin")
    de2 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.6, variant="DE/best/1/exp")
    de3 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.2, variant="DE/rand/1/exp")
    de4 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.2, variant="DE/best/1/bin")
    de5 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.4, F=0.4, variant="DE/best/1/bin")
    de6 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.4, F=0.7, variant="DE/best/1/exp")
    de7 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.4, F=0.4, variant="DE/rand/1/exp")
    de8 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.4, F=0.7, variant="DE/rand/1/bin")
    de9 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.6, F=0.6, variant="DE/best/1/exp")
    de10 = DE(popsize=population_size, seed=seed, sampling=sampling, CR=0.2, F=0.6, variant="DE/rand/1/exp")
    
    ga1 = GA(popsize=population_size,sampling=sampling, seed=seed, crossover=SBX(), mutation=PM(), survival=FitnessSurvival())
    ga2 = GA(popsize=population_size,sampling=sampling, seed=seed, crossover=SPX(), mutation=PM(), survival=FitnessSurvival())
    ga3 = GA(popsize=population_size,sampling=sampling, seed=seed, crossover=SBX(), mutation=BitflipMutation(), survival=FitnessSurvival())
    ga4 = GA(popsize=population_size,sampling=sampling, seed=seed, crossover=SPX(), mutation=BitflipMutation(), survival=FitnessSurvival())
    
    
    # es2=ES(popsize=population_size,seed=seed, sampling=sampling,n_offsprings=200,rule=1.0 / 7.0,phi=1.0,gamma=0.85)

    algorithms = {'DE1': de1, 'DE2': de2, 'DE3': de3, 'DE4': de4, 'DE5': de5, 'DE6': de6, 'DE7': de7, 'DE8': de8,
                  'DE9': de9, 'DE10': de10, 'PSO':pso, 'DE':de, 'ES':es, 'CMAES':cmaes, 'GA':ga,
                 'PSO1':pso1, 'PSO2':pso2, 'PSO3':pso3, 'PSO4': pso4, 'PSO5':pso5, 'PSO6':pso6, 'PSO7': pso7, 'PSO8': pso8, 'PSO9':pso9,'PSO10':pso10,
                 'GA1':ga1, 'GA2':ga2, 'GA3': ga3, 'GA4': ga4
                 }
    return algorithms