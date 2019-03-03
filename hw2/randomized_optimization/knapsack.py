import time
import sys
import java.util.Random as Random

sys.path.append("./ABAGAIL.jar")

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array
from time import clock
from itertools import product

from base import *

# Adapted from https://github.com/pushkar/ABAGAIL/blob/master/jython/knapsack.py

# Random number generator */
random = Random()
# The number of items
NUM_ITEMS = 40
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME

# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

# N = 500
maxIters = 5000
numTrials = 5
# fill = [2] * N
# ranges = array('i', fill)
outfile = OUTPUT_DIRECTORY + '/KNAPSACK/KNAPSACK_{}_{}_LOG.csv'
# ef = FlipFlopEvaluationFunction()
# odd = DiscreteUniformDistribution(ranges)
# nf = DiscreteChangeOneNeighbor(ranges)
# mf = DiscreteChangeOneMutation(ranges)
# cf = SingleCrossOver()
# df = DiscreteDependencyTree(.1, ranges)
# hcp = GenericHillClimbingProblem(ef, odd, nf)
# gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
# pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

# RHC
for t in range(numTrials):
    fname = outfile.format('RHC', str(t + 1))
    with open(fname, 'w') as f:
        f.write('iterations,fitness,time\n')
    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = UniformCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 10)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        score = ef.value(rhc.getOptimal())
        st = '{},{},{}\n'.format(i, score, times[-1])
        print st
        with open(fname, 'a') as f:
            f.write(st)

# SA
for t in range(numTrials):
    for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
        fname = outfile.format('SA{}'.format(CE), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time\n')
        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = UniformCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(1E10, CE, hcp)
        fit = FixedIterationTrainer(sa, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(sa.getOptimal())
            st = '{},{},{}\n'.format(i, score, times[-1])
            print st
            with open(fname, 'a') as f:
                f.write(st)

# GA
for t in range(numTrials):
    for pop, mate, mutate in product([100], [50, 30, 10], [50, 30, 10]):
        fname = outfile.format('GA{}_{}_{}'.format(pop, mate, mutate), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time\n')
        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = UniformCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = FixedIterationTrainer(ga, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(ga.getOptimal())
            st = '{},{},{}\n'.format(i, score, times[-1])
            print st
            with open(fname, 'a') as f:
                f.write(st)

# MIMIC
for t in range(numTrials):
    for samples, keep, m in product([100], [50], [0.1, 0.3, 0.5, 0.7, 0.9]):
        fname = outfile.format('MIMIC{}_{}_{}'.format(samples, keep, m), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time\n')
        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = UniformCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = FixedIterationTrainer(mimic, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(mimic.getOptimal())
            st = '{},{},{}\n'.format(i, score, times[-1])
            print st
            with open(fname, 'a') as f:
                f.write(st)
