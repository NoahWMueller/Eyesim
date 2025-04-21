#!/usr/bin/env python3
from random.eye import *
import random
import math

class Genotype():
    def __init__(self, ID, gene=None):
        self.ID = ID
        if gene is None: # Randomly initialise.
            self.gene = []
            for _ in range(SIZE):
                self.gene.append(random.randint(0,1))
        else:
            self.gene = gene
        self.fitness = 0

POP = 101
ELEM = 10
BITS = 5
SIZE = ELEM * BITS
MAX_ITER = 100
FIT_GOAL = ELEM - 0.1
MUT = int(POP / 2)
DEBUG = True

pool = []
next = []

fitlist = []
fitsum = None
maxfit = 0
best = Genotype(0)
goal = []


def init():
    global next
    for i in range(POP):
        pool.append(Genotype(i))
    for i in range(ELEM):
        goal.append(1 + math.sin(i * 2 * math.pi / ELEM) / 2 )
    next = list(pool)

def gene_slice(gene, i):
    b = i*BITS
    return (gene[b]*16 + gene[b+1]*8 
            + gene[b+2]*4 + gene[b+3]*2
            + gene[b+4] ) / 31

def fitness(gene):
    fitness = 0
    for i in range(ELEM):
        s = gene_slice(gene, i)
        fitness += 1 - abs(goal[i] - s)
    return fitness

def evaluate():
    global fitsum, maxfit, best
    fitsum = 0
    maxfit = 0
    for genotype in pool:
        genotype.fitness = fitness(genotype.gene)
        fitsum += genotype.fitness
        if genotype.fitness > maxfit:
            maxfit = genotype.fitness
            best = genotype

def select_gene():
    wheel = random.randrange(0, int(fitsum)-1)
    count = 0
    for genotype in pool:
        count += genotype.fitness
        if count >= wheel:
            break
    return genotype.gene

def mutation():
    chosen_gene = random.choice(pool)
    chosen_bit = random.randrange(0, SIZE)
    chosen_gene.gene[chosen_bit] = 1 - chosen_gene.gene[chosen_bit]

def crossover(gene_1, gene_2):
    cut = random.randrange(1, SIZE-1)
    child_1 = gene_1[:cut] + gene_2[cut:]
    child_2 = gene_2[:cut] + gene_1[cut:]
    return child_1, child_2

def print_all_genes():
    for genotype in pool:
        print("Gene " + str(genotype.ID) + " POOL: ", end='')
        for bit in genotype.gene:
            print(str(bit), end='')
        print(" NEXT: ", end='')
        for bit in next[genotype.ID].gene:
            print(str(bit), end='')
        print("")


def main():
    global pool, next
    init()

    for iteration in range(MAX_ITER):
        print("-----------------------------------------------")
        print("New Generation: " + str(iteration))
        print_all_genes()
        evaluate()

        next = []
        next.append(best)

        # Crossover.
        for i in range(1, POP, 2):
            parent_1 = select_gene()
            parent_2 = select_gene()
            child_1, child_2 = crossover(parent_1, parent_2)
            next.append(Genotype(i, gene=child_1))
            next.append(Genotype(i+1, gene=child_2))
        
        # Mutation.
        for _ in range(MUT):
            mutation()

        pool = list(next)
    print(maxfit)

if __name__ == "__main__":
    main()