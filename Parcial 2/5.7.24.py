import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List
import copy

epochs = 500
cuotas =  np.array([8.51, 10.68, 12.24, 13.66, 15.37, 17.15, 19.66, 24.69])

class Trader:
    Unique_ids = 0
    
    def __init__(self, capital ,cuotas, Id):
        self.Id = Id
        self.cuotas = cuotas
        self.capital = capital
        
        self.weights = np.random.uniform(0.,1., size=self.cuotas.shape[0])
        self.weights = self.weights / np.sum(self.weights)
        self.fitness = np.min(self.capital * (self.weights * self.cuotas - 1))
    
    def set_fitness(self):
        return np.min(self.capital * (self.weights * self.cuotas - 1))
    
    def mutation(self):
        self.weights += np.random.normal(loc=0., scale=0.02, size=self.weights.shape[0])
        self.weights = np.abs(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        self.fitness = self.set_fitness()
    
    def __repr__(self):
        return "Trader #" + str(self.Id)
    
        
def generate_initial_population(size, capital, cuotas):
    return [Trader(capital, cuotas) for _ in range(size)]

def Genetic(capital,shares,Traders, epochs=5000):
    N = int(0.9*len(Traders))
    for _ in range(epochs):
        for _, trader in enumerate(Traders):
            trader.mutation()
            
        scores = [ (trader.fitness, trader) for trader in Traders ]
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        
        Temp = [r[1] for i, r in enumerate(scores) if i < N]

        
        for i in range(int(0.1 * len(Traders))):
            Traders[i] = Trader(capital,shares,i)
    
    print(scores)

pop1 = generate_population(20, cuotas, 1000000)
a = [i for i in pop1]
Genetic(1000000,cuotas,a)