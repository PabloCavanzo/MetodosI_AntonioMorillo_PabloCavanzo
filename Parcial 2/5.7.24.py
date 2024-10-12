import numpy as np
import matplotlib.pyplot as plt
import random as rd

class Trader:
    Unique_ids = 0
    
    def __init__(self, capital, shares, weights=None):
        self.Id = Trader.Unique_ids
        Trader.Unique_ids += 1
        self.shares = shares
        self.capital = capital
        
        if weights is None:
            self.weights = np.random.uniform(0.,1., size=self.shares.shape[0])
            self.weights /= np.sum(self.weights)
        else:
            self.weights = weights
            
        self.fitness = self.set_fitness()
    
    def set_fitness(self):
        return np.min(self.capital * (self.weights * self.shares - 1))
    
    def mutation(self, mutation_rate=0.05, mutation_scale=0.01):
        for i in range(len(self.weights)):
            if rd.random() < mutation_rate:
                self.weights[i] += np.random.normal(0, mutation_scale)
                self.weights[i] = max(self.weights[i], 0.01)
            
        self.weights /= np.sum(self.weights)
        self.fitness = self.set_fitness()
    
    def __repr__(self):
        return f"Trader #{self.Id} | Fitness: {self.fitness:.2f}"

def selection_pair(population):
    selected = []
    for _ in range(2):
        contenders = rd.sample(population, 3)
        contender = max(contenders, key=lambda trader: trader.fitness)
        selected.append(contender)
    
    return selected

def crossover(parent1, parent2):
    p = rd.randint(1, parent1.weights.size - 1)
    child1_weights = np.concatenate((parent1.weights[:p], parent2.weights[p:]))
    child2_weights = np.concatenate((parent2.weights[:p], parent1.weights[p:]))
    child1 = Trader(parent1.capital, parent1.shares, weights=child1_weights)
    child2 = Trader(parent2.capital, parent2.shares, weights=child2_weights)
    return child1, child2
     
def generate_population(size, capital, cuotas):
    return [Trader(capital, cuotas) for _ in range(size)]
    
def evolve(capital, cuotas, epochs, population_size=500):
    Traders = generate_population(population_size, capital, cuotas)
    best_fitness_over_time = []
    
    for _ in range(epochs):     
        Traders.sort(key=lambda trader: trader.fitness, reverse=True)
        next_generation = Traders[0:2]
        
        best_fitness_over_time.append(Traders[0].fitness)
        
        while len(next_generation) < population_size:
            parents = selection_pair(Traders)
            child1, child2 = crossover(parents[0], parents[1])
            child1.mutation()
            child2.mutation()
            next_generation += [child1, child2]
        
        Traders = next_generation[:population_size]
    
    plt.plot(range(1, epochs+1), best_fitness_over_time)
    plt.xlabel('Época')
    plt.ylabel('Mejor Aptitud')
    plt.title('Evolución')
    plt.show()
    
    return Traders[0].weights


cuotas =  np.array([8.51, 10.68, 12.24, 13.66, 15.37, 17.15, 19.66, 24.69])
weights = evolve(1000000, cuotas, 500, population_size=500)
prof_weights = np.array([0.185, 0.152, 0.137, 0.125, 0.116, 0.107, 0.096, 0.082])
pesos_yo = np.array([0.20221668, 0.16116429, 0.14055939, 0.12629439, 0.11193354, 0.10060435,0.08754264, 0.06968473])
#print(np.min(1000000*(pesos_yo*cuotas-1)))
#print(np.min(1000000*(prof_weights*cuotas-1)))
print("Mejores pesos encontrados:", weights)