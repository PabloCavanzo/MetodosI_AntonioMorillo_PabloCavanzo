import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations_with_replacement

Dict = {0: 'C', 1: 'S'}
Dict2 = {0: 'J', 1: 'B'}
possible_states = np.array([0, 1])
prior = np.array([0.2, 0.8])

def get_permutations(State, N):
    comb_list = list(combinations_with_replacement(State, N))
    permu_list = []
    for comb in comb_list:
        for i in list(permutations(comb, N)):
            if i not in permu_list:
                permu_list.append(i)
    permu_list = np.array(permu_list)
    return permu_list

possible_hidden_states = get_permutations(possible_states, 8)
T = np.array([[0.8, 0.2], [0.2, 0.8]])
E = np.array([[0.5, 0.9], [0.5, 0.1]])
Obs = [1, 0, 0, 0, 1, 0, 1, 0]

def GetProb(transmission, emission, observed, state, prior):
    n = len(observed)
    probability = prior[state[0]]
    for i in range(n - 1):
        probability *= transmission[state[i + 1], state[i]]
    for i in range(n):
        probability *= emission[observed[i], state[i]]
    return probability

def get_prob_state(hidden, transmission, emission, observed, prior):
    probabilities = np.zeros(possible_hidden_states.shape[0])
    for i in range(probabilities.shape[0]):
        probabilities[i] = GetProb(transmission, emission, observed, hidden[i], prior)
    max_i = np.where(probabilities == np.max(probabilities))
    return probabilities, max_i

possible_obs, max_prob = get_prob_state(possible_hidden_states, T, E, Obs, prior)
max_obs = [Dict2[int(i)] for i in possible_hidden_states[max_prob][0]]
obs_stat = [Dict[int(i)] for i in Obs]
max_prob0 = possible_obs[max_prob][0].copy()

plt.plot(possible_obs, color="black")
plt.axhline(max_prob0, color="red", linestyle="--")
plt.grid(True)
plt.show()

possible_observable_states = possible_hidden_states.copy()
pt = 0
for observation in possible_observable_states:
    possible, max_prob = get_prob_state(possible_hidden_states, T, E, observation, prior)
    pt += np.sum(possible)

print("Dado el estado observable", obs_stat, "este tiene una probabilidad de", str(np.sum(possible_obs) * 100) + "%.")
print("\nLa secuencia no observable más probable que puede ocurrir es:", max_obs, "\nCon una probabilidad del:", str(max_prob0 * 100) + "%.")
print("\nLa suma de las posibilidades de todos los estados observables es:", pt)
