
from re import I
import random
import math
from functools import reduce
import numpy as np
from sklearn.cluster import KMeans
import json
import collections
import random
import warnings
import numpy as np
from math import gcd
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
from random import choice, randint


warnings.filterwarnings('ignore')

class PSTNode:
    def __init__(self):
        self.children = collections.defaultdict(PSTNode)
        self.count = 0

class ProbabilisticSuffixTree:
    def __init__(self, max_depth):
        self.root = PSTNode()
        self.max_depth = max_depth

    def add_sequence(self, sequence):
        """Add sequence to the tree."""
        for i in range(len(sequence)):
            for j in range(i + 1, min(i + 1 + self.max_depth, len(sequence) + 1)):
                suffix = tuple(sequence[i:j])
                self._add_suffix(suffix)

    def _add_suffix(self, suffix):
        """Add a suffix to the tree."""
        node = self.root
        for symbol in suffix:
            node = node.children[symbol]
        node.count += 1

    def predict_next(self, last_sequence):
        """Predict the next symbol based on the last sequence context."""
        for i in range(len(last_sequence), 0, -1):
            context = last_sequence[-i:]
            node = self.root
            for symbol in context:
                if symbol in node.children:
                    node = node.children[symbol]
                else:
                    break
            else:
                # Collect predictions from children nodes
                predictions = [(child, node.children[child].count) for child in node.children]

                if not predictions:
                    continue  # Try with a shorter context

                # Normalize to probabilities
                total = sum(count for _, count in predictions)
                probabilities = [(child, count / total) for child, count in predictions]

                # Sample the next symbol based on the probability distribution
                symbols, probs = zip(*probabilities)
                predicted_symbol = random.choices(symbols, probs)[0]  # Randomly sample based on probabilities
                return predicted_symbol

        return None  # No prediction available



# Function to calculate LCM of two numbers
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

# Function to calculate hyperperiod (LCM of all periods in an array)
def calculate_hyperperiod(periods):
    return reduce(lcm, periods)

# Function to create random execution times based on a given period
def cric(a, b, T, total_time):
    C = []
    for i in range(math.ceil(total_time / T) + 10):
        if random.random() < 0.40:
            C.append(a)
        else:
            C.append(b)
    return C

# Function to round a number to the nearest multiple of 50
def round_to_nearest_50(number):
    return round(number / 50) * 50



# UUniFast algorithm for generating task sets with utilization and periods
def uunifast_discrete(C_array,T_array , factor ):


    hyperperiod = calculate_hyperperiod(T_array)
    #print(f"Hyperperiod: {hyperperiod}")
    total_time = hyperperiod * factor  # Example of total time (can be adjusted)
    C_upd=[]
    for i in range (len(C_array)):
      if i<4:
        k=0.3
        c1 = cric(C_array[i], C_array[i]-round(C_array[i]*k),T_array[i] , total_time)
        #c2 = cric(4, 6, 80, total_time)
        C_upd.append(c1)
        k=0.2
      else:
        c1 = cric(C_array[i], C_array[i],T_array[i], T_array[i]*5)
        C_upd.append(c1)


    #print("C_upd",C_upd)
    #print("T_array",T_array)
    return C_upd, T_array



# Helper functions
def lcm(a, b):
    return a * b // gcd(a, b)

def lcm_multiple(numbers):
    return reduce(lcm, numbers)

def uunifast(num_tasks, utilization):
    utilizations = []
    sum_u = utilization
    for i in range(1, num_tasks):
        next_u = sum_u * np.random.uniform(0, 1) ** (1 / (num_tasks - i))
        utilizations.append(sum_u - next_u)
        sum_u = next_u
    utilizations.append(sum_u)
    return utilizations

def precompute_periods(hyperperiod, min_period, max_period):
    """Precompute periods within a specific range that divide the hyperperiod."""
    return [p for p in range(min_period, max_period + 1, 100) if hyperperiod % p == 0]

def generate_taskset(args):
    utilization, hyperperiod, valid_periods = args
    num_tasks = randint(7, 20)  # Random number of tasks between 7 and 20
    #num_tasks = 5
    utilizations = uunifast(num_tasks, utilization)
    while True:
        periods = [choice(valid_periods) for _ in range(num_tasks)]
        if lcm_multiple(periods) == hyperperiod:
            execution_times = [round(util * period) for util, period in zip(utilizations, periods)]
            return {"utilization": utilization, "num_tasks": num_tasks, "tasks": list(zip(execution_times, periods))}

def save_tasksets_to_file_array(tasksets, filename):
    with open(filename, 'w') as file:
        all_C = []  # Store all execution times
        all_T = []  # Store all periods

        for taskset in tasksets:
            C = [task[0] for task in taskset['tasks']]
            T = [task[1] for task in taskset['tasks']]

            # Only add taskset if none of the execution times (C) is zero
            if all(c > 0 for c in C):
                all_C.append(C)
                all_T.append(T)

        # Debugging information
        #print("all_C:", all_C[:100:])  # Print first 100 task sets for verification
        #print("all_T:", all_T[:100:])
        print("Number of valid task sets:", len(all_C))

        # Save C and T arrays to file
        file.write("C:\n")
        file.write(f"{all_C}\n\n")
        file.write("T:\n")
        file.write(f"{all_T}\n\n")
    return   all_C[:1000:],all_T[:1000:]

# Main function to generate task sets and save to a file
def generate_tasksets_and_save(utilizations, hyperperiod, min_period, max_period, total_tasksets, filename):
    # Precompute valid periods within the specified range
    valid_periods = precompute_periods(hyperperiod, min_period, max_period)
    #print("Generating task sets for utilization", utilizations)
    tasks_to_generate = [(utilizations, hyperperiod, valid_periods) for _ in range(total_tasksets)]

    with ProcessPoolExecutor() as executor:
        all_tasksets = list(executor.map(generate_taskset, tasks_to_generate))

    # Save task sets to file in array format
    C_100, T_100=save_tasksets_to_file_array(all_tasksets, filename)
    print(f"Generated {len(all_tasksets)} tasksets.")

    return C_100, T_100


def main():
# Generate task sets and sort them by the lowest period first
  tp_cric=[]
  tp_typ=[]
  typ_ct=[]
  cric_ct=[]
  len_pred=[]
  #utilizations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  utilizations = [0.3]
  for util in utilizations:
    #print("utilization", util)
    hyperperiod = 4500
    min_period = 100  # Minimum period range
    max_period = 900  # Maximum period range
    total_tasksets = 12000
    filename = "synthetic_tasksets_array.txt"
    C_100, T_100=generate_tasksets_and_save(util, hyperperiod, min_period, max_period, total_tasksets, filename)
    C_history=[]
    T_history=[]
    for i in range(500):
        factor=100
        C_array, T_array = uunifast_discrete(C_100[i],T_100[i], factor )
        hyperperiod = calculate_hyperperiod(T_array)
        total_time = hyperperiod * factor

        while len(T_array)==0:
            C_array, T_array, total_time = uunifast_discrete(n_tasks, U_total, T_min, T_max)
        #print("C (array):", C_array)
        #print("T (array):", T_array)
        #print("total time:", total_time)

        C_history.append(C_array)
        T_history.append(T_array)

    # Save C_history and T_history to separate files for this utilization
    c_history_filename = f"C{util:.1f}_40.txt"
    t_history_filename = f"T{util:.1f}_40.txt"

    # Write the whole list as a single entry using JSON format
    with open(c_history_filename, "w") as c_file:
        json.dump(C_history, c_file, indent=4)

    with open(t_history_filename, "w") as t_file:
        json.dump(T_history, t_file, indent=4)

    print(f"Saved entire C_history to {c_history_filename} and T_history to {t_history_filename}")


for _ in range(1):
    main()


