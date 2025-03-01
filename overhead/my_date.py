import random
import math
from functools import reduce
import numpy as np
from sklearn.cluster import KMeans
import json

import collections
import random

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
        if random.random() < 0.30:
            C.append(b)
        else:
            C.append(a)
    return C

# Function to round a number to the nearest multiple of 50
def round_to_nearest_50(number):
    return round(number / 50) * 50

def calculate_multiframe_response_times_upto_time(C, T, total_time):
    n = len(C)
    response_times = {i: [] for i in range(n)}

    for current_time in range(0, total_time):
        for i in range(n):
            if current_time % T[i] == 0:
                # Task i arrives at this time
                frame_index = (current_time // T[i]) % len(C[i])
                current_C = C[i][frame_index]
                R_prev = current_C
                while True:
                    R = current_C + sum(
                        math.ceil(R_prev / T[j]) * C[j][(current_time // T[j]) % len(C[j])]
                        for j in range(i)
                    )
                    if R == R_prev:
                        break
                    R_prev = R
                response_times[i].append((current_time, R))

    return response_times

def calculate_classification_metrics(predicted_pst, actual, threshold):
    # Initialize the counters
    typ_tp = 0
    cric_tp = 0
    cric_fp = 0
    typ_fp = 0

    # Iterate through the predictions and actual values
    for i in range(len(predicted_pst)):
        if predicted_pst[i] < threshold and actual[i] < threshold:
            typ_tp += 1
        if predicted_pst[i] > threshold and actual[i] > threshold:
            cric_tp += 1
        if predicted_pst[i] > threshold and actual[i] < threshold:
            cric_fp += 1
        if predicted_pst[i] < threshold and actual[i] > threshold:
            typ_fp += 1

    # Print the results
    #print("Typical true positive:", typ_tp)
    #print("Critical true positive:", cric_tp)
    #print("Critical false positive:", cric_fp)
    #print("Typical false positive:", typ_fp)
    #print("len of predicted data", len(predicted_pst))

    return typ_tp, cric_tp, len(predicted_pst)



def cluster_data(data, n_clusters=2, random_state=0):

    # Reshape data for KMeans
    data = np.array(data).reshape(-1, 1)

    # Perform KMeans clustering
    # Perform KMeans clustering with 2 clusters and explicit n_init value
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(data)


    # Get the cluster centers and calculate the threshold
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(cluster_centers)

    # Cluster data into two sets based on the threshold
    cluster_1 = [x[0] for x in data if x <= threshold]
    cluster_2 = [x[0] for x in data if x > threshold]

    return round(threshold)


# UUniFast algorithm for generating task sets with utilization and periods
def uunifast_discrete(n, U_total, T_min, T_max, factor):
    utilizations = []
    sum_U = U_total

    for i in range(1, n):
        next_sum_U = sum_U * (random.random() ** (1.0 / (n - i)))
        utilizations.append(sum_U - next_sum_U)
        sum_U = next_sum_U

    utilizations.append(sum_U)

    C_array = []
    T_array = []
    for U in utilizations:
        T = random.randint(T_min // 5, T_max // 5) * 5  # Ensure T is a multiple of 5
        T = round_to_nearest_50(T)  # Round the period to the nearest 50
        C = round(U * T)  # Calculate the computation time as a whole number
        C_array.append(C)
        T_array.append(T)

    hyperperiod = calculate_hyperperiod(T_array)
    #print(f"Hyperperiod: {hyperperiod}")
    total_time = hyperperiod * factor  # Example of total time (can be adjusted)
    C_upd=[]
    for i in range (len(C_array)):
      if i<3:
        k=0.4
        c1 = cric(C_array[i], C_array[i]+round(C_array[i]*k),T_array[i] , total_time)
        #c2 = cric(4, 6, 80, total_time)
        C_upd.append(c1)
        k=0.2
      else:
        c1 = cric(C_array[i], C_array[i],T_array[i], T_array[i]*5)
        C_upd.append(c1)


    #print(C_upd)
    return C_upd, T_array

# Example usage:
n_tasks =3  # Number of tasks
U_total = 0.60  # Total utilization for the task set
T_min = 50  # Minimum period for a task
T_max = 1000  # Maximum period for a task

# Generate task sets and sort them by the lowest period first
tp_cric=[]
tp_typ=[]
len_pred=[]
for i in range(1):
    factor=110
    C_array, T_array = uunifast_discrete(n_tasks, U_total, T_min, T_max,factor)
    hyperperiod = calculate_hyperperiod(T_array)
    total_time = hyperperiod * factor

    while len(T_array)==0:
        C_array, T_array, total_time = uunifast_discrete(n_tasks, U_total, T_min, T_max)
    #print("C (array):", C_array)
    #print("T (array):", T_array)
    # Zip C and T together, sort by T (second element), then unzip
    sorted_tasks = sorted(zip(C_array, T_array), key=lambda x: x[1])
    C_sorted, T_sorted = zip(*sorted_tasks)
    
    #C_sorted=[[9.01, 13.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 13.01, 9.01, 13.01, 13.01, 9.01, 9.01, 9.01, 13.01, 13.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 13.01, 9.01, 9.01, 9.01, 13.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 13.01, 9.01, 9.01, 9.01, 13.01, 13.01, 13.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 13.01, 13.01, 9.01, 9.01, 9.01, 13.01, 9.01, 13.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 13.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 13.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01, 9.01], [5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 7, 5, 7, 5, 5, 5], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    #T_sorted= [30, 50, 100]
    # Convert tuples back to lists
    C_sorted = list(C_sorted)
    T_sorted = list(T_sorted)

    #print(f"Task set {i+1}:")
    #print("C (sorted):", C_sorted)
    #print("T (sorted):", T_sorted)

    response_times = calculate_multiframe_response_times_upto_time(C_sorted, T_sorted, total_time)
    #response_times=[16.009999999999998, 20.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 18.009999999999998, 18.009999999999998, 16.009999999999998, 16.009999999999998, 18.009999999999998, 16.009999999999998, 20.009999999999998, 16.009999999999998, 18.009999999999998, 20.009999999999998, 16.009999999999998, 18.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 22.009999999999998, 22.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 18.009999999999998, 18.009999999999998]
#[18.009999999999998, 18.009999999999998, 16.009999999999998, 16.009999999999998, 18.009999999999998, 16.009999999999998, 20.009999999999998, 16.009999999999998, 18.009999999999998, 20.009999999999998, 16.009999999999998, 18.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 22.009999999999998, 22.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 16.009999999999998, 18.009999999999998, 18.009999999999998]
    # Extract and print the response times for the last task only
    last_task_id = len(C_sorted) - 1
    last_task_response_times = [R for time, R in response_times[last_task_id]]
    #print("rt", last_task_response_times)
    # Print the array as a JSON string
    print(json.dumps(last_task_response_times))

    #Print the response times of the last task
    #print(f"Response times of the last task (Task {last_task_id + 1}):")
    #print(last_task_response_times)
    #print(last_task_response_times[-25:])

    ##2nd task response time
    #hyperperiod = calculate_hyperperiod(T_array)
    ##calculating the len of response time till hyp 1

    k=int(len(last_task_response_times))

    #print(f"Hyperperiod: {hyperperiod}")
    total_time = hyperperiod * (factor+1)  # Example of total time (can be adjusted)
    response_times_hyp2 = calculate_multiframe_response_times_upto_time(C_sorted, T_sorted, total_time)
    last_task_response_times_hyp2 = [R for time, R in response_times_hyp2[last_task_id]]

    ##actual response time
    last_task_response_times_hyp2=last_task_response_times_hyp2[-int(k/factor):]
    #print("actual hyp 2", last_task_response_times_hyp2)


    #last_task_response_times=[43, 43, 43, 43, 51, 51, 43, 51, 43, 43, 43, 45, 43, 43, 43, 43, 51, 43, 51, 43, 51, 43, 53, 51, 45, 43, 53, 45, 51, 45, 45, 45, 43, 45, 43, 43, 43, 43, 43, 43, 51, 43, 43, 43, 43, 45, 43, 51, 43, 43, 45, 45, 51, 53, 43, 43, 43, 43, 45, 43, 43, 43, 43, 45, 51, 45, 43, 51, 43, 43, 45, 45, 51, 45, 45, 53, 51, 51, 45, 43, 43, 53, 43, 43, 45, 51, 43, 43, 45, 53, 43, 51, 43, 43, 45, 43, 43, 43, 43, 53, 43, 43, 51, 43, 43, 43, 45, 43, 45, 43, 51, 43, 51, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 51, 43, 43, 43, 45, 51, 43, 43, 51, 43, 45, 43, 51, 43, 43, 45, 45, 43, 43, 43, 43, 43, 45, 43, 45, 43, 45, 51, 45, 51, 43, 43, 45, 51, 43, 43, 43, 51, 45, 43, 43, 43, 43, 45, 43, 43, 43, 43, 43, 43, 43, 43, 43, 45, 51, 43, 43, 43, 51, 51, 45, 51, 51, 43, 51, 43, 43, 43, 43, 51, 43, 45, 43, 43, 43, 53, 43, 51, 51, 51, 43, 43, 43, 51, 45, 43, 51, 45, 43, 51, 43, 43, 43, 51, 43, 51, 43, 51, 43, 43, 43, 43, 43, 43, 43, 43, 51, 43, 51, 43, 51, 43, 45, 53, 43, 51, 43, 51, 43, 43, 51, 43, 45]

    threshold = cluster_data(last_task_response_times)
    pst = ProbabilisticSuffixTree(max_depth=5)

    # Train the tree with the sequence
    pst.add_sequence(last_task_response_times)

    # Predict the next 10 values
    num_predictions = len(last_task_response_times)
    predicted_values = []


    #print(last_task_response_times[-10:] )
    for _ in range(len(last_task_response_times_hyp2)):
        last_sequence = last_task_response_times[-10:]  # Take the last 5 elements as context
        predicted = pst.predict_next(last_sequence)

        if predicted is not None:
            #print(f"Predicted next value: {predicted}")
            predicted_values.append(predicted)
            last_task_response_times.append(predicted)  # Update the sequence with the predicted value
        else:
            #print("No prediction available")
            break  # Stop predicting if no prediction is available

    #print(f"Next  predicted values: {predicted_values}")
    #print("threshold", threshold)
    #last_task_response_times_hyp2=[43, 43, 43, 43, 51, 51, 43, 51, 43, 43, 43, 45, 43, 43, 43, 43, 51, 43, 51, 43, 51, 43, 53, 51, 45, 43, 53, 45, 51, 45, 45, 45, 43, 45, 43, 43, 43, 43, 43, 43, 51, 43, 43, 43, 43, 45, 43, 51, 43, 43, 45, 45, 51, 53, 43, 43, 43, 43, 45, 43, 43, 43, 43, 45, 51, 45, 43, 51, 43, 43, 45, 45, 51, 45, 45, 53, 51, 51, 45, 43, 43, 53, 43, 43, 45, 51, 43, 43, 45, 53, 43, 51, 43, 43, 45, 43, 43, 43, 43, 53, 43, 43, 51, 43, 43, 43, 45, 43, 45, 43, 51, 43, 51, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 51, 43, 43, 43, 45, 51, 43, 43, 51, 43, 45, 43, 51, 43, 43, 45, 45, 43, 43, 43, 43, 43, 45, 43, 45, 43, 45, 51, 45, 51, 43, 43, 45, 51, 43, 43, 43, 51, 45, 43, 43, 43, 43, 45, 43, 43, 43, 43, 43, 43, 43, 43, 43, 45, 51, 43, 43, 43, 51, 51, 45, 51, 51, 43, 51, 43, 43, 43, 43, 51, 43, 45, 43, 43, 43, 53, 43, 51, 51, 51, 43, 43, 43, 51, 45, 43, 51, 45, 43, 51, 43, 43, 43, 51, 43, 51, 43, 51, 43, 43, 43, 43, 43, 43, 43, 43, 51, 43, 51, 43, 51, 43, 45, 53, 43, 51, 43, 51, 43, 43, 51, 43, 45]
    #predicted_values=[43, 43, 43, 43, 51, 43, 43, 51, 43, 53, 43, 53, 43, 51, 43, 45, 43, 43, 43, 51, 43, 51, 51, 43, 43, 43, 45, 51, 43, 45, 43, 45, 45, 45, 43, 43, 43, 43, 45, 43, 43, 43, 43, 51, 51, 45, 45, 43, 43, 43, 43, 43, 45, 43, 43, 43, 43, 43, 43, 51, 45, 43, 43, 43, 43, 51, 43, 43, 43, 43, 43, 45, 51, 43, 45, 43, 51, 53, 45, 43, 43, 51, 43, 51, 43, 43, 51, 43, 51, 43, 45, 51, 43, 43, 43, 43, 43, 45, 51, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 45, 43, 43, 43, 43, 43, 43, 45, 43, 45, 43, 43, 43, 43, 43, 43, 51, 51, 45, 43, 45, 43, 45, 43, 43, 43, 43, 43, 43, 43, 43, 43, 51, 45, 43, 43, 43, 43, 51, 43, 43, 43, 43, 45, 43, 51, 53, 45, 43, 43, 51, 43, 51, 43, 45, 43, 43, 43, 53, 45, 51, 43, 43, 43, 43, 43, 51, 43, 43, 51, 43, 45, 43, 45, 45, 45, 43, 43, 43, 45, 51, 53, 45, 43, 45, 53, 43, 51, 45, 43, 43, 43, 43, 43, 43, 43, 43, 51, 51, 45, 45, 43, 43, 43, 53, 45, 51, 43, 43, 43, 43, 45, 51, 43, 43, 43, 43, 51, 45, 43, 43, 43, 43, 45, 53, 43, 43, 51, 43, 51, 51, 43, 43, 43, 43, 43, 45, 53, 43, 43, 51, 51, 43]
    #threshold=50

    typ_tp, cric_tp, len_pred_1=calculate_classification_metrics(predicted_values, last_task_response_times_hyp2, threshold)
    tp_typ.append(typ_tp)
    tp_cric.append(cric_tp)
    len_pred.append(len_pred_1)
    #print(i)


#print(np.mean(tp_typ))
#print(np.mean(tp_cric))
#print(np.mean(len_pred))
