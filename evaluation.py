

from re import I
import random
import math
from functools import reduce
import numpy as np
from sklearn.cluster import KMeans
import ast
import collections
import random
import warnings

import numpy as np
from scipy.stats import bernoulli

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

    type_cointoss=0
    cric_cointoss=0
    # Iterate through the predictions and actual values
    head_count=0
    tail_count=0

    typ_gt=0
    cric_gt=0

    #print(len(predicted_pst),len(actual))

    ct_pf=0
    pst_fp=0
    for i in range(len(predicted_pst)):
        #print(i)
        if predicted_pst[i] < threshold and actual[i] < threshold:
            typ_tp += 1
        if predicted_pst[i] >= threshold and actual[i] >= threshold:
            cric_tp += 1
        if predicted_pst[i] > threshold and actual[i] < threshold:
            pst_fp+= 1
        #if predicted_pst[i] < threshold and actual[i] > threshold:
        #    typ_fp += 1

        ##implementing coin toss
        #random_integer =random.randint(1, 2)


        # Bernoulli Distribution (Single Toss)
        p = 0.5  # Probability of heads
        single_toss = bernoulli.rvs(p)  # Simulating one coin flip

        # Mapping 0 to 2 (Tails) and 1 to 1 (Heads)
        random_integer = 1 if single_toss == 1 else 2

        if random_integer == 1:
                head_count += 1
        if random_integer == 2:
                tail_count += 1
        if actual[i] <= threshold:
                typ_gt += 1
        if actual[i] > threshold:
                cric_gt += 1

        if  single_toss == 1 and actual[i] < threshold:
                type_cointoss += 1
        if  single_toss == 0 and actual[i] >= threshold:
                cric_cointoss += 1
        if  single_toss == 0 and actual[i] < threshold:
                ct_pf+= 1



    #print("head tail typ cric" , head_count,tail_count, typ_gt, cric_gt)
    return typ_tp, cric_tp, len(predicted_pst),type_cointoss,cric_cointoss, head_count,tail_count, typ_gt, cric_gt, pst_fp, ct_pf



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

    return threshold


def main():
  # Example usage:
  n_tasks =random.randint(4, 10)  # Number of tasks
  n_tasks=10
  U_total = 0.20  # Total utilization for the task set
  T_min = 50  # Minimum period for a task
  T_max = 200  # Maximum period for a task

# Generate task sets and sort them by the lowest period first
  tp_cric=[]
  tp_typ=[]

  typ_ct=[]
  cric_ct=[]


  len_pred=[]

  head_count_sum=[]
  tail_count_sum=[]
  typ_gt_sum=[]
  cric_gt_sum=[]

  coin_fp=[]
  pstt_fp=[]



  file_path1 = 'C0.9m.txt'
  file_path2 = 'T0.9m.txt'

  # Read the file and convert the string to a Python list
  with open(file_path1, 'r') as file:
      content = file.read()
      C_history = ast.literal_eval(content)

  # Read the file and convert the string to a Python list
  with open(file_path2, 'r') as file:
      content = file.read()
      T_history = ast.literal_eval(content)

  for i in range(100):
      factor=50
      #print(i)
      C_array, T_array = C_history[i], T_history[i]
      #print("c_array",C_array)
      #print("t_array",T_array)
      hyperperiod = 1000
      total_time = hyperperiod * factor

      # Zip C and T together, sort by T (second element), then unzip
      sorted_tasks = sorted(zip(C_array, T_array), key=lambda x: x[1])
      C_sorted, T_sorted = zip(*sorted_tasks)

      # Convert tuples back to lists
      C_sorted = list(C_sorted)
      T_sorted = list(T_sorted)


      response_times = calculate_multiframe_response_times_upto_time(C_sorted, T_sorted, total_time)

      # Extract and print the response times for the last task only
      last_task_id = len(C_sorted) - 1-1
      #last_task_response_times = [R for time, R in response_times[last_task_id]]
      last_task_response_times = [round(R, 2) for time, R in response_times[last_task_id]]

      #print("last task response time befor pst",last_task_response_times)


      k=int(len(last_task_response_times))
      #print("k",k)
      #print(f"Hyperperiod: {hyperperiod}")
      total_time = hyperperiod * (factor+20)  # Example of total time (can be adjusted)
      response_times_hyp2 = calculate_multiframe_response_times_upto_time(C_sorted, T_sorted, total_time)


      last_task_response_times_hyp2 = [R for time, R in response_times_hyp2[last_task_id]]
      #print("last task response time hyp2", len(last_task_response_times_hyp2))
      ##actual response time
      #last_task_response_times_hyp2=last_task_response_times_hyp2[-int(k/factor):]
      last_task_response_times_hyp2=last_task_response_times_hyp2[k:]
      #print("actual hyp 2", last_task_response_times_hyp2)

      threshold = cluster_data(last_task_response_times)
      #print("threshold",threshold)

      #pst = ProbabilisticSuffixTree(max_depth=5)
      pst = ProbabilisticSuffixTree(max_depth=20)

      # Train the tree with the sequence
      pst.add_sequence(last_task_response_times)

      # Predict the next 10 values
      num_predictions = len(last_task_response_times)
      predicted_values = []


      #print(last_task_response_times[-10:] )
      #print("las task hyp2 len", len(last_task_response_times_hyp2))
      window=20
      for _ in range(len(last_task_response_times_hyp2)-window):
          #last_sequence = last_task_response_times_hyp2[-20:]  # Take the last 20 elements as context
          #predicted = pst.predict_next(last_sequence)
          last_sequence = last_task_response_times_hyp2[_: _+window]
          predicted = pst.predict_next(last_sequence)

          if predicted is not None:
              #print(f"Predicted next value: {predicted}")
              predicted_values.append(predicted)
              last_task_response_times.append(predicted)  # Update the sequence with the predicted value
          else:
              #print("No prediction available")
              break  # Stop predicting if no prediction is available


      #print("len", len(predicted_values), len(last_task_response_times_hyp2))
      #print("predicted",predicted_values)
      #print("actual...",last_task_response_times_hyp2)
      typ_tp, cric_tp, len_pred_1, tp_typ_ct,tp_cric_ct, head_count_s,tail_count_s, typ_gt_s, cric_gt_s, pst_fp, ct_pf=calculate_classification_metrics(predicted_values, last_task_response_times_hyp2[20:], threshold)
      tp_typ.append(typ_tp)
      tp_cric.append(cric_tp)
      len_pred.append(len_pred_1)
      typ_ct.append(tp_typ_ct)
      cric_ct.append(tp_cric_ct)
      head_count_sum.append(head_count_s)
      tail_count_sum.append(tail_count_s)
      typ_gt_sum.append(typ_gt_s)
      cric_gt_sum.append(cric_gt_s)
      coin_fp.append(ct_pf)
      pstt_fp.append(pst_fp)
      #print(i)


  #print(np.mean(tp_typ),sum(tp_typ))
  #print(np.mean(tp_cric),sum(tp_cric))
  #print(np.mean(len_pred),sum(len_pred))
  #print()

  #print("c history", C_history)
  #print("t history", T_history)
  #print("c history", C_history[0])
  return sum(tp_typ),sum(tp_cric),sum(len_pred), sum(typ_ct), sum(cric_ct), sum(head_count_sum), sum(tail_count_sum), sum(typ_gt_sum), sum(cric_gt_sum), sum (coin_fp), sum (pstt_fp)

typsum,cricsum,predsum, typct, cricct,headcountst,tailcountst, typgtst, cricgtst, coin_sum, pst_sum =0,0,0,0,0,0,0,0,0,0,0
for _ in range(1):
    typ_sum,cric_sum,pred_sum, typct_sum,cricct_sum, head_count_st,tail_count_st, typ_gt_st, cric_gt_st, csum, psum=main()
    typsum+=typ_sum
    cricsum+=cric_sum
    predsum+=pred_sum
    typct+= typct_sum
    cricct+=cricct_sum

    headcountst+=head_count_st
    tailcountst+=tail_count_st
    typgtst+=typ_gt_st
    cricgtst+=cric_gt_st

    coin_sum+=csum
    pst_sum+=psum



print("typ_sum",typsum)
print("cric_sum",cricsum)
#print("pst FP",pst_sum)
print("ct FP",coin_sum)
print("pred_sum",predsum)
#print("typct_sum",typct)
#print("cricct_sum",cricct)
print("ip for typ",round(typsum/(predsum*0.9),2))
print("ip for cric",round(cricsum/(predsum*0.1),2))
#print("typct",round(typct/(predsum*0.9),2))
#print("cricct",round(cricct/(predsum*0.1),2))


print("IP PST:", round((typsum+cricsum) / (typgtst+cricgtst), 2))
print(pst_sum/predsum)
#print("typct:", round((typct+cricct) / (headcountst+tailcountst), 2))
#print("IP for cric:", round(cricsum / cricgtst, 2))
#print("cricct:", round(cricct / tailcountst, 2))
