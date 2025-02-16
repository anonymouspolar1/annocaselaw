import numpy as np
import json
from sklearn.metrics import f1_score
from sklearn.utils import resample


def compute_ci(y_true, y_pred):
    n_bootstraps = 200  
    f1_scores_weighted = []

    # Bootstrap resampling from stored API predictions
    for _ in range(n_bootstraps):
        # Resample indices
        indices = resample(range(len(y_true)), replace=True, random_state=None)

        # Use resampled indices to select true labels and API predictions
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]

        # Compute weighted F1 score
        f1_weighted = f1_score(y_true_resampled, y_pred_resampled, average='weighted')
        f1_scores_weighted.append(f1_weighted)

    # Compute 95% Confidence Interval (percentile method)
    lower = np.percentile(f1_scores_weighted, 2.5)
    upper = np.percentile(f1_scores_weighted, 97.5)
    mean = np.mean(f1_scores_weighted)

    return mean, lower, upper


def longest_common_substring(X, Y):
    """Find the longest common substring between prediction and ground truth"""
    m, n = len(X), len(Y)

    # If necessary, swap to ensure Y is the shorter string
    if m < n:
        X, Y = Y, X
        m, n = n, m

    # Initialize variables
    max_len = 0
    end_index = 0
    dp = [0] * (n + 1)

    # Fill dp array and track the max length of common substring
    for i in range(1, m + 1):
        prev = 0  # To store the previous diagonal value
        for j in range(1, n + 1):
            temp = dp[j]
            if X[i - 1] == Y[j - 1]:
                dp[j] = prev + 1
                if dp[j] > max_len:
                    max_len = dp[j]
                    end_index = i
            else:
                dp[j] = 0
            prev = temp

    # Extract the longest common substring from X
    longest_substr = X[end_index - max_len:end_index]

    return longest_substr, max_len

def find_substring_indices(string, substring):
    start_index = string.find(substring)
    
    if start_index == -1:
        print(substring)
        print("substring not found")
        return None  # Substring not found
    
    end_index = start_index + len(substring) - 1
    return [start_index, end_index]

def merge(intervals):  
    intervals.sort()
    result = [intervals[0]]

    for start, end in intervals[1:]:
        last_end = result[-1][1]

        if last_end < start:
            result.append([start, end])
            
        else:
            result[-1][1] = max(last_end, end)

    return result

def calculate_score(ground_truth_key, predictions_key):
    """"Score predictions for a single annotation type"""
    if ground_truth_key and predictions_key:
        # calculate overlap intervals
        intervals = []
        missed = 0
        for pred in predictions_key:
            overlap, overlap_length = longest_common_substring(ground_truth_key, pred)
            if overlap_length > 0:
                (start, end) = find_substring_indices(ground_truth_key, overlap)
                intervals.append([start, end])
                                 
            missed += len(pred) - overlap_length
    
        if not intervals:
            return [0.0, 0.0]
            
        # merge intervals
        merged_intervals = merge(intervals)
        
        total_covered = sum(end - start + 1 for start, end in merged_intervals)
        total_length = len(ground_truth_key)
        total_pred_length = sum([len(pred) for pred in predictions_key])
    
        recall = total_covered / total_length
        precision = 1 - missed/total_pred_length

        return precision, recall
        
    elif not ground_truth_key:
        return [0.0, 1.0]
    elif not predictions_key:
        return [1.0, 0.0]
    else:
        return [1.0, 1.0]
    
def amalgamate_annotations(ground_truth_path):
    """Turns list of strings annotations into single string"""
    with open(ground_truth_path, 'r', encoding='utf-8') as json_file:
            case_file = json.load(json_file)
    
    ground_truth = {}
    for field in case_file['annotations']:
        ground_truth[field] =  " ".join(case_file['annotations'][field])
    
    return ground_truth

def evaluate(ground_truth, preds):
    performance = np.zeros((5, 3))
    for i, key in enumerate(ground_truth.keys()):
        precision, recall = calculate_score(ground_truth[key], preds[key])
        f1 = (2 * precision * recall) / (precision + recall)

        performance[i] = np.array([precision, recall, f1])

    return performance