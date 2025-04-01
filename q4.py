import numpy as np
import pandas as pd
sample_table = [[25,2,0,0,0],[30,2,0,1,0],[35,1,0, 0,1],[40,0,0,0,1],[45,0,0,0,1],[50,0,1,1,0],[55,1,1,1,1],[60,2,0,0,0]]
sample_table_coloumns = ["Age", "Income", "Student", "Credit Rating", "Buy Computer"]
df = pd.DataFrame.from_records(sample_table, columns=sample_table_coloumns)

def calculate_gini(p_class):
    return 2*(1-p_class)*(p_class)

def weight_gini_impurity(left, right, target_col):
    number_left = len(left)
    number_right = len(right)
    total = number_left + number_right

    p_left = 0
    p_right = 0
    if number_left > 0:
        p_left = left[target_col].mean()
    if number_right > 0:
        p_right = right[target_col].mean() 

    # Calculate the binary Gini impurity using 2 * p * (1-p)
    gini_left = 2 * p_left * (1 - p_left)
    gini_right = 2 * p_right * (1 - p_right)

    return (number_left / total) * gini_left + (number_right / total) * gini_right

income_label = [0.5,1.5]
student_status_label = [0.5]
credit_status = [0.5]
age_status = [27.5,32.5,37.5,42.5,47.5,52.5,57.5]

feature_array = [age_status, income_label, student_status_label, credit_status]
input_columns = sample_table_coloumns[:-1]
best_gini = float('inf')
best_feature = None
best_split = None

# Calculating the best split, < and >=
for idx in range(len(input_columns)):
    col_name = input_columns[idx]
    # Get candidate thresholds for this feature
    candidate_thresholds = feature_array[idx]
    for threshold in candidate_thresholds:
        left_split = df[df[col_name] < threshold]
        right_split = df[df[col_name] >= threshold]
        gini_impurity = weight_gini_impurity(left_split, right_split, sample_table_coloumns[-1])
        if gini_impurity < best_gini:
            best_gini = gini_impurity
            best_feature = col_name
            best_split = threshold

print("Best Gini impurity:", best_gini)
print("Best feature:", best_feature)
print("Best split threshold:", best_split)

def build_tree(df, depth, max_depth, min_samples, input_columns, feature_array, target_col):
    # Stopping conditions:
    #   1. Maximum depth reached
    #   2. Too few samples to split further
    #   3. Node is pure (all target values the same)
    if depth >= max_depth or len(df) < min_samples or df[target_col].nunique() == 1:
        # Create a leaf node with the majority class prediction
        prediction = df[target_col].mode()[0]
        return {"type": "leaf", "prediction": prediction}
    
    best_gini = float('inf')
    best_feature = None
    best_split = None
    best_left = None
    best_right = None
    
    for idx in range(len(input_columns)):
        col_name = input_columns[idx]
        candidate_thresholds = feature_array[idx]  # Candidate thresholds for this feature
        for threshold in candidate_thresholds:
            left_split = df[df[col_name] < threshold]
            right_split = df[df[col_name] >= threshold]
            
            if len(left_split) == 0 or len(right_split) == 0:
                continue
            
            gini_impurity = weight_gini_impurity(left_split, right_split, target_col)
            if gini_impurity < best_gini:
                best_gini = gini_impurity
                best_feature = col_name
                best_split = threshold
                best_left = left_split
                best_right = right_split

    if best_feature is None:
        prediction = df[target_col].mode()[0]
        return {"type": "leaf", "prediction": prediction}

    # Recursively build the left and right subtrees.
    left_subtree = build_tree(best_left, depth + 1, max_depth, min_samples, input_columns, feature_array, target_col)
    right_subtree = build_tree(best_right, depth + 1, max_depth, min_samples, input_columns, feature_array, target_col)
    
    return {
        "type": "node",
        "feature": best_feature,
        "threshold": best_split,
        "left": left_subtree,
        "right": right_subtree
    }

max_depth  = 3
min_samples = 2

tree = build_tree(df, depth=0, max_depth=max_depth, min_samples=min_samples, 
                  input_columns=input_columns, feature_array=feature_array, 
                  target_col=sample_table_coloumns[-1])

def print_tree(node, indent=""):
    """
    Recursively print the decision tree in a friendly format.
    
    Parameters:
        node (dict): A node in the decision tree.
        indent (str): Indentation string for current depth level.
    """
    if node["type"] == "leaf":
        print(indent + "Leaf: Predict =", node["prediction"])
    else:
        print(indent + f"Node: if {node['feature']} < {node['threshold']}")
        print(indent + "  Left:")
        print_tree(node["left"], indent + "    ")
        print(indent + "  Right:")
        print_tree(node["right"], indent + "    ")

print("Decision Tree Structure:")
print_tree(tree)

def predict(tree, sample):
    if tree["type"] == "leaf":
        return tree["prediction"]
    else:
        feature = tree["feature"]
        threshold = tree["threshold"]
        if sample[feature] < threshold:
            return predict(tree["left"], sample)
        else:
            return predict(tree["right"], sample)

test_sample = {"Age": 42, "Income": 1, "Student": 0, "Credit Rating": 1}
print("Predicted class:", predict(tree, test_sample))

def bagging(data, feature_array, target, max_depth, min_samples, num_tree, max_feature=None):
    """
    Build an ensemble of trees using bootstrap sampling.
    
    Parameters:
      data (pd.DataFrame): The training data.
      feature_array (list): List of candidate thresholds for each predictor.
      target (str): Target column name.
      max_depth (int): Maximum depth of each tree.
      min_samples (int): Minimum samples required to split.
      num_tree (int): Number of trees to build.
      max_feature (int or None): If provided, use only this many random predictors at each split.
      
    Returns:
      trees (list): List of trees.
      oob_indices_list (list): List of numpy arrays containing indices not in the bootstrap sample.
    """
    sample_size = len(data)
    trees = []
    oob_indices_list = []
    for i in range(num_tree):
        bootstrap_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        oob_indices = np.setdiff1d(np.arange(sample_size), bootstrap_indices)
        oob_indices_list.append(oob_indices)
        
        bootstrap_data = data.iloc[bootstrap_indices]
        tree = build_tree(bootstrap_data, depth=0, max_depth=max_depth, min_samples=min_samples,
                          input_columns=input_columns, feature_array=feature_array, target_col=target)
        trees.append(tree)
    print(tree)
    print(oob_indices_list)
    return trees, oob_indices_list

def compute_oob_error(data, trees, oob_indices_list, target):
    """
    Compute the Out-Of-Bag (OOB) error for an ensemble.
    
    Parameters:
      data (pd.DataFrame): The full training data.
      trees (list): List of trees in the ensemble.
      oob_indices_list (list): List of arrays with OOB indices per tree.
      target (str): Target column name.
      
    Returns:
      oob_error (float): Fraction of misclassified samples (if any).
    """
    predictions = {}
    n = len(data)
    
    for tree, oob_idxs in zip(trees, oob_indices_list):
        for idx in oob_idxs:
            sample = data.iloc[idx]
            pred = tree_predict(tree, sample)
            if idx in predictions:
                predictions[idx].append(pred)
            else:
                predictions[idx] = [pred]
    
    errors = 0
    total = 0
    for idx, preds in predictions.items():
        # Majority vote: if average >= 0.5, vote 1, else 0.
        vote = 1 if np.mean(preds) >= 0.5 else 0
        true_val = data.iloc[idx][target]
        if vote != true_val:
            errors += 1
        total += 1
    return errors / total if total > 0 else None

def tree_predict(tree, sample):
    """
    Recursively traverse the tree to make a prediction for a single sample.
    """
    if tree["type"] == "leaf":
        return tree["prediction"]
    else:
        if sample[tree["feature"]] < tree["threshold"]:
            return tree_predict(tree["left"], sample)
        else:
            return tree_predict(tree["right"], sample)

num_tree = 10  
max_depth = 3
min_samples = 2

# Bagging with all predictors.
ensemble_trees, oob_list = bagging(df, feature_array, sample_table_coloumns[-1],
                                   max_depth, min_samples, num_tree, max_feature=None)
oob_error = compute_oob_error(df, ensemble_trees, oob_list, sample_table_coloumns[-1])
print("OOB Error (bagging 10 trees using all predictors):", oob_error)

# Bagging with random predictor subsetting (using only 2 random predictors per split).
# To use random predictor subsetting, modify build_tree to choose only max_feature predictors at each split.
ensemble_trees_sub, oob_list_sub = bagging(df, feature_array, sample_table_coloumns[-1],
                                           max_depth, min_samples, num_tree, max_feature=2)
oob_error_sub = compute_oob_error(df, ensemble_trees_sub, oob_list_sub, sample_table_coloumns[-1])
print("OOB Error (bagging 10 trees using 2 random predictors per split):", oob_error_sub)

            

