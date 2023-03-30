"""
In this problem, you will be implementing bagging ensemble to improve
the stability and accuracy of your base models. Select and train 3 base models with
bootstrapping the training set. To predict the correctness, generate
3 predictions by using the base model and average the predicted correctness. Report the final
validation and test accuracy.
"""
from utils import *
import numpy as np
from knn import *


def resample(sparse_matrix, num_set):
    """Set num_set elements that are not np.nan to np.nan (sparsify)."""
    num_set_so_far = 0
    while num_set_so_far < num_set:
        x_idx = np.random.choice(sparse_matrix.shape[0], 1)
        y_idx = np.random.choice(sparse_matrix.shape[1], 1)
        if not np.isnan(sparse_matrix[x_idx, y_idx]): # TODO: use isnan elsewhere
            sparse_matrix[x_idx, y_idx] = np.nan
            num_set_so_far += 1
    return sparse_matrix

def sparse_matrix_evaluate_ensemble(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)

def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Create 3 samples from the training set.
    percent_set = 0.2
    num_set = int(percent_set * sparse_matrix.shape[0] * sparse_matrix.shape[1])
    matrix1 = resample(sparse_matrix, num_set)
    matrix2 = resample(sparse_matrix, num_set)
    matrix3 = resample(sparse_matrix, num_set)

    # Compute predictions and average them, then take the mode and compute accuracy.
    nbrs = KNNImputer(n_neighbors=k)
    mat1 = nbrs.fit_transform(matrix1)
    mat2 = nbrs.fit_transform(matrix2)
    mat3 = nbrs.fit_transform(matrix3)
    acc = sparse_matrix_evaluate_ensemble(valid_data, mat)
