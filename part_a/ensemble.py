"""
In this problem, you will be implementing bagging ensemble to improve
the stability and accuracy of your base models. Select and train 3 base models with
bootstrapping the training set. To predict the correctness, generate
3 predictions by using the base model and average the predicted correctness. Report the final
validation and test accuracy.
"""
import copy

from utils import *
import numpy as np
from knn import *
from tqdm import tqdm


def resample(sparse_matrix, percent_set):
    """Set num_set elements that are not np.nan to np.nan (sparsify)."""
    # TODO: Ask TA: Is this the right way to do resampling?, this is sample by entry describe in report
    # TODO: Should I sample rows to?
    sparse_matrix = copy.deepcopy(sparse_matrix)
    print(f'OG sparse_matrix.shape: {sparse_matrix.shape}')

    # Create a list of all non-nan indices
    indices = np.argwhere(~np.isnan(sparse_matrix))
    num_set = int(percent_set * indices.shape[0])
    print(f'indices.shape: {indices.shape}, num_set: {num_set}')

    # Shuffle the list
    np.random.shuffle(indices)

    # Set the elems to nan
    for i in tqdm(range(num_set)):
        target_index = indices[i]
        sparse_matrix[target_index[0], target_index[1]] = np.nan

    print(f'New sparse_matrix.shape: {sparse_matrix.shape}')
    return sparse_matrix

    # num_set_so_far = 0
    # while num_set_so_far < num_set:
    # # TODO: Ask TA: OK to do range? (Using range so have tqdm)
    # # for i in tqdm(range(num_set)):
    #     # Print current progress with replacing print line
    #     print(f"[{num_set_so_far}/{num_set}]")
    #     x_idx = np.random.choice(sparse_matrix.shape[0], 1)
    #     y_idx = np.random.choice(sparse_matrix.shape[1], 1)
    #     if not np.isnan(sparse_matrix[x_idx, y_idx]):  # TODO: use isnan elsewhere
    #         sparse_matrix[x_idx, y_idx] = np.nan
    #         num_set_so_far += 1
    # print(f"NAN'd {num_set_so_far} elements (wanted to NAN {num_set}).")
    # return sparse_matrix


def sparse_matrix_evaluate_ensemble(data, mat1, mat2, mat3, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in tqdm(range(len(data["is_correct"]))):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        mat1_pred = mat1[cur_user_id, cur_question_id] >= threshold
        mat2_pred = mat2[cur_user_id, cur_question_id] >= threshold
        mat3_pred = mat3[cur_user_id, cur_question_id] >= threshold

        # This basically does mode operation.
        # TODO: Ask TA: Is this the right way to do ensemble? Mode or average before? Matter?
        pred = (int(mat1_pred) + int(mat2_pred) + int(mat3_pred)) >= 2

        # Now, compare the prediction with the actual value.
        if pred == data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def main():
    # TODO: Need to change .. back to .
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Create 3 samples from the training set.
    percent_set = 0.2
    print(f"Percent of non-NaN elements set to np.nan: {percent_set * 100}%")
    print(f'Resampling {percent_set * 100}% of the training set 3 times...')
    print("\nResampling matrix 1...")
    matrix1 = resample(sparse_matrix, percent_set)
    print("\nResampling matrix 2...")
    matrix2 = resample(sparse_matrix, percent_set)
    print("\nResampling matrix 3...")
    matrix3 = resample(sparse_matrix, percent_set)

    # Compute predictions and average them, then take the mode and compute accuracy.
    k1, k2, k3 = 11, 11, 12  # TODO: Choose best k's.
    print("\nFitting matrices...")
    nbrs1 = KNNImputer(n_neighbors=k1)
    mat1 = nbrs1.fit_transform(matrix1)
    print("Finished computing matrix 1.")
    nbrs2 = KNNImputer(n_neighbors=k2)
    mat2 = nbrs2.fit_transform(matrix2)
    print("Finished computing matrix 2.")
    nbrs3 = KNNImputer(n_neighbors=k3)
    mat3 = nbrs3.fit_transform(matrix3)
    print("Finished computing matrix 3.")

    print("\nEvaluating matrices...")
    val_acc = sparse_matrix_evaluate_ensemble(val_data, mat1, mat2, mat3)
    print(f"\nValidation accuracy: {val_acc}")

    # Check single matrix accuracies
    val_acc_mat1 = sparse_matrix_evaluate(val_data, mat1)
    print(f"\nValidation accuracy (mat1 only): {val_acc_mat1}")
    val_acc_mat2 = sparse_matrix_evaluate(val_data, mat2)
    print(f"\nValidation accuracy (mat2 only): {val_acc_mat2}")
    val_acc_mat3 = sparse_matrix_evaluate(val_data, mat3)
    print(f"\nValidation accuracy (mat3 only): {val_acc_mat3}")

    # TODO: Only at the end.
    # test_data = sparse_matrix_evaluate_ensemble(test_data, mat1, mat2, mat3)


if __name__ == "__main__":
    main()
