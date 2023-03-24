from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = np.transpose(nbrs.fit_transform(np.transpose(matrix)))
    acc = sparse_matrix_evaluate(valid_data, mat)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Q1. (a)
    print('\nkNN Impute by User:')
    k_vals = [1, 6, 11, 16, 21, 26]
    accuracy = []
    
    for k in k_vals:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        print(f'Validation Accuracy for k={k}: {acc}')
        accuracy.append(acc)
    
    # plotting
    plt.figure()
    
    plt.plot(k_vals, accuracy, '-o')
    plt.title('Validation Accuracy vs. k (Impute by User)')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_vals)
    plt.savefig('./part_a/images/val_acc_vs_k-knn_impute_by_user.png')
    
    # Q1. (b)
    k_ = 11
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_)
    print(f'Test Accuracy for k^*={k_}: {test_acc}')
    
    # Q1. (c)
    print('\nkNN Impute by Item:')
    k_vals = [1, 6, 11, 16, 21, 26]
    accuracy = []
    
    for k in k_vals:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        print(f'Validation Accuracy for k={k}: {acc}')
        accuracy.append(acc)
    
    # plotting
    plt.figure()
    
    plt.plot(k_vals, accuracy, '-o')
    plt.title('Validation Accuracy vs. k (Impute by Item)')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_vals)
    plt.savefig('./part_a/images/val_acc_vs_k-knn_impute_by_item.png')
    
    k_ = 21
    test_acc = knn_impute_by_item(sparse_matrix, test_data, k_)
    print(f'Test Accuracy for k^*={k_}: {test_acc}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
