from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(224)

def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    
    u[n] = u[n] + lr * (c - np.dot(u[n], z[q])) * z[q]
    z[q] = u[n] + lr * (c - np.dot(u[n], z[q])) * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################    
    for _ in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        
    mat = u @ np.transpose(z)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("./data").toarray() # (542, 1774)
    train_data = load_train_csv("./data") # 56688
    val_data = load_valid_csv("./data") # 7086
    test_data = load_public_test_csv("./data") # 3543

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    print('\nSVD:')
    k_vals = [1, 10, 15, 20, 25, 30]
    
    for k in k_vals:
        reconst_matrix = svd_reconstruct(train_matrix, k)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        print(f'Validation Accuracy for k={k}: {acc}')
        
    k_ = 25
    reconst_matrix = svd_reconstruct(train_matrix, k_)
    acc = sparse_matrix_evaluate(test_data, reconst_matrix)
    print(f'Test Accuracy for k^*={k_}: {acc}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################  
    lr = 0.5
    num_iteration = 700
    x = list(range(1, num_iteration+1))
    k = 20
    
    train_loss = []
    val_loss = []
    
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                        size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                        size=(len(set(train_data["question_id"])), k))
    
    print(f'\nALS (k={k}, lr={lr}, iters={num_iteration}):')
    for _ in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
    
        train_loss.append(squared_error_loss(train_data, u, z))
        val_loss.append(squared_error_loss(val_data, u, z))
    
    # plot train loss
    plt.figure()
    plt.plot(x, train_loss, '-b')
    plt.title(f'Train Squared-Error-Loss vs. Iterations (k={k}, lr={lr})')
    plt.xlabel('Iteration')
    plt.ylabel('Squared-Error-Loss')
    plt.savefig(f'./part_a/images/ALS-train_loss_vs_iter-k={k}-lr={str(lr).replace(".", "p")}-iters={num_iteration}.png')
    
    # plot validation loss
    plt.figure()
    plt.plot(x, val_loss, '-r')
    plt.title(f'Validation Squared-Error-Loss vs. Iterations (k={k}, lr={lr})')
    plt.xlabel('Iteration')
    plt.ylabel('Squared-Error-Loss')
    plt.savefig(f'./part_a/images/ALS-val_loss_vs_iter-k={k}-lr={str(lr).replace(".", "p")}-iters={num_iteration}.png')
    
    # report final results  
    mat = u @ np.transpose(z)
    train_acc = sparse_matrix_evaluate(train_data, mat)
    val_acc = sparse_matrix_evaluate(val_data, mat)
    test_acc = sparse_matrix_evaluate(test_data, mat)
    
    print(f'Train Loss: {train_loss[-1]}')
    print(f'Validation Loss: {val_loss[-1]}')
    print(f'Train Accuracy: {train_acc}')
    print(f'Validation Accuracy: {val_acc}')
    print(f'Test Accuracy: {test_acc}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
