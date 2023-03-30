from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(matrix, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c_ij = matrix[i, j]
            first_term = -c_ij * np.log(np.exp(beta[j] - theta[i]) + 1)
            second_term = -(1 - c_ij) * np.log(1 + np.exp(theta[i] - beta[j]))
            log_lklihood += first_term + second_term
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for idx in range(len(theta)):
        theta_derivative = 0.
        beta_derivative = 0.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                c_ij = matrix[i, j]
                if c_ij != np.nan:
                    theta_derivative += ...  # TODO
                    beta_derivative += ...  # TODO
        theta[idx] = theta[idx] - lr * theta_derivative

    for 
        beta[idx] = beta[idx] - lr * beta_derivative
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(matrix, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = None
    beta = None

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(matrix, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(matrix, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = ...
    iterations = ...
    theta, beta, val_acc_lst = irt(sparse_matrix, val_data, lr, iterations)
    final_val_acc = evaluate(data=val_data, theta=theta, beta=beta)
    final_test_acc = evaluate(data=test_data, theta=theta, beta=beta)
    if final_val_acc != val_acc_lst[-1]:
        print('\033[93m' + f'Final validation accuracy: {final_val_acc} is not the same as '
                           f'the last element in the list: {val_acc_lst[-1]}' + '\033[0m')
    print('-' * 30)
    print(f'Final validation accuracy: {final_val_acc}')
    print(f'Final test accuracy: {final_test_acc}')
    print('-' * 30)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
