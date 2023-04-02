from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import load_train_csv, load_train_sparse, load_valid_csv, load_public_test_csv, create_and_save_plot

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
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
    for user_idx, question_idx, is_correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        x = beta[question_idx] - theta[user_idx]
        log_lklihood += -is_correct * np.logaddexp(0, x) - (1 - is_correct) * np.logaddexp(0, -x)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
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
    # TODO: what about nan entries in sparse matrix?
    # Gradient step for theta
    for user_idx, question_idx, is_correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        x = theta[user_idx] - beta[question_idx]
        theta[user_idx] += lr * (is_correct - sigmoid(x))

    # Gradient step for beta
    for user_idx, question_idx, is_correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        x = theta[user_idx] - beta[question_idx]
        beta[question_idx] += lr * (sigmoid(x) - is_correct)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_data, val_data, lr, iterations):
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
    # Initialize theta and beta TODO:  random.randn? 0-1?
    theta = np.random.randn(542)
    beta = np.random.randn(1774)

    val_acc_lst = []
    trn_acc_lst = []
    trn_neg_llds = []
    val_neg_llds = []

    for i in tqdm(range(iterations)):
        # print(f'ITERATION {i}:')

        # update theta and beta
        theta, beta = update_theta_beta(train_data, lr, theta, beta)

        # negative likelihood on training set
        trn_neg_lld = neg_log_likelihood(data=train_data, theta=theta, beta=beta)
        trn_neg_llds.append(trn_neg_lld)

        # accuracy on training set TODO: TA stick to val?
        trn_score = evaluate(data=train_data, theta=theta, beta=beta)
        trn_acc_lst.append(trn_score)

        # print("TRN NLLK: {} \t TRN Score: {}".format(trn_neg_lld, trn_score))

        # negative likelihood on validation set
        val_neg_lld = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        val_neg_llds.append(val_neg_lld)

        # accuracy on validation set
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)

        # print("VAL NLLK: {} \t VAL Score: {}".format(val_neg_lld, val_score), '\n')

    return theta, beta, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds


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


def train(train_data, val_data, test_data, lr, iterations, save_plots=False):
    # Train the model.
    theta, beta, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds = irt(train_data, val_data, lr, iterations)

    # Evaluate on the validation and test set.
    final_val_acc = evaluate(data=val_data, theta=theta, beta=beta)
    final_test_acc = evaluate(data=test_data, theta=theta, beta=beta)

    # final_test_acc = 0.0  # TODO: REMOVE WHEN WANT TO SEE
    if final_val_acc != val_acc_lst[-1]:
        print('\033[93m' + f'Final validation accuracy: {final_val_acc} is not the same as '
                           f'the last element in the list: {val_acc_lst[-1]}' + '\033[0m')
    print('-' * 30)
    print(f'Hyperparameters: lr={lr}, iterations={iterations}')
    print(f'Final training accuracy: {round(trn_acc_lst[-1] * 100, 4)}')
    print(f'Final validation accuracy: {round(final_val_acc * 100, 4)}')
    print(f'Final test accuracy: {round(final_test_acc * 100, 4)}')
    print('-' * 30)

    # Plot all lists. TODO: change .. to .?
    if save_plots:
        create_and_save_plot(range(iterations), trn_acc_lst, val_acc_lst,
                             'Iterations', 'Training Accuracy',
                             'Accuracy vs. Iterations', '../part_a/images/irt_acc.png')
        create_and_save_plot(range(iterations), trn_neg_llds, val_neg_llds,
                             'Iterations', 'Negative Log Likelihood',
                             'Negative Log Likelihood vs. Iterations', '../part_a/images/irt_nllk.png')

    return theta, beta, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds, final_test_acc


def main():
    # TODO: Change .. to .?
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Set hyperparameters
    lr = 0.01
    iterations = 100

    # Train model
    theta, beta, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds, final_test_acc = \
        train(train_data, val_data, test_data, lr, iterations, save_plots=True)

    ## PART D ##
    # TODO: Switch to deterministic mode?
    # js = np.random.choice(sparse_matrix.shape[1], 3, replace=False)
    js = [214, 1555, 667]
    print(f"Questions: {js}")
    # Range theta from -10 to 10 with step size 0.1
    theta_range = np.arange(-10, 10, 0.1)
    for j_idx, j in enumerate(js):
        probs = []
        for theta in theta_range:
            probs.append(sigmoid(theta - beta[j]))
        plt.plot(theta_range, probs, label=f"Question {j}, beta={round(beta[j], 2)})")
    plt.xlabel("Theta")
    plt.ylabel("Probability of Answering Correctly")
    plt.title("Probability of Answering Correctly vs. Theta")
    plt.legend()
    plt.savefig("../part_a/images/irt_prob_vs_theta.png")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    main()

    # TODO: How come val neg llk is lower than trn neg llk? Shouldn't it be the other way around?
