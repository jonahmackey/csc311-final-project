import csv
import os

from matplotlib import pyplot as plt

from utils import load_train_csv, load_train_sparse, load_valid_csv, load_public_test_csv, create_and_save_plot
from part_a.item_response import sigmoid
from part_a.item_response import train as v0_train
import numpy as np
from tqdm import tqdm


def v1_compute_x(user_idx, question_idx, subject_idxs, theta, beta):
    return theta[user_idx][subject_idxs].sum() - beta[question_idx]


def v2_compute_x(user_idx, question_idx, subject_idxs, theta, beta):
    return theta[user_idx][subject_idxs].sum() - len(subject_idxs) * beta[question_idx]


def v3_compute_x(user_idx, question_idx, subject_idxs, theta, beta, similarities):
    return theta[user_idx][subject_idxs].sum() - beta[question_idx] * \
        similarities[question_idx][subject_idxs].sum()


def v1_neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood of version 1.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param theta: List of Vectors
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = beta[question_idx] - theta[user_idx][subject_idxs].sum()
        log_lklihood += -is_correct * np.logaddexp(0, x) - (1 - is_correct) * np.logaddexp(0, -x)
    return -log_lklihood


def v2_neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood of version 2.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param theta: List of Vectors
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = len(subject_idxs) * beta[question_idx] - theta[user_idx][subject_idxs].sum()
        log_lklihood += -is_correct * np.logaddexp(0, x) - (1 - is_correct) * np.logaddexp(0, -x)
    return -log_lklihood


def v3_neg_log_likelihood(data, theta, beta, similarities):
    """Compute the negative log-likelihood of version 3.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param theta: List of Vectors
    :param beta: Vector
    :param similarities: Matrix of similarities of questions to subjects (questions x subjects).
    :return: float
    """
    log_lklihood = 0.
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = beta[question_idx] * similarities[question_idx][subject_idxs].sum() \
            - theta[user_idx][subject_idxs].sum()
        log_lklihood += -is_correct * np.logaddexp(0, x) - (1 - is_correct) * np.logaddexp(0, -x)
    return -log_lklihood


def v1_update(data, lr, theta, beta):
    """Update theta and beta using gradient descent for V1.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    # Gradient step for theta
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = v1_compute_x(user_idx, question_idx, subject_idxs, theta, beta)
        grad = is_correct - sigmoid(x)
        for subject_idx in subject_idxs:  # TODO: Check with Jonah if this is correct.
            theta[user_idx][subject_idx] += lr * grad

    # Gradient step for beta
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = v1_compute_x(user_idx, question_idx, subject_idxs, theta, beta)
        beta[question_idx] += lr * (sigmoid(x) - is_correct)
    return theta, beta


def v2_update(data, lr, theta, beta):
    """Update theta and beta using gradient descent for V2.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    # Gradient step for theta
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = v2_compute_x(user_idx, question_idx, subject_idxs, theta, beta)
        grad = is_correct - sigmoid(x)
        for subject_idx in subject_idxs:
            theta[user_idx][subject_idx] += lr * grad

    # Gradient step for beta
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = v2_compute_x(user_idx, question_idx, subject_idxs, theta, beta)
        grad = len(subject_idxs) * (sigmoid(x) - is_correct)
        beta[question_idx] += lr * grad
    return theta, beta


def v3_update(data, lr, theta, beta, similarities):
    """Update theta, beta, and similarities using gradient descent for V3.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param similarities: Matrix of similarities of questions to subjects (questions x subjects).
    :return: tuple of vectors
    """
    # Gradient step for theta
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = v3_compute_x(user_idx, question_idx, subject_idxs, theta, beta, similarities)
        grad = is_correct - sigmoid(x)
        for subject_idx in subject_idxs:
            theta[user_idx][subject_idx] += lr * grad

    # Gradient step for beta
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = v3_compute_x(user_idx, question_idx, subject_idxs, theta, beta, similarities)
        grad = similarities[question_idx][subject_idxs].sum() * (sigmoid(x) - is_correct)
        beta[question_idx] += lr * grad

    # Gradient step for similarities, TODO: Should I be looping over all users too?
    for user_idx, question_idx, is_correct, subject_idxs in \
            zip(data["user_id"], data["question_id"], data["is_correct"], data["subject_ids"]):
        x = v3_compute_x(user_idx, question_idx, subject_idxs, theta, beta, similarities)
        grad = beta[question_idx] * (sigmoid(x) - is_correct)
        for subject_idx in subject_idxs:
            similarities[question_idx][subject_idx] += lr * grad
    return theta, beta, similarities


def irt(train_data, val_data, lr, iterations, num_subjects, version):
    """ Train IRT model.

    :param train_data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param val_data: Same as train_data but for validation set.
    :param lr: float
    :param iterations: int
    :param num_subjects: int
    :param version: int (1, 2, or 3)
    :return: (theta, beta, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds)
    """
    # Initialize theta and beta TODO:  random.randn? 0-1?
    theta = np.random.randn(542, num_subjects)
    beta = np.random.randn(1774)
    if version == 3:
        similarities = np.random.randn(1774, num_subjects)

    val_acc_lst = []
    trn_acc_lst = []
    trn_neg_llds = []
    val_neg_llds = []

    for i in tqdm(range(iterations)):
        if version == 3:
            theta, beta, similarities = v3_update(train_data, lr, theta, beta, similarities)
            trn_neg_lld = v3_neg_log_likelihood(data=train_data, theta=theta, beta=beta, similarities=similarities)
            trn_score = v3_evaluate(data=train_data, theta=theta, beta=beta, similarities=similarities)
            val_neg_lld = v3_neg_log_likelihood(data=val_data, theta=theta, beta=beta, similarities=similarities)
            val_score = v3_evaluate(data=val_data, theta=theta, beta=beta, similarities=similarities)
        elif version == 2:
            theta, beta = v2_update(train_data, lr, theta, beta)
            trn_neg_lld = v2_neg_log_likelihood(data=train_data, theta=theta, beta=beta)
            trn_score = v2_evaluate(data=train_data, theta=theta, beta=beta)
            val_neg_lld = v2_neg_log_likelihood(data=val_data, theta=theta, beta=beta)
            val_score = v2_evaluate(data=val_data, theta=theta, beta=beta)
        elif version == 1:
            theta, beta = v1_update(train_data, lr, theta, beta)
            trn_neg_lld = v1_neg_log_likelihood(data=train_data, theta=theta, beta=beta)
            trn_score = v1_evaluate(data=train_data, theta=theta, beta=beta)
            val_neg_lld = v1_neg_log_likelihood(data=val_data, theta=theta, beta=beta)
            val_score = v1_evaluate(data=val_data, theta=theta, beta=beta)
        else:
            raise ValueError("Version must be 1, 2, or 3.")

        trn_neg_llds.append(trn_neg_lld)
        trn_acc_lst.append(trn_score)
        val_neg_llds.append(val_neg_lld)
        val_acc_lst.append(val_score)
        if i % 10 == 0:
            print(f"\nITERATION: {i}")
            print(f"\tTRN NLLK: {trn_neg_lld} \t TRN Score: {trn_score}")
            print(f"\tVAL NLLK: {val_neg_lld} \t VAL Score: {val_score}", '\n')

    if version == 3:
        return theta, beta, similarities, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds
    else:
        return theta, beta, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds


def v1_evaluate(data, theta, beta):
    """Evaluate the v1 model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        subject_idxs = data["subject_ids"][i]  # TODO: make sure right
        x = (v1_compute_x(u, q, subject_idxs, theta, beta)).sum()
        assert v1_compute_x(u, q, subject_idxs, theta, beta) == x
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def v2_evaluate(data, theta, beta):
    """Evaluate the v2 model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        subject_idxs = data["subject_ids"][i]  # TODO: make sure right
        x = (v2_compute_x(u, q, subject_idxs, theta, beta)).sum()
        assert v2_compute_x(u, q, subject_idxs, theta, beta) == x
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def v3_evaluate(data, theta, beta, similarities):
    """Evaluate the v3 model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list, subject_ids: list[list[int]]}
    :param theta: Vector
    :param beta: Vector
    :param similarities: Matrix of similarities of questions to subjects (questions x subjects).
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        subject_idxs = data["subject_ids"][i]  # TODO: make sure right
        x = (v3_compute_x(u, q, subject_idxs, theta, beta, similarities)).sum()
        assert v3_compute_x(u, q, subject_idxs, theta, beta, similarities) == x
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def train(train_data, val_data, test_data, question_meta, lr, iterations, version):
    all_subject_ids = []
    question_id_to_subject_ids = {}
    for question_id, subject_ids in question_meta:
        question_id = int(question_id)
        subject_ids = subject_ids.strip('][').split(', ')
        subject_ids = [int(s) for s in subject_ids]
        # Add subject ids to list of all subject ids
        all_subject_ids.extend(subject_ids)
        # Add to dict mapping question id to subject ids so that we can update data afterwards
        question_id_to_subject_ids[question_id] = subject_ids
    all_subject_ids = list(set(all_subject_ids))
    if all_subject_ids != list(range(len(all_subject_ids))):
        print("Note: Subject ids are not contiguous integers starting at 0.")
        print("The following subject ids are missing:")
        print(sorted(set(range(len(all_subject_ids))) - set(all_subject_ids)))
    all_subject_ids.sort()
    print("All subject ids:", all_subject_ids)

    # num_subjects is set to the largest subject id + 1, not the number of subjects
    # as this makes the code simpler (so that we can use subject ids as indices)
    # TODO: make sure this is right, Check that Beta is the same?
    num_subjects = all_subject_ids[-1] + 1

    # Update data to include subject ids
    for data in [train_data, val_data, test_data]:
        data["subject_ids"] = []
        for i, q in enumerate(data["question_id"]):
            if q not in question_id_to_subject_ids:
                raise ValueError("Question id {} not in question_meta.".format(q))
            subject_ids = question_id_to_subject_ids[q]
            data["subject_ids"].append(subject_ids)

    if version == 3:
        theta, beta, similarities, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds = \
            irt(train_data, val_data, lr, iterations, num_subjects, version)
        final_val_acc = v3_evaluate(data=val_data, theta=theta, beta=beta, similarities=similarities)
        final_test_acc = v3_evaluate(data=test_data, theta=theta, beta=beta, similarities=similarities)
    else:
        theta, beta, val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds = \
            irt(train_data, val_data, lr, iterations, num_subjects, version)
        if version == 1:
            final_val_acc = v1_evaluate(data=val_data, theta=theta, beta=beta)
            final_test_acc = v1_evaluate(data=test_data, theta=theta, beta=beta)
        elif version == 2:
            final_val_acc = v2_evaluate(data=val_data, theta=theta, beta=beta)
            final_test_acc = v2_evaluate(data=test_data, theta=theta, beta=beta)

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

    # Plot all lists with version in title and filename
    # create_and_save_plot(range(iterations), trn_acc_lst, val_acc_lst,
    #                      'Iterations', 'Accuracy',
    #                      f'Accuracy vs. Iterations (v{version})',
    #                      f'./part_b_images/irt_acc_v{version}.png')
    # create_and_save_plot(range(iterations), trn_neg_llds, val_neg_llds,
    #                      'Iterations', 'Negative Log Likelihood',
    #                      f'Negative Log Likelihood vs. Iterations (v{version})',
    #                      f'./part_b_images/irt_nllk_v{version}.png')

    return val_acc_lst, trn_acc_lst, trn_neg_llds, val_neg_llds, final_test_acc


def load_question_meta(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        question_meta = []
        for line in reader:
            question_meta.append(line)
    return question_meta


def plot_all_versions(iterations, v0_trn, v0_val, v1_trn, v1_val, v2_trn, v2_val, v3_trn, v3_val,
                      title, x_label, y_label, filename,
                      v0_test_pt=None, v1_test_pt=None, v2_test_pt=None, v3_test_pt=None):
    # Plot all training and validation data on same plot
    # (training is solid, validation is dashed, and each version is a different colour)
    # v0 should be black
    plt.figure()
    plt.plot(range(iterations), v0_trn, 'k', label='v0 training')
    plt.plot(range(iterations), v0_val, 'k--', label='v0 validation')
    plt.plot(range(iterations), v1_trn, 'r', label='v1 training')
    plt.plot(range(iterations), v1_val, 'r--', label='v1 validation')
    plt.plot(range(iterations), v2_trn, 'g', label='v2 training')
    plt.plot(range(iterations), v2_val, 'g--', label='v2 validation')
    plt.plot(range(iterations), v3_trn, 'b', label='v3 training')
    plt.plot(range(iterations), v3_val, 'b--', label='v3 validation')

    if v0_test_pt is not None:
        plt.plot(v0_test_pt[0], v0_test_pt[1], 'ko', label='v0 test')
    if v1_test_pt is not None:
        plt.plot(v1_test_pt[0], v1_test_pt[1], 'ro', label='v1 test')
    if v2_test_pt is not None:
        plt.plot(v2_test_pt[0], v2_test_pt[1], 'go', label='v2 test')
    if v3_test_pt is not None:
        plt.plot(v3_test_pt[0], v3_test_pt[1], 'bo', label='v3 test')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(fontsize='small')
    plt.savefig(filename)


def main():
    os.makedirs('./part_b_images', exist_ok=True)

    # TODO: Change .. to .
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    question_meta = load_question_meta("./data/question_meta.csv")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lrs = [0.01, 0.001, 0.0001, 0.1]
    iterations = 30
    for lr in lrs:
        print(f'Training v0 with lr={lr} and iterations={iterations}...')
        _, _, v0_val_acc_lst, v0_trn_acc_lst, v0_trn_neg_llds, v0_val_neg_llds, v0_test_acc = \
            v0_train(train_data, val_data, test_data, lr=lr, iterations=iterations)
        print(f'Training v1 with lr={lr} and iterations={iterations}...')
        v1_val_acc_lst, v1_trn_acc_lst, v1_trn_neg_llds, v1_val_neg_llds, v1_test_acc = \
            train(train_data, val_data, test_data, question_meta, lr=lr, iterations=iterations, version=1)
        print(f'Training v2 with lr={lr} and iterations={iterations}...')
        v2_val_acc_lst, v2_trn_acc_lst, v2_trn_neg_llds, v2_val_neg_llds, v2_test_acc = \
            train(train_data, val_data, test_data, question_meta, lr=lr, iterations=iterations, version=2)
        print(f'Training v3 with lr={lr} and iterations={iterations}...')
        v3_val_acc_lst, v3_trn_acc_lst, v3_trn_neg_llds, v3_val_neg_llds, v3_test_acc = \
            train(train_data, val_data, test_data, question_meta, lr=lr, iterations=iterations, version=3)

        plot_all_versions(iterations, v0_trn_acc_lst, v0_val_acc_lst,
                          v1_trn_acc_lst, v1_val_acc_lst,
                          v2_trn_acc_lst, v2_val_acc_lst,
                          v3_trn_acc_lst, v3_val_acc_lst,
                          'Accuracy vs. Iterations', 'Iterations', 'Accuracy',
                          f'./part_b_images/irt_acc_all_lr{lr}.png',
                          v0_test_pt=(iterations, v0_test_acc),
                          v1_test_pt=(iterations, v1_test_acc),
                          v2_test_pt=(iterations, v2_test_acc),
                          v3_test_pt=(iterations, v3_test_acc))

        plot_all_versions(iterations, v0_trn_neg_llds, v0_val_neg_llds,
                          v1_trn_neg_llds, v1_val_neg_llds,
                          v2_trn_neg_llds, v2_val_neg_llds,
                          v3_trn_neg_llds, v3_val_neg_llds,
                          'Negative Log Likelihood vs. Iterations', 'Iterations', 'Negative Log Likelihood',
                          f'./part_b_images/irt_nllk_all_lr{lr}.png')


if __name__ == "__main__":
    main()
    # TODO: make sure subject_ids are the actual indices of the subjects
    # TODO: V2: beta first in first term make sure rest of math is right
    # TODO: V3: add 'in implementatioin' notes
