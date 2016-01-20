import numpy as np
from enum import Enum


class Algorithm(Enum):
    RegressTree = 1
    ModelTree = 2


def split_by_col(parent_mat, col_index, col_value):
    m, n = np.shape(parent_mat)
    select_col = parent_mat[:, col_index]
    left_index_list = select_col > col_value
    left_index_list = left_index_list.tolist()
    # right_index_list = select_col <= col_value
    left_mat = np.empty((0, n))
    right_mat = np.empty((0, n))
    for i in range(m):
        if left_index_list[i][0] is True:
            left_mat = np.vstack((left_mat, parent_mat[i, :].copy()))
        else:
            right_mat = np.vstack((right_mat, parent_mat[i, :].copy()))
    return left_mat, right_mat


def _calc_mean(mat):
    return np.mean(mat[:, -1])


def _calc_vat(mat):
    m = len(mat)
    value_mat = mat[:, -1]
    return np.var(value_mat, axis=0) * m


def _to_set(input_mat):
    r_set = set()
    input_list = input_mat.tolist()
    for l in input_list:
        r_set.add(l[0])
    return r_set


def get_best_split(parent_mat, algorithm, min_err, min_row_count):
    if algorithm == Algorithm.RegressTree:
        return _get_best_regress_split(parent_mat, min_err, min_row_count)
    elif algorithm == Algorithm.ModelTree:
        return _get_best_model_split(parent_mat, min_err, min_row_count)
    return _get_best_regress_split(parent_mat, min_err, min_row_count)


def _get_best_regress_split(parent_mat, min_err, min_row_count):
    value_col = parent_mat[:, -1]
    # only has one value, do not split
    if len(_to_set(value_col)) == 1:
        return -1, _calc_mean(parent_mat)  # this tuple means do not split
    # get the best split column index and value
    m, n = np.shape(parent_mat)
    p_var = _calc_vat(parent_mat)
    best_var = float("inf")
    best_index = -1
    best_value = 0
    for index in range(n - 1):
        col_values = _to_set(parent_mat[:, index])
        for value in col_values:
            l_mat, r_mat = split_by_col(parent_mat, index, value)
            if len(l_mat) < min_row_count or len(r_mat) < min_row_count:
                continue
            c_var = _calc_vat(l_mat) + _calc_vat(r_mat)
            if c_var < best_var:
                best_index = index
                best_value = value
                best_var = c_var
    # if the variance between parent and children is too small, do not split
    if (p_var - best_var) < min_err:
        return -1, _calc_mean(parent_mat)
    return best_index, best_value


def _calc_weight(mat):
    m, n = np.shape(mat)
    # build y mat
    y_mat = np.empty((m, 0))
    y_mat = np.hstack((y_mat, mat[:, n-1].copy()))
    # build x mat: add constant to mat
    x_mat = np.ones((m, 1))
    for i in range(n-1):
        x_mat = np.hstack((x_mat, mat[:, i].copy()))
    tmp_mat = x_mat.transpose() * x_mat
    if np.linalg.det(tmp_mat) == 0.0:
        print("the xTx matrix is singular, can not do inverse")
        return x_mat, y_mat, []
    weights = np.linalg.inv(tmp_mat) * x_mat.transpose() * y_mat
    return x_mat, y_mat, weights


def _calc_err(mat):
    x_mat, y_mat, weights = _calc_weight(mat)
    if len(weights) == 0:
        return float("inf")
    y_tmp_mat = x_mat * weights
    err = np.sum(np.power(y_mat - y_tmp_mat, 2))
    return err


def _get_best_model_split(parent_mat, min_err, min_row_count):
    value_col = parent_mat[:, -1]
    # only has one value, do not split
    if len(_to_set(value_col)) == 1:
        return -1, _calc_weight(parent_mat)[2]  # this tuple means do not split
    # get the best split column index and value
    m, n = np.shape(parent_mat)
    p_err = _calc_err(parent_mat)
    best_err = float("inf")
    best_index = -1
    best_value = 0
    for index in range(n - 1):
        col_values = _to_set(parent_mat[:, index])
        for value in col_values:
            l_mat, r_mat = split_by_col(parent_mat, index, value)
            if len(l_mat) < min_row_count or len(r_mat) < min_row_count:
                continue
            c_err = _calc_err(l_mat) + _calc_err(r_mat)
            if c_err < best_err:
                best_index = index
                best_value = value
                best_err = c_err
    # if the variance between parent and children is too small, do not split
    if (p_err - best_err) < min_err:
        return -1, _calc_weight(parent_mat)[2]
    return best_index, best_value




