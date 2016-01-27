import numpy as np


def duclidean_distance(x, y):
    return 1.0 / (1.0 + np.linalg.norm(x - y))


def pearson_correlation(x, y):
    return 0.5 + 0.5 * np.corrcoef(x, y, rowvar=0)[0][1]


def cosine_similarity(x, y):
    num = float(np.sum(x.T * y))
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.5 + 0.5 * (num / norm)


def standard_score(data_mat, user_index, item_index, similarity_function):
    score_total = 0.0
    similarity_total = 0.0
    m, n = np.shape(data_mat)
    # calculate the similarity between the recommend item and other items
    for j in range(n):
        # this item need to be recommend
        if data_mat[user_index, j]== 0:
            continue
        # find the uses scored item_index and item_j
        item_user_link = np.nonzero(np.logical_and(data_mat[:, item_index].A > 0, data_mat[:, j].A > 0))[0]
        if len(item_user_link) == 0:
            continue
        similarity = similarity_function(data_mat[item_user_link, item_index], data_mat[item_user_link, j])
        score = similarity * data_mat[user_index, j]
        similarity_total += similarity
        score_total += score
    if similarity_total == 0:
        return 0
    return score_total / similarity_total


def get_singular_count(singular_array):
    square_array = np.square(singular_array)
    all = np.sum(square_array)
    total = 0.0
    n = len(singular_array)
    for i in range(n):
        total += square_array[i]
        if (total / all) > 0.9:
            return i + 1
    return n


def svd_score(data_mat, user_index, item_index, similarity_function):
    score_total = 0.0
    similarity_total = 0.0
    m, n = np.shape(data_mat)
    u, s, v_t = np.linalg.svd(data_mat)
    k = get_singular_count(s)
    s_k = np.eye(k) * s[:k]
    singular_mat = data_mat.T * u[:, :k] * np.linalg.inv(s_k)
    for j in range(n):
        if data_mat[user_index, j] == 0 or j == item_index:
            continue
        similarity = similarity_function(singular_mat[item_index, :].T, singular_mat[j, :].T)
        score = similarity * data_mat[user_index, j]
        similarity_total += similarity
        score_total += score
    if similarity_total == 0:
        return 0
    return score_total / similarity_total


def recommend(data_mat, user_index, similarity_function=cosine_similarity, score_function=standard_score):
    recommend_list = []
    m, n = np.shape(data_mat)
    for j in range(n):
        if data_mat[user_index, j] == 0:
            recommend_list.append([j, score_function(data_mat, user_index, j, similarity_function)])
    recommend_list.sort(key=lambda x: x[1], reverse=True)
    return recommend_list
