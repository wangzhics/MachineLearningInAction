# coding=utf-8
# 当有中文注释时，必须加上文件编码信息，否则在Window下调试会出错
from pandas import DataFrame
from math import log2


def calc_shannon_entropy(labels_array):
    """
    calculate the shannon entropy of a label array
    :param labels_array: the label array
    :return: the shannon entropy
    """
    labels_length = len(labels_array)
    labels_dict = {}
    for label in labels_array:
        if label not in labels_dict:
            labels_dict[label] = 0
        labels_dict[label] += 1
    entropy = 0
    for label, label_count in labels_dict.items():
        prob = float(label_count) / labels_length
        entropy -= prob * log2(prob)
    return entropy


def split_by_feature(parent_frame, feature):
    """
    split a data frame by a feature
    :param parent_frame: data frame
    :param feature: the feature name
    :return: sub frame
    """
    frame_dict = {}
    # count and split the feature group
    parent_columns = parent_frame.columns
    for parent_row in parent_frame.iterrows():
        # get feature value
        f_value = parent_row[1][feature]
        if f_value not in frame_dict:
            frame_dict[f_value] = DataFrame([], columns=parent_frame.columns)
        frame_dict[f_value] = frame_dict[f_value].append(parent_row[1], ignore_index=True)
    # remove the useless feature
    for f, frame in frame_dict.items():
        frame.pop(feature)
    return frame_dict


def get_best_feature(data_frame):
    # get result column
    frame_columns = data_frame.columns
    feature_count = len(frame_columns) - 1
    result_title = frame_columns[feature_count]
    base_entropy = calc_shannon_entropy(data_frame[result_title])
    # start to traversal all feature to find best feature
    base_feature = ''
    base_info_gain = 0.0
    parent_frame_length = len(data_frame)
    for i in range(feature_count):
        # calc the entropy if split by this feature
        feature_title = frame_columns[i]
        sub_frames = split_by_feature(data_frame, feature_title)
        sub_entropy = 0.0
        for feature_value, sub_frame in sub_frames.items():
            # print("feature[%s-%d]'s info gain is %s" % (feature_title, feature_value, str(sub_frame)))
            prob = len(sub_frame) / float(parent_frame_length)
            sub_entropy += prob * calc_shannon_entropy(sub_frame[result_title])
        info_gain = base_entropy - sub_entropy
        # print("feature[%s]'s info gain is %f" % (feature_title, info_gain))
        if info_gain > base_info_gain:
            base_feature = feature_title
            base_info_gain = info_gain
    return base_feature

