from lr.core import *
import matplotlib.pyplot as plt
import numpy as np


def paint_boundary(plot, weights_list, b_label, b_color, b_style):
    x_list = np.arange(-3.5, 3.5, 0.1)
    y_list = []
    for x in x_list:
        y = (0 - weights_list[0][0] - weights_list[1][0] * x) / weights_list[2][0]
        y_list.append(y)
    plt.plot(x_list, y_list, label=b_label, color=b_color, linestyle=b_style)


if __name__ == '__main__':
    feature_list = []
    label_list = []
    # read the text
    try:
        with open("testSet.txt", "r") as file:
            for each_line in file:
                line_array = each_line.strip().split()
                feature_list.append([1.0, float(line_array[0]), float(line_array[1])])
                label_list.append(int(line_array[2]))
    except IOError as io_error:
        print("Open File[testSet.txt] Error:" + str(io_error))
    # prepare train the data
    lr = LogisticRegression()
    lr.add_train_features(feature_list, label_list)
    # print features to graph
    plt_0_x_list = []
    plt_0_y_list = []
    plt_1_x_list = []
    plt_1_y_list = []
    for i in range(len(label_list)):
        if label_list[i] == 0:
            plt_0_x_list.append(feature_list[i][1])
            plt_0_y_list.append(feature_list[i][2])
        else:
            plt_1_x_list.append(feature_list[i][1])
            plt_1_y_list.append(feature_list[i][2])
    plt.scatter(plt_0_x_list, plt_0_y_list, c="red", s=20)
    plt.scatter(plt_1_x_list, plt_1_y_list, c="yellow", marker="s", s=20)
    plt.xlabel("X1")
    plt.ylabel("X2")
    # training with GradientAscent
    lr.train(TrainAlgorithm.GradientAscent)
    paint_boundary(plt, lr.get_weights().tolist(), "GradientAscent", "black", "solid")

    plt.legend()

    plt.savefig("result.png")
