from lr.core import *


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
    # train the data
    lr = LogisticRegression()
    lr.add_train_features(feature_list, label_list)
    lr.train(TrainAlgorithm.GradientAscent)