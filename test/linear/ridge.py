from linear.ridge import RidgeRegress
import matplotlib.pyplot as plt


def read_data():
    x_rows = []
    y_rows = []
    try:
        with open("abalone.txt", "r") as file:
            data_row = []
            tmp_str = ''
            for each_line in file:
                line_array = each_line.strip().split("\t")
                x_row = []
                for i in range(8):
                    x_row.append(float(line_array[i]))
                x_rows.append(x_row)
                y_rows.append(float(line_array[8]))
        return x_rows, y_rows
    except IOError as io_error:
        print(str(io_error))

if __name__ == '__main__':
    x_arrays, y_array = read_data()

    # RidgeRegress
    ridge = RidgeRegress(x_arrays, y_array)
    axis_x_list = [i for i in range(-10, 20)]
    axis_y_lists = []
    wights_list = []
    # train and get weights
    for i in axis_x_list:
        wights_list.append(ridge.calc_ridge_weight(i))
    # build y axis values
    for i in range(8):
        axis_y_list = []
        for j in range(30):
            axis_y_list.append(wights_list[j][i])
        axis_y_lists.append(axis_y_list)
    # paint
    for i in range(8):
        plt.plot(axis_x_list, axis_y_lists[i])

    plt.savefig("ridge.png")
