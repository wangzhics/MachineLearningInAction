from linear.stepwise import StepwiseRegress
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
    stepwise = StepwiseRegress(x_arrays, y_array)
    weight_arrays = stepwise.train(0.005, 1000)
    # get axis_x_list
    axis_x_list = [i for i in range(1000)]
    # get axis_y_lists
    axis_y_lists = []
    for i in range(8):
        axis_y_lists.append([])
    for i in axis_x_list:
        for j in range(8):
            axis_y_lists[j].append(weight_arrays[i][j])
    for i in range(8):
        plt.plot(axis_x_list, axis_y_lists[i])

    plt.savefig("stepwise.png")
