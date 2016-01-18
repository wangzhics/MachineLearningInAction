from linear.local_weight import LocallyWeighted
from linear.ordinary import LeastSquares
import matplotlib.pyplot as plt


def read_data():
    x_rows = []
    y_rows = []
    try:
        with open("ex0.txt", "r") as file:
            data_row = []
            tmp_str = ''
            for each_line in file:
                line_array = each_line.strip().split("\t")
                x_rows.append([float(line_array[0]), float(line_array[1])])
                y_rows.append(float(line_array[2]))
        return x_rows, y_rows
    except IOError as io_error:
        print(str(io_error))

if __name__ == '__main__':
    x_arrays, y_array = read_data()
    # sort x_arrays by x2
    xy_array = []
    for i in range(len(x_arrays)):
        xy_array.append([x_arrays[i], y_array[i]])
    xy_array.sort(key=lambda x: x[0][1])
    # rebuild x_arrays, y_array
    x_arrays = []
    y_array = []
    for xy in xy_array:
        x_arrays.append(xy[0])
        y_array.append(xy[1])
    # regress
    ols = LocallyWeighted(x_arrays, y_array)
    x_array = []
    y1_array = []
    y2_array = []
    y1 = 0
    y2 = 0
    for i in range(len(x_arrays)):
        y1 = ols.regress(x_arrays[i], 1.0)
        y2 = ols.regress(x_arrays[i], 0.01)
        x_array.append(x_arrays[i][1])
        y1_array.append(y1)
        y2_array.append(y2)
    # paint
    plt.ylim((0.0, 1.2))
    plt.ylim((3.0, 5.2))
    plt.scatter(x_array, y_array, alpha=0.5, s=10)
    plt.plot(x_array, y1_array, label="K=1.0", color="green", linestyle="solid")
    plt.plot(x_array, y2_array, label="K=0.01", color="black", linestyle="dashed")
    # StandardRegress
    y0_array = []
    ordinary = LeastSquares(x_arrays, y_array)
    ordinary.train()
    for i in range(len(x_arrays)):
        y0_array.append(ordinary.regress(x_arrays[i]))
    plt.plot(x_array, y0_array, label="ols", color="blue", linestyle="dotted")
    plt.legend()
    plt.savefig("lwlr.png")
