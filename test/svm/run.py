import numpy as np

if __name__ == '__main__':

    m2 = np.mat([[1, 2, 3],[4, 5, 6],[1, 2, 3],[4, 5, 6]])
    print(m2)
    print(m2[:, 1] > 2)
    a = (m2[:, 1] > 2)
    print(a[:, 0])
    m1 = np.mat([[1, 2, 0], [0, 1, 2]])
    print(np.nonzero(m1))
    print(len(m1))
    print(len(m2))
    m3 = np.empty((0, 3))
    print(m3)
    print(np.shape(m3))

