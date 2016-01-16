import numpy as np

if __name__ == '__main__':
    m1 = np.mat([1, 2, 3])
    m2 = np.mat([[1, 2, 3],[4, 5, 6],[1, 2, 3],[4, 5, 6]])
    m3 = np.ones(3)
    m4 = np.ones((1, 3))
    print(m1)
    print(m1.transpose())
    print(m1.T)
    print(m2)
    print(m2.transpose())
    print(m2.T)
    print(m3)
    print(m3.transpose())
    print(m3.T)
    # transpose depend on shape
    m3.shape = (1, 3)
    print(m3.transpose())
    print(m3.T)
    print(m4)
    print(m4.transpose())
    print(m4.T)
    print(m2)
    print(m2[1, :])
    print(m2[:, 1])
    all = [m1, m2, m3, m4]
    print(all)
    m5 = np.mat(m3)
    print(m5)
    print(m5.T)