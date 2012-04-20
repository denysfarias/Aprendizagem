import numpy as np
import matplotlib.pyplot as plt
import random

def generate_dataset(c_mean, c_cov, c_samples, fname, r = 100, plot = False):
    '''
    Generates instances based on:
        - c_mean: the mean of the axis
        - c_cov:  the covariance matrix of the axis
        - c_samples: the number of samples you need
        - r: how many times you want to run the algorithm
    '''
    min_diff = -1
    x_c, y_c = np.asarray([]), np.asarray([])
    for i in range(r):
        x, y = np.random.multivariate_normal(c_mean, c_cov, c_samples).T
        tc = np.sum(np.abs(c_cov - np.cov(x,y)))
        tm = np.sum(np.abs(c_mean - np.asarray([np.mean(x), np.mean(y)])))
        if min_diff == -1 or tc + tm < min_diff:
            x_c, y_c = x, y
            min_diff = tc + tm
        
    if plot:
        print '------------------'
        print fname
        print 'expected cov' 
        print c_cov
        print 'actual cov'
        print np.cov(x,y)
        print 'expected mean'
        print c_mean
        print 'actual mean'
        print [np.mean(x), np.mean(y)]
        print '------------------'

        plt.plot(x, y, 'x')
        plt.axis('equal')
        plt.savefig(fname + '.png')
        plt.clf()
    
    return x, y
        
def get_train_and_test(x, y, train_proportion = 0.7):
    m = np.asmatrix([np.asarray(x), np.asarray(y)]).T
    random.shuffle(m)
    train = m[:len(m) * train_proportion]
    test  = m[len(m) * train_proportion:]
    return train, test

def get_dataset(c_mean, c_cov, c_samples, fname, r = 100, plot = False, train_proportion = 0.7):
    x, y = generate_dataset(c_mean, c_cov * c_cov, c_samples, fname, r, plot = plot)
    train, test = get_train_and_test(x, y, train_proportion)

    f = open(fname + '-tra.txt', 'w')
    for t in np.asarray(train):
        f.write(', '.join(str(i) for i in t) + '\n')
    f.close()
    
    f = open(fname + '-tst.txt', 'w')
    for t in np.asarray(test):
        f.write(', '.join(str(i) for i in t) + '\n')
    f.close()

    return train, test


if __name__ == '__main__':
    c_cov_factor = np.asarray([[1,0], [0,1]])
    
    c_mean = [0,0]
    c_cov = np.asarray([[2, 1.7],[1.7,1]])
    c_samples = 150

    get_dataset(c_mean, c_cov * (c_cov * c_cov_factor), c_samples, 'Classe1', r = 1000, plot = True)

    c_mean = [0,3]
    c_cov = np.asarray([[0.5, 0], [0,0.5]])
    c_samples = 100

    get_dataset(c_mean, c_cov * (c_cov * c_cov_factor), c_samples, 'Classe2', r = 1000, plot = True)

    c_mean = [4, 3]
    c_cov = np.asarray([[2, -1.7], [-1.7, 1]])
    c_samples = 50

    get_dataset(c_mean, c_cov * (c_cov * c_cov_factor), c_samples, 'Classe3', r = 1000, plot = True)






