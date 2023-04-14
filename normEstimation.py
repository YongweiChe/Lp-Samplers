import numpy as np
import math
import pickle
from uniform import UniformRV

def pstableMedian(p, num_samples):
    u = np.random.uniform(low=-math.pi/2, high=math.pi/2, size=(num_samples))
    r = np.random.uniform(low=0, high=1, size=(num_samples))

    X = np.sin(p * u) / (np.cos(u))**(1 / p) * \
        (np.cos(u * (1 - p)) / (-np.log(r)))**((1 - p) * p)

    return np.median(np.abs(X))

# A = r x n matrix
# for fixed i (each row), is k-wise independent
# the seeds for each row i is 2-wise independent
class pstableMatrix:
    def __init__(self, p, r, n, k):
        self.p = p
        self.r = r
        self.n = n
        self.k = k
        
        self.A = [ [None, None] for _ in range(r) ]
        
        for i in range(r):
            self.A[i][0] = UniformRV(k, low=-math.pi/2, high=math.pi/2) # u
            self.A[i][1] = UniformRV(k, low=0, high=1) # r
        
    def shape(self):
        return (self.r, self.n)

    # 
    def get(self, i, j):
        u = self.A[i][0].sample(j)
        r = self.A[i][1].sample(j)
        
        X = math.sin(self.p * u) / (math.cos(u))**(1 / self.p) * \
        (math.cos(u * (1 - self.p)) / (-math.log(r)))**((1 - self.p) * self.p)
        
        return X
    

# Indyk's Algorithm
class LpNormSketch:
    # run initialization to estimate the median of the sketch
    def __init__(self, p, n, eps):
        C1 = 10
        C2 = 10
        
        self.p = p
        self.n = n
        self.eps = eps
        
        k = int(C1 * eps**(-p) * math.log(1 / eps)**(3 * p))
        r = int(C2 * 1/(eps**2))
        
        # A in R^{r x n}
        self.A = pstableMatrix(p=p, r=r, n=n, k=k)
        
        self.y = np.zeros((r))
        
        # TODO: Preprocessing: Compute the estimated median of the p-stable dist
        self.distMedian = pstableMedian(p, num_samples=5000000)

    # multiply new entry by every element in A, y = Ax
    # time complexity O(rk) because A.get requires UniformRV.sample which takes k time
    def update(self, i, delta): 
        # update = [ 0, 0, ... , delta, 0, ..., 0, 0 ]
        r, n = self.A.shape()

        for j in range(r):
            self.y[j] += self.A.get(j, i) * delta
    
    # return median of y divided by median(Dist)
    def getNorm(self):
        print(self.y)
        print(f'({np.median(self.y)} / {self.distMedian})')
        return np.median(self.y) / self.distMedian


def main():
    print("p-stable distribution test")
    A = pstableMatrix(1.5, 3, 10, 3)
    print(A.A)
    for i in range(3):
        for j in range(10):
            print(f'Point at ({i}, {j}): {A.get(i, j)}')
    
    print("Lp Norm Sketch")
    p = 1.5
    n = 100
    eps = 0.1
    sketch = LpNormSketch(p, n, eps)
    sketch.update(3, 10)
    sketch.update(10, 4)
    print(sketch.getNorm())

if __name__ == '__main__':
    main()