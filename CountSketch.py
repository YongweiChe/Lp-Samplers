import numpy as np
from hash import HashFamily

# CountSketch according to JST11
class CountSketch:
    def __init__(self, l, w, k):
        self.l = l  # number of hash functions
        self.w = w  # width of the CountSketch
        self.S = np.zeros((l, w))  # linear sketch
        self.h = HashFamily(k, m=w, num_funcs=l)
        self.r = HashFamily(k, m=2, num_funcs=l) # returns 0 or 1, remember to transform

    def __str__(self):
        return str(self.S)
    
    def getSize(self):
        return (self.l, self.w)
    
    def update(self, i, delta):  # S[h(i)] <- S[h(i)] + delta * r(i)
        for j in range(self.l):
            # need int() cast because sometimes python makes 0 a float 0.0
            self.S[j, int(self.h.hash(j, i))] += (self.r.hash(j, i) * 2 - 1) * delta
    
    # point query of value at index i
    def query(self, i): 
        vals = []
        for j in range(self.l):
            # print(self.h.hash(j, i))
            vals.append((self.r.hash(j, i) * 2 - 1) * self.S[j, int(self.h.hash(j, i))])
        
        vals.sort()
        
        return vals[len(vals) // 2]
    
    # computes the l2 norm of the vector
    def getL2norm(self): # AMS '96 algorithm
        return np.median(np.sum(self.S**2, axis=1))


def computeLpNorm(vec, p):
    return np.sum(vec**p)**(1/p)

def main():
    U = 100
    M = 3
    cm = CountSketch(21, 25, 3)

    actual_counts = np.zeros(U)
    for i in range(1000):
        random_update = np.random.randint(U)
        random_delta = np.random.rand() * M  # - M / 2 
        actual_counts[random_update] += random_delta
        cm.update(random_update, random_delta)

    error_count = 0
    for i in range(U):
        if actual_counts[i] != cm.query(i):
            error_count += 1
            print(
                f'Query Answer for {i}: {actual_counts[i]} vs estimated {cm.query(i)}')

    print(f'The error count is: {error_count}')
    
    print(f'The actual l2 norm is: {np.sum(actual_counts**2)}')
    print(f'The estimated l2 norm is: {cm.getL2norm()}')


if __name__ == '__main__':
    main()