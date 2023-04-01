import numpy as np
from hash import HashFamily

class CountMin:
    def __init__(self, l, w): # length, w is width
        self.l = l
        self.w = w
        self.S = np.zeros((l, w))
        self.hashes = HashFamily(k=2, m=w, num_funcs=l)

    def update(self, i):
        for j in range(self.l):
            self.S[j, self.hashes.hash(j, i)] += 1

    def query(self, i):
        estimate = self.S[0, self.hashes.hash(0, i)]
        for j in range(self.l):
            estimate = min(estimate, self.S[j, self.hashes.hash(j, i)])
        
        return estimate

def main():
    print("Initializing CountMin...")
    U = 1000
    cm = CountMin(10, 300)
    
    actual_counts = np.zeros(U)
    for i in range(1000):
        random_update = np.random.randint(U)
        actual_counts[random_update] += 1
        cm.update(random_update)

    error_count = 0
    for i in range(U):
        if actual_counts[i] != cm.query(i):
            error_count += 1
            print(f'Query Answer for {i}: {actual_counts[i]} vs estimated {cm.query(i)}')

    print(f'The error count is: {error_count}')
    
if __name__ == '__main__':
    main()
    
