import numpy as np
import matplotlib.pyplot as plt


# hashes from [n] -> [m]
class HashFamily:
    def __init__(self, k, m, num_funcs, large_prime=(2**61 - 1)):
        self.k = k
        self.m = m
        self.large_prime = large_prime
        self.num_funcs = num_funcs
        
        # note, this may go wrong if there are collisions. 
        # The probability of this is incredibly small.
        self.hashes = np.random.randint(large_prime, size=(num_funcs, k))
        
    
    def hash(self, i, x): # x is the number in [n] to be hashed i is the hash h_i function
        a_i = self.hashes[i]
        x = np.array([x**i for i in range(len(a_i))])
        res = np.dot(a_i, x) % self.large_prime % self.m
        
        return res
        

def main():
    print("hello, hash")
    hf = HashFamily(k=2, m=100, num_funcs=1000)
    
    data = []
    for i in range(hf.num_funcs):
        data.append(hf.hash(45, i))
    
    print(hf.hash(45, 2))
    plt.hist(data)
    plt.show()
    

if __name__ == '__main__':
    main()