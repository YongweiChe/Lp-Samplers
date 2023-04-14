import numpy as np
import matplotlib.pyplot as plt

# TODO: Verify correctness of this
# generates num_vars k-wise independent random variables
class UniformRV:
    def __init__(self, k, num_vars, low=0, high=1, debug=True):
        self.debug = debug
        self.uniforms = []
        if debug:
            self.uniforms = np.random.uniform(low, high, num_vars)
            # (f'debug uniforms: {self.uniforms}')
            
        
        self.k = k
        self.b_kwise = np.random.uniform(0, 1, k)
        self.scaling = np.random.uniform(5, 10)
        self.low = low
        self.high = high
        
    def sample(self, i):
        if self.debug:
            return self.uniforms[i]
            
        x = ((i + 1) * self.scaling) % 1 # all you need is a deterministic mapping

        x_poly = np.array([x**i for i in range(len(self.b_kwise))])

        u = np.dot(x_poly, self.b_kwise) % 1
        width = self.high - self.low
        return (u * width) + self.low
    

def main():
    t = UniformRV(5, 10)
    arr = []
    for i in range(25):
        print(t.sample(i))
        arr.append(t.sample(i))
        
    plt.plot(arr)
    plt.show()

if __name__ == '__main__':
    main()
