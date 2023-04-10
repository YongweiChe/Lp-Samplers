import numpy as np
import math
from CountSketch import CountSketch
from normEstimation import LpNormSketch

# get t_i, which is the uniform scaling factor
def getUniformScalingFactor(b_kwise, i):
    x = i / len(b_kwise)
    
    x_poly = np.array([x**i for i in range(len(b_kwise))])
    
    return np.dot(x_poly, b_kwise)
    

# From 'Tight Bounds for Lp Samplers - Jowhari, Saglman, Tardos'
class ApproximateLpSampler: 
    
    # Initialization Stage
    def __init__(self, p, n, eps):
        
        # "LARGE ENOUGH CONSTANT FACTORS"
        C1, C2, C3 = [10, 10, 10]
        
        self.p = p
        self.n = n
        self.eps = eps
        
        # 2. for p = 1, set k = m = O(log(1/eps)) with large enough constant factor
        k = math.ceil(C1 * math.log(1/eps))
        m = math.ceil(k)
        
        # 1. for 0 < p < 2, p \neq 1, set k = 10 ceil(1/|p - 1|) and m = O(\eps^(-max(0, p-1)))
        # with large enough constant factor
        if p != 1:
            k = 10 * math.ceil(1/abs(p - 1))
            m = math.ceil(C2 * eps**(-max(0, p - 1)))
        
        # Set beta = eps^(1 - 1/p) and l = O(log(n))
        
        beta = eps**(1 - 1/p)
        l = math.ceil(C3 * math.log(n))
        
        # Select k-wise independent uniform scaling factors t_i \in [0, 1] for i \in [n]
        
        # Generating uniform scaling factors: 
        #   https://sites.math.rutgers.edu/~sk1233/courses/topics-S18/lec5.pdf
        
        self.b_kwise = np.random.uniform(0, 1, k) # b_0 + b_1 * x + ... + b_d x^d are d+1-wise independent
        
        # use getUniformScalingFactor(b_kwise, i) to get the scaling factor t_i
        
        # Select the appropriate random linear functions for the execution of the count-sketch
        # algorithm and L and L' for the norm estimations in the processing stage.
        
        # PROCESSING STAGE INITIALIZATION
        # 1. Use count-sketch with parameter m for the scaled vector z \in R^n with z_i = x_i / t_i^(1/p)
        
        # JST11 definition of count-sketch: 
        #   h_j: [n] -> [6m]
        #   j \in [l]: l = O(log(n))

        self.zCountSketch = CountSketch(6 * m, l, k)
        
        ### UNCLEAR WHAT THE SPACE CONSTRAINTS OF THESE SHOULD BE ###
        
        # TODO: 2. Maintain a linear sketch L(x) as needed for the Lp norm approximation of x
        # Use this algorithm: https://dl.acm.org/doi/pdf/10.1145/1147954.1147955
        # Will need to maintain p-stable distribution
        self.xLpSketch = LpNormSketch(p, n, eps)
        
        # 3. Maintain a linear sketch L'(z) as needed for the  L2 norm estimation of x
        self.xL2Sketch = CountSketch(6 * m, l, k) # use the getL2Norm method
        
    # Processing Stage
    def insert(self, i, delta):
        # 1. Use count-sketch with parameter m for the scaled vector z \in R^n with z_i = x_i/(t_i)^(1/p)
        t_i = getUniformScalingFactor(self.b_kwise, i)
        self.zCountSketch.update(i, delta / (t_i**(1/self.p)))
        
        # 2. Maintain a linear sketch L(x) as needed for the Lp norm approximation of x
        self.xLpSketch.update(i, delta) # TODO: FINISH THIS IMPLEMENTATION
        
        # 3. Maintain a linear sketch L'(z) as needed for the L2 norm estimation of z
        self.xL2Sketch.update(i, delta)
        
    # Recovery Stage
    def sample(self):
        print("sampling...")
        
        # 1. Compute the output z* of the CountSketch and its best m-sparse approximation z_hat
            # TODO: The entire output!!!???!?!?!?!?!??!
        
        # 2. Based on L(x) (xLpSketch) compute a real r with \|x\|_p \leq r \leq 2\|x\|_p
        r = self.xLpSketch.getNorm()
        
        # 3. Based on L'(z - z_hat) = L'(z) - L'(z_hat))
        
        # TODO: Write code to combine two L2 Sketches
        # TODO: initialize and fill CountSketch with m-sparse vector z_hat
        
        # 4. Find i with |z_i^*| maximal
        
        # TODO: Find maximal |z_i^*| by iterating through CountSketch??
        
        # 5. If s > \beta m^{1/2} * r or |z_i^*| < \eps^{-1\p} * r output FAIL
        
        # TODO: Basic conditional hypothesis testing
        
        # 6. Output i as the sample and z_i^* t_i^(1/p) as an approximation for x_i
        
        # TODO: return stuff
        
        


def main():
    print("Initializing Approximate Lp Sampler...")
    
    sampler = ApproximateLpSampler(1.5, 100, 0.1)
    


if __name__ == '__main__':
    main()



