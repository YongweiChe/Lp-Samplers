import numpy as np
import math
import heapq
from CountSketch import CountSketch
from normEstimation import LpNormSketch
from uniform import UniformRV

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
        
        self.k = k
        self.m = m
        
        # 1. for 0 < p < 2, p \neq 1, set k = 10 ceil(1/|p - 1|) and m = O(\eps^(-max(0, p-1)))
        # with large enough constant factor
        if p != 1:
            k = 10 * math.ceil(1/abs(p - 1))
            m = math.ceil(C2 * eps**(-max(0, p - 1)))
        
        # Set beta = eps^(1 - 1/p) and l = O(log(n))
        
        self.beta = eps**(1 - 1/p)
        l = math.ceil(C3 * math.log(n))
        
        # Select k-wise independent uniform scaling factors t_i \in [0, 1] for i \in [n]
        
        # Generating uniform scaling factors: 
        #   https://sites.math.rutgers.edu/~sk1233/courses/topics-S18/lec5.pdf
        
        self.t = UniformRV(k, n) # b_0 + b_1 * x + ... + b_d x^d are d+1-wise independent
        
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
        
        # 2. Maintain a linear sketch L(x) as needed for the Lp norm approximation of x
        # Use this algorithm: https://dl.acm.org/doi/pdf/10.1145/1147954.1147955
        # Will need to maintain p-stable distribution
        self.xLpSketch = LpNormSketch(p, n, eps=1/3) # No need for the bounds on these to be as tight as they are
        
        # 3. Maintain a linear sketch L'(z) as needed for the  L2 norm estimation of x
        # m = O(1/eps^2) and we need eps = 1/3 for the probability bound
        # l = 1 - delta
        
        C_l2 = 10
        self.xL2Sketch = CountSketch(9 * C_l2, 10 * C_l2, k) # use the getL2Norm method
        
    # Processing Stage
    def insert(self, i, delta):
        # 1. Use count-sketch with parameter m for the scaled vector z \in R^n with z_i = x_i/(t_i)^(1/p)
        t_i = self.t.sample(i)
        self.zCountSketch.update(i, delta / (t_i**(1/self.p)))
        
        # 2. Maintain a linear sketch L(x) as needed for the Lp norm approximation of x
        self.xLpSketch.update(i, delta) # !!DONE!!: FINISH THIS IMPLEMENTATION
        
        # 3. Maintain a linear sketch L'(z) as needed for the L2 norm estimation of z
        self.xL2Sketch.update(i, delta)
        
    # Recovery Stage
    def sample(self):
        # 1. Compute the output z* of the CountSketch and its best m-sparse approximation z_hat
        m_heap = []
        heapq.heapify(m_heap) # pop removes smallest item from heap
        
        # loop through the entire countsketch to get the top entries
        for i in range(self.n):
            val = self.zCountSketch.query(i)
            heapq.heappush(m_heap, (val, i)) # tuple is (magnitude, index) for tracking later on
            if len(m_heap) > self.m:
                heapq.heappop(m_heap)
        
        ### IMPORTANT: m_heap now store z_hat ###
        
        # 2. Based on L(x) (xLpSketch) compute a real r with \|x\|_p \leq r \leq 2\|x\|_p
        r = self.xLpSketch.getNorm() / (1 - 1/3) # see paper for proof
        
        # 3. Based on L'(z - z_hat) = L'(z) - L'(z_hat))
        
        # 3.1 intermediate step: create and make countsketch for z_hat (ACTUALLY NOT NEEDED), JUST ALTER THE ORIGINAL COUNTSKETCH WITH THE UPDATES
        for val, index in m_heap:
            self.zCountSketch.update(index, -val)
        
        s = self.zCountSketch.getL2norm()
        
        self.zCountSketch.update(index, val) # reset the values of the countsketch
        # compute a real s such that ||z - z_hat||_2 < s < ||z - z_hat||
        # TODO: perform transform of s to fit in 1 and 2 of the respective norm
        
        # 4. Find i with |z_i^*| maximal
        max_index = 0
        max_val = m_heap[0][0]
        for val, index in m_heap:
            if val > max_val:
                max_val = val
                max_index = index
        
        # 5. If s > \beta m^{1/2} * r or |z_i^*| < \eps^{-1\p} * r output FAIL
        # TODO: re-instate hypothesis testing
        if (s > self.beta * self.m**0.5 * r ) or (abs(max_val) < self.eps**(1/self.p) * r):
            return None # Failure condition
        
        # 6. Output i as the sample and z_i^* t_i^(1/p) as an approximation for x_i
        return max_index, max_val * self.t.sample(i)**(1/self.p)


def main():
    print("Initializing Approximate Lp Sampler...")
    num_trials = 500
    num_failures = 0
    
    frequencies = np.array([100, 200, 500, 300, 750])
    
    sample_counts = np.zeros(frequencies.shape)
    sample_estimates = np.zeros(frequencies.shape)
    
    # p, n, eps
    p = 1.5
    n = 100
    eps = 0.1
    
    for i in range(num_trials):
        if i % 10 == 0:
            print(f'trial: {i}')
        sampler = ApproximateLpSampler(p=p, n=n, eps=eps)
        for idx, freq in enumerate(frequencies):
            sampler.insert(idx, freq)
        
        result = sampler.sample()
        
        if result is None:
            num_failures += 1
        else:
            sampled_id, sampled_estimate = result
            sample_counts[sampled_id] += 1
            sample_estimates[sampled_id] += sampled_estimate
            
    sample_estimates = sample_estimates / sample_counts
    
    
    print(f'success rate: {1 - num_failures / num_trials}')
    print(f'true lp probabilities: {frequencies**p / np.sum(frequencies**p)}')
    print(f'sampled probabilities: {sample_counts / np.sum(sample_counts)}')
    print(f'sample estimates: {sample_estimates}')
    


if __name__ == '__main__':
    main()



