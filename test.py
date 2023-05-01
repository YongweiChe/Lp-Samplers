import numpy as np
from lpSampler import ApproximateLpSampler


# tests I want to run over Lp Samplers:
# uniform random sampling error
# sparse vector error
# some high, some medium, some low error

def sparseAccuracyTest(p, n, num_trials, eps, num_sparse, window=5000):
    frequencies = np.zeros((n))
    
    for i in range(num_sparse):
        frequencies[np.random.randint(n)] = np.random.uniform(0, window)
        
    print(frequencies)
        
    num_failures = 0

    sample_counts = np.zeros(frequencies.shape)
    sample_estimates = np.zeros(frequencies.shape)

    sampler = ApproximateLpSampler(p, n, eps)
    sampler.getSize()

    for i in range(num_trials):
        print(f'trial: {i}')
        sampler = ApproximateLpSampler(p, n, eps)
        for idx, freq in enumerate(frequencies):
            # print(f'update {idx}')
            sampler.insert(idx, freq)

        # print(f'sampling')
        result = sampler.sample()

        if result is None:
            num_failures += 1
        else:
            sampled_id, sampled_estimate = result
            sample_counts[sampled_id] += 1
            sample_estimates[sampled_id] += sampled_estimate

    true = frequencies**p / np.sum(frequencies**p)
    sampled = sample_counts / np.sum(sample_counts)
    print(f'success rate: {1 - num_failures / num_trials}')
    print(f'true lp probabilities: {true}')
    print(f'sampled probabilities: {sampled}')

    sampler.getSize()

    print(f'correctness checking: 1 = {np.sum(sampled)}')
    print(f'correctness checking: 1 = {np.sum(true)}')

    # baseline against random sample:
    rand_guessing = np.random.rand(n)
    # to make it a probability distribution
    rand_guessing /= np.sum(rand_guessing)

    print(
        f'MSE against totally random vector: {np.sum((true - rand_guessing)**2)}')
    print(f'MSE: {np.sum((true - sampled)**2)}')
    print(f'Total size of vector: {np.sum(true**2)}')

# heheh
def accuracyTest(p, n, num_trials, eps):
    print('test')
    
    frequencies = np.random.rand(n) * 10
    print(f'freq sum: {np.sum(frequencies)}')
    num_failures = 0
    
    sample_counts = np.zeros(frequencies.shape)
    sample_estimates = np.zeros(frequencies.shape)
    
    sampler = ApproximateLpSampler(p, n, eps)
    sampler.getSize()
    
    for i in range(num_trials):
        print(f'trial: {i}')
        sampler = ApproximateLpSampler(p, n, eps)
        for idx, freq in enumerate(frequencies):
            # print(f'update {idx}')
            sampler.insert(idx, freq)
        
        # print(f'sampling')
        result = sampler.sample()

        if result is None:
            num_failures += 1
        else:
            sampled_id, sampled_estimate = result
            sample_counts[sampled_id] += 1
            sample_estimates[sampled_id] += sampled_estimate
    
    true = frequencies**p / np.sum(frequencies**p)
    sampled = sample_counts / np.sum(sample_counts)
    print(f'success rate: {1 - num_failures / num_trials}')
    print(f'true lp probabilities: {true}')
    print(f'sampled probabilities: {sampled}')
    
    sampler.getSize()
    
    print(f'correctness checking: 1 = {np.sum(sampled)}')
    print(f'correctness checking: 1 = {np.sum(true)}')

    # baseline against random sample:
    rand_guessing = np.random.rand(n)
    rand_guessing /= np.sum(rand_guessing) # to make it a probability distribution
    
    print(f'MSE against totally random vector: {np.sum((true - rand_guessing)**2)}')
    print(f'MSE of sampled output: {np.sum((true - sampled)**2)}')
    print(f'Total size of vector: {np.sum(true**2)}')

def main():
    # accuracyTest(1.5, 500, 1000, 0.5)
    sparseAccuracyTest(0.5, 500, 1000, 0.5, num_sparse=20, window=100)

if __name__ == '__main__':
    main()