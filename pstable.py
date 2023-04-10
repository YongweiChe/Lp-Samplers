import numpy as np
import math
import pickle

# generate 1 p-stable random variable
def pstable(p):
    u = np.random.uniform(low=-math.pi/2, high=math.pi/2)
    r = np.random.uniform(low=0, high=1)

    X = math.sin(p * u) / (math.cos(u))**(1 / p) * \
        (math.cos(u * (1 - p)) / (-math.log(r)))**((1 - p) * p)

# code to generate medians for p-stable distributions
def main():
    num_estimates = 201

    p_arr = np.linspace(0, 2, num_estimates)

    medians = {}

    for p in p_arr:
        num_samples = 5000000
        u = np.random.uniform(low=-math.pi/2, high=math.pi/2, size=(num_samples))
        r = np.random.uniform(low=0, high=1, size=(num_samples))

        X = np.sin(p * u) / (np.cos(u))**(1 / p) * \
            (np.cos(u * (1 - p)) / (-np.log(r)))**((1 - p) * p)

        medians[p] = np.median(np.abs(X))

    print(medians)
    medians_file = open('medians.pkl', 'wb')
    pickle.dump(medians, medians_file)
    medians_file.close()
    


if __name__ == '__main__':
    main()