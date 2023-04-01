import numpy as np
import math

class LpNormSketch:
    def __init__(self, p, n, eps):
        self.p = p
        self.n = n
        self.eps = eps
        
    def update(self, i, delta):
        return 0
    
    def getNorm(self):
        return 0


def main():
    print("Lp Norm Sketch")
    p = 1.5
    n = 100
    eps = 0.1
    sketch = LpNormSketch(p, n, eps)
    print(sketch.getNorm())

if __name__ == '__main__':
    main()