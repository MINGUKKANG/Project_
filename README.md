
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def gibbs_sampler(initial_value, iteration):
    v_i = initial_value
    for i in range(iteration):
        u_i = np.random.binomial(20, v_i)
        v_i = np.random.beta(3 + u_i, 20 - u_i + 5)

    return u_i,v_i
```
