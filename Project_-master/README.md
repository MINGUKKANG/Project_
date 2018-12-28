
```python
<그래프 그리는데 필요한 모듈을 임포트한다.>

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


<합성법을 사용하여 베타-이항 분포에서 값을 샘플링한다.>

# 500개의 표본을 추출한다. from beta(3,5)
sample_beta = np.random.beta(3,5, 500)

u_holder = []
for i in range(500): 
    u_true = np.random.binomial(20, sample_beta[i]) # 이항분포에서 u를 샘플링한다.
    u_holder.append(u_true) # u값을 저장한다.

<깁스샘플러를 이용한 샘플링을 한다.>

### 깁스샘플러 함수, 함수의 결과 각각 1개의 u ~ f(u), v~ f(v)가 나온다.
def gibbs_sampler(initial_value, iteration):
    v_i = initial_value
    for i in range(iteration):
        u_i = np.random.binomial(20, v_i)
        v_i = np.random.beta(3 + u_i, 20 - u_i + 5)

    return u_i,v_i

u_gibbs_holder = []
v_gibbs_holder = []

for j in range(500):
    u_G, v_G = gibbs_sampler(np.random.uniform(0,1), iteration = 30) # 깁스 샘플링을 실시한다.
    u_gibbs_holder.append(u_G) # u_gibbs 값을 저장한다.
    v_gibbs_holder.append(v_G) # v_gibbs 값을 저장한다.

<식 5.101을 이용한 확률분포의 계산>

probability = [] # f(u)의 값을 넣기 위한 저장소
u_x = [] # u_x는 최종적으로 [0,1,2,3,4,...20]이 될 것이다.
for k in range(21): # k = 0~20 
    c_value = comb(20,k, exact = True) # 조합 값을 계산한다.
    u = 0
    for m in range(500):
        # 5.101식을 전개하여 개산한다.
        u += c_value * (v_gibbs_holder[m]**k) * ((1-v_gibbs_holder[m])**(20 -k))*(1/500)
    probability.append(u) # f(u = k)의 값을 저장한다. k = 0 ~ 20
    u_x.append(k) # k의 값을 저장한다.
    
<히스토그램 및 확률분포 그리기>

# histogram for Beta-Binomial distribution(picture 5-2)
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist([u_holder, u_gibbs_holder], color = ["r", "b"], bins = 21, label = ["S_Real", "S_Gibbs_sampler"])
ax.grid(True)
ax.legend(loc=1)
ax.set_xlabel("u ~ f(u)")
ax.set_ylabel("Frequency")
ax.set_title("Beta-Binomial histogram")
plt.show()

# Distribution for Beta-Binomial distribution(picture 5-3)
u_holder, bins = np.histogram(u_holder,21) # 구간을 0 ~20까지 나눈다.
u_holder = np.divide(u_holder, 500) # 확률 분포의 값을 계산한다.
u_x_0 = np.subtract(u_x ,0.4) # 그래프를 겹치지 않게 그리기 위한 Trick
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.bar(u_x_0, u_holder, width = 0.4, color = "r", label = "Real_dist")
ax1.bar(u_x, probability, width = 0.4, color = "b", label = "Approximation_dist")
ax1.set_xlim(-1,21)
ax1.legend(loc=1)
ax1.set_xlabel("u ~ f(u)")
ax1.set_ylabel("Probability")
ax1.set_title("Beta-Binomial distribution")
plt.show()
```

```python
<문제 1번>

# 프로그래밍을 위한 모듈을 임포트 한다.
import numpy as np
import matplotlib.pyplot as plt

# 시스템 신뢰도를 계산하기 위한 함수를 정의한다.
def Reliability(p):
    R = 2*(p**5) -5*(p**4) + 2*(p**3) +2*(p**2) # **는 제곱연산을 의미한다.
    return R

p_list = np.linspace(0,1,1000) # 0과 1사이를 1000등분하여 값을 생성한 후 list에 저장한다. 구성요소의 신뢰도 값을 의미함.
R_holder = [] # 시스템의 신뢰도를 저장하기 위한 홀더를 정의해준다.

for p in p_list:
    R_holder.append(Reliability(p)) # 구성요소의 신뢰도 값을 넣어서 시스템의 신뢰도를 계산한 후, 위의 홀더에 저장한다.

# X축을 구성요소의 신뢰도 값, y축을 시스템의 신뢰도 값으로 하여 그래프를 작성한다.
plt.plot(p_list, R_holder)
plt.xlabel("component reliability")
plt.ylabel("system reliability")
plt.title("HW_1")
plt.show()
```
