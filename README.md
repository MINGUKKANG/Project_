import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def gibbs_sampler(initial_value, iteration):
    v_i = initial_value
    for i in range(iteration):
        u_i = np.random.binomial(20, v_i)
        v_i = np.random.beta(3 + u_i, 20 - u_i + 5)

    return u_i,v_i

# 500 samples are extracted from beta(3,5)
sample_beta = np.random.beta(3,5, 500)

u_holder = []
for i in range(500): # 500 is the number of iteration.
    u_true = np.random.binomial(20, sample_beta[i]) # sampling from binomial(20, sample_beta[i]), i = 1,2...500
    u_holder.append(u_true)



u_gibbs_holder = []
v_gibbs_holder = []

for j in range(500):
    u_G, v_G = gibbs_sampler(np.random.uniform(0,1), iteration = 30)
    u_gibbs_holder.append(u_G)
    v_gibbs_holder.append(v_G)


probability = []
u_x = []
for k in range(20):
    c_value = comb(20,k, exact = True)
    u = 0
    for m in range(500):
        u += c_value * (v_gibbs_holder[m]**k) * ((1-v_gibbs_holder[m])**(20 -k))*(1/500)
    probability.append(u)
    u_x.append(k)


# histogram for Beta-Binomial distribution(picture 5-2)
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist([u_holder, u_gibbs_holder], color = ["r", "b"], bins = 20, label = ["S_Real_dist", "S_Gibbs_sampler"])
ax.grid(True)
ax.legend(loc=1)
ax.set_xlabel("u ~ f(u)")
ax.set_ylabel("Frequency")
ax.set_title("Beta-Binomial distribution")
plt.show()

u_holder, bins = np.histogram(u_holder,20)
u_holder = np.divide(u_holder, 500)
u_x_0 = np.subtract(u_x ,0.4)
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.bar(u_x_0, u_holder, width = 0.4, color = "r")
ax1.bar(u_x, probability, width = 0.4, color = "b")
plt.show()
'''
ax1.hist([u_holder, u_monte_holder], normed = True ,color = ["r", "b"], bins = 20, label = ["S_Real_dist", "S_monte_sampler"])
ax1.grid(True)
ax1.legend(loc=1)
ax1.set_xlabel("u ~ f(u)")
ax1.set_ylabel("Frequency")
ax1.set_title("Beta-Binomial distribution")
plt.show()
'''
