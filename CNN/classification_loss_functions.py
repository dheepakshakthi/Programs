'''
cross entropy loss function:
L(y, f(x)) = -[y*log(f(x)) + (1-y)*log(1-f(x))]

hinge loss: 
L(y, f(x)) = max(0, 1-y*f(x))

focal loss:
fl(pt) = -(alpha_t *(1-pt)^gamma)*log(pt)

Kullback-Leibler divergence:
sum(probability of event happening under probability distribution p * log(probability of event happening under probability distribution p / probability of event happening under probability distribution q))

'''

import numpy as np
import scipy.stats as stats

true_values = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
predicted_values = [0.9, 0.1, 0.8, 0.7, 0.2, 0.6, 0.3, 0.4, 0.95, 0.85, 0.15, 0.75, 0.25, 0.55, 0.45, 0.65, 0.35, 0.5, 0.05, 0.55]
true_values = np.array(true_values)
predicted_values = np.array(predicted_values)


# cross entropy loss function
loss = -np.mean(true_values * np.log(predicted_values) + (1 - predicted_values) * np.log(1 - predicted_values))
print("cross entropy loss:", loss)

# hinge loss function
true_values_hinge = 2 * true_values - 1
predicted_scores_hinge = 2 * predicted_values - 1

hinge_losses = np.maximum(0, 1-(true_values+1e-10)*(predicted_values+1e-10))
hinge_loss = np.mean(hinge_losses)
print("hinge loss:", hinge_loss)

#focal loss function
gamma = 2.0
alpha = 0.25

p_t = np.where(true_values == 1, predicted_values, 1 - predicted_values)+1e-10
alpha_t = np.where(true_values == 1, alpha, 1 - alpha)
focal_losses = -(alpha_t * (1 - p_t) ** gamma) * np.log(p_t)
focal_loss = np.mean(focal_losses)
print("focal loss:", focal_loss)

# Kullback-Leibler divergence
p = predicted_values + 1e-10  
q = true_values + 1e-10

p = p / np.sum(p)
q = q / np.sum(q)
kld = stats.entropy(p, q)
print("Kullback-Leibler divergence:", kld)