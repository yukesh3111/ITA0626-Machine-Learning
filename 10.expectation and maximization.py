import numpy as np
from scipy.stats import norm

# Generate synthetic data from two Gaussian distributions
np.random.seed(0)
data = np.concatenate([np.random.normal(0, 1, 300), np.random.normal(5, 1, 700)])

# Initialize parameters
mu1, sigma1 = -1, 1
mu2, sigma2 = 1, 1
pi1, pi2 = 0.5, 0.5

# Define the number of iterations for the EM algorithm
num_iterations = 100

for iteration in range(num_iterations):
    # Expectation step (E-step)
    likelihood1 = norm.pdf(data, loc=mu1, scale=sigma1)
    likelihood2 = norm.pdf(data, loc=mu2, scale=sigma2)
    total_likelihood = pi1 * likelihood1 + pi2 * likelihood2
    posterior1 = (pi1 * likelihood1) / total_likelihood
    posterior2 = (pi2 * likelihood2) / total_likelihood

    # Maximization step (M-step)
    mu1 = np.sum(posterior1 * data) / np.sum(posterior1)
    mu2 = np.sum(posterior2 * data) / np.sum(posterior2)
    sigma1 = np.sqrt(np.sum(posterior1 * (data - mu1)**2) / np.sum(posterior1))
    sigma2 = np.sqrt(np.sum(posterior2 * (data - mu2)**2) / np.sum(posterior2))
    pi1 = np.mean(posterior1)
    pi2 = np.mean(posterior2)

# Print the final estimated parameters
print("Estimated Parameters:")
print(f"mu1: {mu1:.2f}, sigma1: {sigma1:.2f}, pi1: {pi1:.2f}")
print(f"mu2: {mu2:.2f}, sigma2: {sigma2:.2f}, pi2: {pi2:.2f}")
