'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('Salary_dataset.csv')

ye_mean = df['YearsExperience'].mean()
sal_mean = df['Salary'].mean()

cov_matrix = df[['YearsExperience', 'Salary']].cov()
print("Covariance matrix:\n", cov_matrix)

eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eig_vals)
print("Eigenvectors:\n", eig_vecs)

des = eig_vals.argsort()[::-1]

eigenvalues = eig_vals[des]

eigenvectors = eig_vecs[:,des]

explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
print("explained variance:", explained_var)

n_components = np.argmax(explained_var >= 0.50) + 1
print("components", n_components)

u = eigenvectors[:,:n_components]
pca_component = pd.DataFrame(u, index = df.index, columns = ['PC1','PC2'])

print("PCA components:\n", pca_component)'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Salary_dataset.csv")

x = df.values

X_meaned = x - np.mean(x, axis=0)
cov_matrix = np.cov(X_meaned, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

n_components = 2
eigenvectors_subset = eigenvectors[:, :n_components]
X_reduced = np.dot(X_meaned, eigenvectors_subset)

print("PC1", X_reduced[:, 0])
print("PC2", X_reduced[:, 1])

plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()