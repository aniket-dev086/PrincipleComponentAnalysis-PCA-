import numpy as np
import matplotlib.pyplot as plt

# 1. Generate 2D Sample Data
np.random.seed(42)
x = np.round(np.random.normal(0, 1, 100), 2)
y = np.round(1.5 * x + np.random.normal(0, 0.5, 100), 2)
X = np.vstack((x, y)).T

print("--- Step 1: Generate 2D Sample Data ---")
print("Original Dataset Shape:", X.shape)
print("First 5 rows of original data X:\n", X[:5])

# 2. Mean Centering
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

print("\n--- Step 2: Mean Centering ---")
print("Calculated Mean (X, Y):", X_mean)
print("First 5 rows of centered data X_centered:\n", X_centered[:5])

# 3. Covariance Matrix
cov_matrix = np.cov(X_centered, rowvar=False)

print("\n--- Step 3: Covariance Matrix ---")
print("Covariance Matrix:\n", cov_matrix)

# 4. Eigenvalues & Eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort in descending order of eigenvalues
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

print("\n--- Step 4: Eigenvalues & Eigenvectors ---")
print("Sorted Eigenvalues (Explained Variance):\n", eigenvalues)
print("Sorted Eigenvectors (Principal Components):\n", eigenvectors)
print(f"PC1 captures {eigenvalues[0]/sum(eigenvalues)*100:.2f}% of the total variance.")

# 5. Project data to 1D (Dimensionality Reduction)
first_pc = eigenvectors[:, 0]
X_reduced = np.dot(X_centered, first_pc)

# Project 1D back to 2D space for visualization
X_projected = np.outer(X_reduced, first_pc)

print("\n--- Step 5: Projection to 1D ---")
print("Reduced 1D Data Shape:", X_reduced.shape)
print("First 5 reduced values:\n", X_reduced[:5])

# 6. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left Plot: Original centered data and Eigenvectors
ax1.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6, color='blue', label='Centered Data')
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    scale = np.sqrt(val) * 2  # scale for visualization
    ax1.plot([0, vec[0] * scale], [0, vec[1] * scale],
             color='red' if i == 0 else 'green', linewidth=3,
             label=f'PC{i+1} (Var: {val:.2f})')
ax1.set_title('Original Centered Data & Principal Components')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.axis('equal')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Right Plot: Projected 1D Data
ax2.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.3, color='blue', label='Original Centered Data')
ax2.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.8, color='red', label='Projected 1D Data')
ax2.plot([-first_pc[0]*3, first_pc[0]*3], [-first_pc[1]*3, first_pc[1]*3], color='black', linestyle='--', alpha=0.5, label='PC1 Axis')
ax2.set_title('PCA Dimension Reduction to 1D')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()
