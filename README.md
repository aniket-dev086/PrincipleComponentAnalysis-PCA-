# Principal Component Analysis (PCA)

## Description
This project demonstrates Principal Component Analysis (PCA) using NumPy and Matplotlib. It generates a 2D dataset, calculates the covariance matrix, finds eigenvalues and eigenvectors, and reduces the data from two dimensions to one while preserving maximum variance.

## Objectives
- Generate a 2D sample dataset.
- Perform mean centering.
- Calculate the covariance matrix.
- Compute and sort eigenvalues and eigenvectors.
- Reduce data from 2D to 1D using the first principal component.
- Visualize principal components and dimensionality reduction.

## Libraries Used
- **NumPy:** Data generation, matrix operations, covariance, and eigenvalue calculations.
- **Matplotlib:** Visualizing the original data, principal components, and projected data.

## Implementation Steps
1. Generate correlated 2D sample data.
2. Center the data by subtracting its mean.
3. Calculate the covariance matrix.
4. Compute and sort eigenvalues and eigenvectors.
5. Project the data onto the first principal component.
6. Plot the principal components and reduced data.

## Results
The project displays the original centered dataset, principal component directions, explained variance of the first principal component, and the 1D projection of the original 2D data.

## How to Run
1. Open the notebook in Google Colab or Jupyter Notebook.
2. Install NumPy and Matplotlib if required.
3. Run all cells to view the calculations and plots.

## Conclusion
This project demonstrates how PCA reduces data dimensionality while retaining the direction of maximum variance, making data easier to analyze and visualize.
