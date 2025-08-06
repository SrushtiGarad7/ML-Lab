import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw, geom

#Generate the three original samples 
np.random.seed(0)
gaussian_data   = np.random.normal(loc=5, scale=2, size=10000)
powerlaw_data   = powerlaw.rvs(a=0.3, size=10000, random_state=0)
geometric_data  = geom.rvs(p=0.005, size=10000, random_state=0)

#Boxplot of Originals 
plt.figure(figsize=(8, 5))
plt.boxplot([gaussian_data, powerlaw_data, geometric_data],
            labels=['Gaussian', 'PowerLaw', 'Geometric'])
plt.title('Original Distributions')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


#1) Max–Min Scaling (divide by max)
gaussian_max_scaled  = gaussian_data  / np.max(gaussian_data)
powerlaw_max_scaled  = powerlaw_data  / np.max(powerlaw_data)
geometric_max_scaled = geometric_data / np.max(geometric_data)

# Histogram comparison: Gaussian
plt.figure(figsize=(6, 4))
plt.hist(gaussian_data, bins=20, alpha=0.5, edgecolor='black', label='Original Gaussian')
plt.hist(gaussian_max_scaled, bins=20, alpha=0.5, edgecolor='black', label='Max–Scaled Gaussian')
plt.title('Max–Scale vs Original (Gaussian)')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram comparison: PowerLaw
plt.figure(figsize=(6, 4))
plt.hist(powerlaw_data, bins=20, alpha=0.5, edgecolor='black', label='Original PowerLaw')
plt.hist(powerlaw_max_scaled, bins=20, alpha=0.5, edgecolor='black', label='Max–Scaled PowerLaw')
plt.title('Max–Scale vs Original (PowerLaw)')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram comparison: Geometric
plt.figure(figsize=(6, 4))
plt.hist(geometric_data, bins=20, alpha=0.5, edgecolor='black', label='Original Geometric')
plt.hist(geometric_max_scaled, bins=20, alpha=0.5, edgecolor='black', label='Max–Scaled Geometric')
plt.title('Max–Scale vs Original (Geometric)')
plt.legend()
plt.tight_layout()
plt.show()

# Boxplot of Max–Scaled
plt.figure(figsize=(8, 5))
plt.boxplot([gaussian_max_scaled, powerlaw_max_scaled, geometric_max_scaled],
            labels=['Gaussian', 'PowerLaw', 'Geometric'])
plt.title('Boxplot: Max–Min Scaling')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


#2) Sum Scaling (divide by sum) 
gaussian_sum_scaled  = gaussian_data  / np.sum(gaussian_data)
powerlaw_sum_scaled  = powerlaw_data  / np.sum(powerlaw_data)
geometric_sum_scaled = geometric_data / np.sum(geometric_data)

# Histogram comparison: Gaussian
plt.figure(figsize=(6, 4))
plt.hist(gaussian_data, bins=20, alpha=0.5, edgecolor='black', label='Original Gaussian')
plt.hist(gaussian_sum_scaled, bins=20, alpha=0.5, edgecolor='black', label='Sum–Scaled Gaussian')
plt.title('Sum–Scale vs Original (Gaussian)')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram comparison: PowerLaw
plt.figure(figsize=(6, 4))
plt.hist(powerlaw_data, bins=20, alpha=0.5, edgecolor='black', label='Original PowerLaw')
plt.hist(powerlaw_sum_scaled, bins=20, alpha=0.5, edgecolor='black', label='Sum–Scaled PowerLaw')
plt.title('Sum–Scale vs Original (PowerLaw)')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram comparison: Geometric
plt.figure(figsize=(6, 4))
plt.hist(geometric_data, bins=20, alpha=0.5, edgecolor='black', label='Original Geometric')
plt.hist(geometric_sum_scaled, bins=20, alpha=0.5, edgecolor='black', label='Sum–Scaled Geometric')
plt.title('Sum–Scale vs Original (Geometric)')
plt.legend()
plt.tight_layout()
plt.show()

# Boxplot of Sum–Scaled
plt.figure(figsize=(8, 5))
plt.boxplot([gaussian_sum_scaled, powerlaw_sum_scaled, geometric_sum_scaled],
            labels=['Gaussian', 'PowerLaw', 'Geometric'])
plt.title('Boxplot: Sum Scaling')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


#3) Z–Score Standardization
gaussian_zscore  = (gaussian_data  - np.mean(gaussian_data))  / np.std(gaussian_data)
powerlaw_zscore  = (powerlaw_data  - np.mean(powerlaw_data))  / np.std(powerlaw_data)
geometric_zscore = (geometric_data - np.mean(geometric_data)) / np.std(geometric_data)

# Histogram comparison: Gaussian
plt.figure(figsize=(6, 4))
plt.hist(gaussian_data, bins=20, alpha=0.5, edgecolor='black', label='Original Gaussian')
plt.hist(gaussian_zscore, bins=20, alpha=0.5, edgecolor='black', label='Z–Scored Gaussian')
plt.title('Z–Score vs Original (Gaussian)')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram comparison: PowerLaw
plt.figure(figsize=(6, 4))
plt.hist(powerlaw_data, bins=20, alpha=0.5, edgecolor='black', label='Original PowerLaw')
plt.hist(powerlaw_zscore, bins=20, alpha=0.5, edgecolor='black', label='Z–Scored PowerLaw')
plt.title('Z–Score vs Original (PowerLaw)')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram comparison: Geometric
plt.figure(figsize=(6, 4))
plt.hist(geometric_data, bins=20, alpha=0.5, edgecolor='black', label='Original Geometric')
plt.hist(geometric_zscore, bins=20, alpha=0.5, edgecolor='black', label='Z–Scored Geometric')
plt.title('Z–Score vs Original (Geometric)')
plt.legend()
plt.tight_layout()
plt.show()

# Boxplot of Z–Scored
plt.figure(figsize=(8, 5))
plt.boxplot([gaussian_zscore, powerlaw_zscore, geometric_zscore],
            labels=['Gaussian', 'PowerLaw', 'Geometric'])
plt.title('Boxplot: Z–Score Standardization')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


#4) Percentile Ranking (0–1 uniform)
# Manually compute rank / (N-1)
sorted_gauss = np.argsort(gaussian_data)
ranks_gauss  = np.empty_like(sorted_gauss)
ranks_gauss[sorted_gauss] = np.arange(len(gaussian_data))
gaussian_percentile = ranks_gauss / (len(gaussian_data) - 1)

sorted_power = np.argsort(powerlaw_data)
ranks_power  = np.empty_like(sorted_power)
ranks_power[sorted_power] = np.arange(len(powerlaw_data))
powerlaw_percentile = ranks_power / (len(powerlaw_data) - 1)

sorted_geom = np.argsort(geometric_data)
ranks_geom  = np.empty_like(sorted_geom)
ranks_geom[sorted_geom] = np.arange(len(geometric_data))
geometric_percentile = ranks_geom / (len(geometric_data) - 1)

# Histogram comparison: Gaussian
plt.figure(figsize=(6, 4))
plt.hist(gaussian_data, bins=20, alpha=0.5, edgecolor='black', label='Original Gaussian')
plt.hist(gaussian_percentile, bins=20, alpha=0.5, edgecolor='black', label='Percentile Gaussian')
plt.title('Percentile vs Original (Gaussian)')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram comparison: PowerLaw
plt.figure(figsize=(6, 4))
plt.hist(powerlaw_data, bins=20, alpha=0.5, edgecolor='black', label='Original PowerLaw')
plt.hist(powerlaw_percentile, bins=20, alpha=0.5, edgecolor='black', label='Percentile PowerLaw')
plt.title('Percentile vs Original (PowerLaw)')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram comparison: Geometric
plt.figure(figsize=(6, 4))
plt.hist(geometric_data, bins=20, alpha=0.5, edgecolor='black', label='Original Geometric')
plt.hist(geometric_percentile, bins=20, alpha=0.5, edgecolor='black', label='Percentile Geometric')
plt.title('Percentile vs Original (Geometric)')
plt.legend()
plt.tight_layout()
plt.show()

# Boxplot of Percentiles
plt.figure(figsize=(8, 5))
plt.boxplot([gaussian_percentile, powerlaw_percentile, geometric_percentile],
            labels=['Gaussian', 'PowerLaw', 'Geometric'])
plt.title('Boxplot: Percentile Scaling')
plt.ylabel('Value')
plt.tight_layout()
plt.show()