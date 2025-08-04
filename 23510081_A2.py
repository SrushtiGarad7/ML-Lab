import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw, geom, rankdata
from sklearn.preprocessing import QuantileTransformer

np.random.seed(0)
gaussian = np.random.normal(loc=5, scale=2, size=10000)
powerlaw_var = powerlaw.rvs(a=0.3, size=10000)
geometric = geom.rvs(p=0.005, size=10000)

all_data = [gaussian, powerlaw_var, geometric]
labels = ['Gaussian', 'PowerLaw', 'Geometric']

# Original boxplot
plt.figure()
plt.boxplot(all_data, labels=labels, patch_artist=True,
            boxprops=dict(facecolor="#AED6F1"))
plt.title('Original Distributions')
plt.ylabel('Value')
plt.show()

# Transformation methods
methods = {}
methods['Max Scale'] = [x / x.max() for x in all_data]
methods['Sum Scale'] = [x / x.sum() for x in all_data]
methods['Z-Score'] = [(x - x.mean()) / x.std() for x in all_data]
methods['Percentile'] = [(rankdata(x) - 1) / (len(x) - 1) for x in all_data]

medians = [np.median(x) for x in all_data]
target_median = np.mean(medians)
methods['Equal Median'] = [x * (target_median / m) for x, m in zip(all_data, medians)]

qt = QuantileTransformer(output_distribution='normal', random_state=42)
qdata = qt.fit_transform(np.column_stack(all_data)).T
methods['QuantileNorm'] = [qdata[i] for i in range(len(all_data))]

colors = ['#FF9999', '#99FF99', '#9999FF', '#FFCC99', '#CC99FF', '#66CCCC']

for idx, (name, transformed) in enumerate(methods.items()):
    color = colors[idx % len(colors)]
    for original, new in zip(all_data, transformed):
        plt.hist(original, bins=20, alpha=0.5, edgecolor='black', color="#B0BEC5", label='Original')
        plt.hist(new, bins=20, alpha=0.5, edgecolor='black', color=color, label=name)
        plt.title(f'{name} vs Original')
        plt.legend()
        plt.show()
    plt.figure()
    plt.boxplot(transformed, labels=labels, patch_artist=True,
                boxprops=dict(facecolor=color))
    plt.title(f'Boxplot: {name}')
    plt.ylabel('Value')
    plt.show()
