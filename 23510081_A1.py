import numpy as np
import matplotlib.pyplot as plt

def run_trials(array_size: int, trials: int) -> list[int]:
    results = []
    for _ in range(trials):
        arr = np.random.normal(loc=0, scale=1, size=array_size)
        cur_min = float('inf')
        updates = 0
        for x in arr:
            if x < cur_min:
                cur_min = x
                updates += 1
        results.append(updates)
    return results

sizes = [
    10, 20, 50, 100, 200, 500, 1_000, 2_000,
    5_000, 10_000, 20_000, 50_000, 100_000,
    200_000, 500_000, 1_000_000
]
trials = 100
means = []
medians = []
geomeans = []

for N in sizes:
    counts = run_trials(N, trials)
    mean = np.mean(counts)
    median = np.median(counts)
    geomean = np.exp(np.mean(np.log(counts)))

    means.append(mean)
    medians.append(median)
    geomeans.append(geomean)

    print(f"N={N} → mean updates = {mean:.2f}, median updates = {median:.2f}, geometric mean updates = {geomean:.2f}")

    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=range(min(counts), max(counts) + 2), edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of updates for N={N}')
    plt.xlabel('Number of updates')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(7, 5))
plt.loglog(sizes, means, marker='o', label='Mean')
plt.loglog(sizes, medians, marker='s', label='Median')
plt.loglog(sizes, geomeans, marker='^', label='Geometric Mean')
plt.title('Statistics of updates vs N (normal distribution)')
plt.xlabel('Array size N (log scale)')
plt.ylabel('Updates (log scale)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()