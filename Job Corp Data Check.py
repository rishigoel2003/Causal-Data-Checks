import numpy as np
import pandas as pd

from typing import NamedTuple, Optional,Tuple
from scipy.spatial.distance import cdist
from scipy.sparse import issparse

import os


from tqdm.auto import tqdm



class ATETrainDataSet(NamedTuple):
    treatment: np.ndarray
    characteristics: np.ndarray
    outcome: np.ndarray


class ATETestDataSet(NamedTuple):
    treatment: np.ndarray
    structural: Optional[np.ndarray]



class AbsKernel:
    def fit(self, data: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        # Compute gradient of kernel matrix with respect to data2.
        raise NotImplementedError

    def cal_kernel_mat_jacob(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        # Compute jacobi matrix of kernel matrix. dx1 dx2 k(x1, x2)
        # assert data1 and data2 are single dimensions
        raise NotImplementedError

class GaussianKernel(AbsKernel):
    sigma: np.float64

    def __init__(self):
        super(GaussianKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.sigma = np.median(dists)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        return np.exp(-dists)

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        res = np.exp(-dists)[:, :, np.newaxis]
        res = res * 2 / self.sigma * (data1[:, np.newaxis, :] - data2[np.newaxis, :, :])
        return res

    def cal_kernel_mat_jacob(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        res = np.exp(-dists) * (2 / self.sigma - 4 / self.sigma * dists)
        return res



def generate_train_jobcorp():
    data = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\JCdata.csv", sep=" ")

    sub = data
    sub = data.loc[data["m"] > 0, :]
    sub = sub.loc[sub["d"] >= 40, :]
    outcome = sub["m"].to_numpy()
    treatment = sub["d"].to_numpy()
    characteristics = sub.iloc[:, 3:].to_numpy()
    return ATETrainDataSet(characteristics=characteristics,
                           outcome=outcome[:,np.newaxis],
                           treatment=treatment[:,np.newaxis])


def generate_test_jobcorp():
    return ATETestDataSet(treatment=np.linspace(40, 2500, 1000)[:, np.newaxis],
                          structural=None)

def get_kernel_func() -> Tuple[AbsKernel, AbsKernel]:
    return GaussianKernel(), GaussianKernel()





Train = generate_train_jobcorp()
Test = generate_test_jobcorp()

train_treatment = np.array(Train.treatment, copy=True)
outcome = np.array(Train.outcome, copy=True)



print(np.shape(train_treatment))
print(np.shape(outcome))




train_treatment = np.array(Train.treatment, copy=True)
outcome = np.array(Train.outcome, copy=True)


characteristics_kernel_func, treatment_kernel_func = get_kernel_func()
characteristics_kernel_func.fit(Train.characteristics, )
treatment_kernel_func.fit(Train.treatment, )

treatment_kernel = treatment_kernel_func.cal_kernel_mat(Train.treatment, Train.treatment)
characteristics_kernel = characteristics_kernel_func.cal_kernel_mat(Train.characteristics, Train.characteristics)
n_data = treatment_kernel.shape[0]



#All Prior Code is almost exactly extracted from https://github.com/liyuan9988/KernelCausalFunction/tree/master














def check_conditions(lam):
    # Create the kernel matrix
    kernel_mat = treatment_kernel * characteristics_kernel + n_data * lam * np.eye(n_data)
    # mean_characteristics_kernel = np.mean(characteristics_kernel, axis=0)

    # Check sparsity
    non_zero_elements = np.count_nonzero(kernel_mat)
    total_elements = kernel_mat.size
    sparsity = 1 - (non_zero_elements / total_elements)
    
    # Check condition number
    condition_number = np.linalg.cond(kernel_mat)
    
    # Check rank
    rank = np.linalg.matrix_rank(kernel_mat)

    # print(f"Sparsity: {sparsity:.4f}")
    # print(f"Condition Number: {condition_number:.4f}")
    # print(f"Rank: {rank}")

    return sparsity,condition_number,rank


#make a loss function that balances high rank, low condition number and high sparsity

def loss(sparsity,condition,rank):
    return 1/sparsity + condition + 1/rank







n_lambs = 1

lambda_vals = np.linspace(0.005,1,n_lambs)


import matplotlib.pyplot as plt



# Path to the CSV file
csv_path = os.path.join(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks", 'kernel_conditions_Job_Corps.csv')

# Initialize the CSV file with headers before the loop
with open(csv_path, 'w') as f:
    f.write('Lambda,Sparsity,Condition Number,Rank\n')



n_lambs = 1000 #~5 hours (~18 seconds per iteration)
lambda_vals = np.linspace(0.005, 1, n_lambs)

sparsity_list = np.zeros(n_lambs)
condition_list = np.zeros(n_lambs)
rank_list = np.zeros(n_lambs)

# Initialize count correctly in the loop
count = 0

# Run check_conditions for each lam and collect results
for i in tqdm(range(n_lambs)):
    sparsity, condition, rank = check_conditions(lambda_vals[count])
    sparsity_list[count] = sparsity
    condition_list[count] = condition
    rank_list[count] = rank

    with open(csv_path, 'a') as f:
        f.write(f"{lambda_vals[count]},{sparsity},{condition},{rank}\n")

    count += 1  # Increment correctly


base_path = r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\Job-Corps"

plt.figure(figsize=(10, 6))
plt.plot(lambda_vals, sparsity_list, label='Sparsity', color='b')
plt.xlabel('Lambda')
plt.ylabel('Sparsity')
plt.title('Sparsity vs Lambda')
plt.grid(True)
sparsity_plot_path = os.path.join(base_path, 'sparsity_plot.png')
plt.savefig(sparsity_plot_path)
plt.close()  # Close the figure

# Plot and save condition
plt.figure(figsize=(10, 6))
plt.plot(lambda_vals, condition_list, label='Condition', color='g')
plt.xlabel('Lambda')
plt.ylabel('Condition')
plt.title('Condition vs Lambda')
plt.grid(True)
condition_plot_path = os.path.join(base_path, 'condition_plot.png')
plt.savefig(condition_plot_path)
plt.close()  # Close the figure

# Plot and save rank
plt.figure(figsize=(10, 6))
plt.plot(lambda_vals, rank_list, label='Rank', color='r')
plt.xlabel('Lambda')
plt.ylabel('Rank')
plt.title('Rank vs Lambda')
plt.grid(True)
rank_plot_path = os.path.join(base_path, 'rank_plot.png')
plt.savefig(rank_plot_path)
plt.close()  # Close the figure

# print(f"CSV saved to: {csv_path}")
# print(f"Plot saved to: {plot_path}")