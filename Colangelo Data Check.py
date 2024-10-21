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



from itertools import product
import numpy as np
from numpy.random import default_rng
import logging
from scipy.stats import norm
from typing import Tuple, TypeVar



np.random.seed(42)
logger = logging.getLogger()


def psi(t: np.ndarray) -> np.ndarray:
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)


def f(p: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 100 + (10 + p) * s * psi(t) - 2 * p


def generate_test_colangelo() -> ATETestDataSet:
    """
    Returns
    -------
    test_data : ATETestDataSet
        Uniformly sampled from price. time and emotion is averaged to get structural.
    """
    treatment = np.linspace(0.0, 1.0, 11)
    structural = 1.2 * treatment + treatment * treatment

    return ATETestDataSet(treatment=treatment[:, np.newaxis],
                          structural=structural[:, np.newaxis])


def generate_train_colangelo(data_size: int,
                             rand_seed: int = 42) -> ATETrainDataSet:
    """
    Generate the data in Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments
    [Colangelo and Lee, 2020]

    Parameters
    ----------
    data_size : int
        size of data
    rand_seed : int
        random seed


    Returns
    -------
    train_data : ATETrainDataSet
    """

    rng = default_rng(seed=rand_seed)
    characteristics_dim = 100
    characteristics_cov = np.eye(characteristics_dim)
    characteristics_cov += np.diag([0.5] * (characteristics_dim - 1), 1)
    characteristics_cov += np.diag([0.5] * (characteristics_dim - 1), -1)
    characteristics = rng.multivariate_normal(np.zeros(characteristics_dim), characteristics_cov,
                                       size=data_size)  # shape of (data_size, characteristics_dim)

    theta = np.array([1.0 / ((i + 1) ** 2) for i in range(characteristics_dim)])
    treatment = norm.cdf(characteristics.dot(theta) * 3) + 0.75 * rng.standard_normal(size=data_size)
    outcome = 1.2 * treatment + 1.2 * characteristics.dot(theta) + treatment ** 2 + treatment * characteristics[:, 0]
    outcome += rng.standard_normal(size=data_size)

    return ATETrainDataSet(characteristics=characteristics,
                           treatment=treatment[:, np.newaxis],
                           outcome=outcome[:, np.newaxis])

def get_kernel_func() -> Tuple[AbsKernel, AbsKernel]:
    return GaussianKernel(), GaussianKernel()





Train = generate_train_colangelo(3000)
Test = generate_test_colangelo()

train_treatment = np.array(Train.treatment, copy=True)
outcome = np.array(Train.outcome, copy=True)




train_treatment = np.array(Train.treatment, copy=True)
outcome = np.array(Train.outcome, copy=True)

n_data = train_treatment.shape[0]


def generate_spd_matrix(n):
    """
    Generate a random symmetric positive definite matrix of size n x n.
    
    Args:
        n (int): The size of the matrix.
        
    Returns:
        np.ndarray: A random symmetric positive definite matrix.
    """
    # Create a random n x n matrix
    A = np.random.rand(n, n)
    
    # Symmetrize A by doing A * A^T
    spd_matrix = np.dot(A, A.T)
    
    # # Ensure the matrix is positive definite by adding n * I (diagonal matrix)
    # spd_matrix += n * np.eye(n)
    
    return spd_matrix



characteristics_kernel_func, treatment_kernel_func = get_kernel_func()
characteristics_kernel_func.fit(Train.characteristics, )
treatment_kernel_func.fit(Train.treatment, )

treatment_kernel = treatment_kernel_func.cal_kernel_mat(Train.treatment, Train.treatment)
characteristics_kernel = characteristics_kernel_func.cal_kernel_mat(Train.characteristics, Train.characteristics)

# treatment_kernel = generate_spd_matrix(n_data)
# characteristics_kernel = generate_spd_matrix(n_data)


n_data = treatment_kernel.shape[0]



#All Prior Code is almost exactly extracted from https://github.com/liyuan9988/KernelCausalFunction/tree/master














def check_conditions(lam):
    # Create the kernel matrix
    kernel_mat = treatment_kernel * characteristics_kernel + n_data * lam * np.eye(n_data)
    # mean_characteristics_kernel = np.mean(characteristics_kernel, axis=0)

    # Check sparsity
    # non_zero_elements = np.count_nonzero(kernel_mat)
    # total_elements = kernel_mat.size
    # sparsity = 1 - (non_zero_elements / total_elements)
    
    # Check condition number
    condition_number = np.linalg.cond(kernel_mat)
    
    # Check rank
    # rank = np.linalg.matrix_rank(kernel_mat)

    # print(f"Sparsity: {sparsity:.4f}")
    # print(f"Condition Number: {condition_number:.4f}")
    # print(f"Rank: {rank}")

    return condition_number


#make a loss function that balances high rank, low condition number and high sparsity

def loss(sparsity,condition,rank):
    return 1/sparsity + condition + 1/rank







import matplotlib.pyplot as plt



# Path to the CSV file
csv_path = os.path.join(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks", 'kernel_conditions_Colangelo.csv')

# Initialize the CSV file with headers before the loop
with open(csv_path, 'w') as f:
    f.write('Lambda,Condition Number\n')



n_lambs = 1000 # (~5 seconds per iteration)
lambda_vals = np.linspace(0.005, 1, n_lambs)

sparsity_list = np.zeros(n_lambs)
condition_list = np.zeros(n_lambs)
rank_list = np.zeros(n_lambs)

# Initialize count correctly in the loop
count = 0

# Run check_conditions for each lam and collect results
for i in tqdm(range(n_lambs)):
    condition= check_conditions(lambda_vals[count])
    # sparsity_list[count] = sparsity
    condition_list[count] = condition
    # rank_list[count] = rank

    with open(csv_path, 'a') as f:
        f.write(f"{lambda_vals[count]},{condition}\n")

    count += 1  # Increment correctly


base_path = r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\Colangelo"

# plt.figure(figsize=(10, 6))
# plt.plot(lambda_vals, sparsity_list, label='Sparsity', color='b')
# plt.xlabel('Lambda')
# plt.ylabel('Sparsity')
# plt.title('Sparsity vs Lambda')
# plt.grid(True)
# sparsity_plot_path = os.path.join(base_path, 'sparsity_plot.png')
# plt.savefig(sparsity_plot_path)
# plt.close()  # Close the figure






import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Plot and save condition
plt.figure(figsize=(10, 6))
plt.plot(lambda_vals, condition_list, label='Condition', color='g')
plt.xlabel('Lambda')
plt.ylabel('Condition')
plt.title('Condition vs Lambda')
plt.grid(True)

# Disable scientific notation for both axes
ax = plt.gca()
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter(useOffset=False))
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter(useOffset=False))

# Force the numbers to be shown as integers/floats instead of scientific notation
ax.ticklabel_format(style='plain')

# Save plot
condition_plot_path = os.path.join(base_path, 'condition_plot.png')
plt.savefig(condition_plot_path)
plt.close()  # Close the figure



# # Plot and save rank
# plt.figure(figsize=(10, 6))
# plt.plot(lambda_vals, rank_list, label='Rank', color='r')
# plt.xlabel('Lambda')
# plt.ylabel('Rank')
# plt.title('Rank vs Lambda')
# plt.grid(True)
# rank_plot_path = os.path.join(base_path, 'rank_plot.png')
# plt.savefig(rank_plot_path)
# plt.close()  # Close the figure

# print(f"CSV saved to: {csv_path}")
# print(f"Plot saved to: {plot_path}")