import numpy as np
import pandas as pd

from typing import NamedTuple, Optional,Tuple
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from itertools import product
from numpy.random import default_rng


import os


from tqdm.auto import tqdm



#try to understand this stuff for the LOOCV

def cal_loocv(K, y, lam):
    nData = K.shape[0]
    I = np.eye(nData)
    H = I - K.dot(np.linalg.inv(K + lam * nData * I))
    tildeH_inv = np.diag(1.0 / np.diag(H))
    return np.linalg.norm(tildeH_inv.dot(H.dot(y)))


def cal_loocv_emb(K, kernel_y, lam):
    nData = K.shape[0]
    I = np.eye(nData)
    Q = np.linalg.inv(K + lam * nData * I)
    H = I - K.dot(Q)
    tildeH_inv = np.diag(1.0 / np.diag(H))

    return np.trace(tildeH_inv @ H @ kernel_y @ H @ tildeH_inv)


class CATETrainDataSet(NamedTuple):
    treatment: np.ndarray
    characteristics: np.ndarray
    covariate: np.ndarray
    outcome: np.ndarray


class CATETestDataSet(NamedTuple):
    treatment: np.ndarray
    covariate: np.ndarray
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


class BinaryKernel(AbsKernel):

    def __init__(self):
        super(BinaryKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs):
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        res = data1.dot(data2.T)
        res += (1 - data1).dot(1 - data2.T)
        return res


def generate_test_synthetic() -> CATETestDataSet:
    covariate = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    treatment = np.array([1, 1, 1, 1, 1])  # only test D=1
    structural = covariate * ((1 + 2 * covariate) ** 2) * ((covariate - 1) ** 2)
    return CATETestDataSet(covariate=covariate[:, np.newaxis],
                           treatment=treatment[:, np.newaxis],
                           structural=structural[:, np.newaxis])


def generate_train_synthetic(data_size: int,
                             rand_seed: int = 42) -> CATETrainDataSet:
    rng = default_rng(seed=rand_seed)
    covariate = rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    x1 = 1 + 2 * covariate + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    x2 = 1 + 2 * covariate + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    x3 = (covariate - 1) ** 2 + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    backdoor = np.c_[x1, x2, x3]
    prob = 1.0 / (1.0 + np.exp(-0.5 * (covariate + x1 + x2 + x3)))
    treatment = (rng.random(data_size) < prob).astype(float)
    outcome = covariate * x1 * x2 * x3 + rng.normal(0.0, 0.25, size=(data_size, ))
    outcome *= treatment
    return CATETrainDataSet(treatment=treatment[:, np.newaxis],
                            characteristics=backdoor,
                            covariate=covariate[:, np.newaxis],
                            outcome=outcome[:, np.newaxis])



def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel]:
    if data_name == "job_corp":
        return GaussianKernel(), GaussianKernel(), GaussianKernel()
    elif data_name == "synthetic":
        return GaussianKernel(), BinaryKernel(), GaussianKernel()
    else:
        return GaussianKernel(), GaussianKernel(), GaussianKernel()






#All Prior Code is almost exactly extracted from https://github.com/liyuan9988/KernelCausalFunction/tree/master









def check_conditions(Covariate_kernel, all_kernel_mat,n_data,lam1,lam2):
    # Create the kernel matrix
    kernel_mat = all_kernel_mat + n_data * lam1 * np.eye(n_data)
    cov_kernel_mat = Covariate_kernel +  n_data * lam2 * np.eye(n_data)

    condition_number = np.linalg.cond(kernel_mat)
    condition_number_cov = np.linalg.cond(cov_kernel_mat)


    return condition_number,condition_number_cov


import matplotlib.pyplot as plt


# Path to the CSV file
csv_path = os.path.join(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks", 'kernel_conditions_Synthetic_Scaling_CATE.csv')

# Initialize the CSV file with headers before the loop
with open(csv_path, 'w') as f:
    f.write('n,Lambda,Condition,Lambda_2,Covariate Condition\n')



n_data_vals = 500 # 
data_size_vals = np.linspace(500, 4000, n_data_vals)
data_size_vals = [int(x) for x in data_size_vals]

print(data_size_vals)

# sparsity_list = np.zeros(n_lambs)
condition_list = np.zeros(n_data_vals)
condition_list_cov = np.zeros(n_data_vals)

# rank_list = np.zeros(n_lambs)

# Initialize count correctly in the loop
count = 0

# Run check_conditions for each lam and collect results
for i in tqdm(range(n_data_vals)):

    data_size=data_size_vals[i]

    Train = generate_train_synthetic(data_size)
    Test = generate_test_synthetic()

    train_treatment = np.array(Train.treatment, copy=True)
    outcome = np.array(Train.outcome, copy=True)
    train_covariate = np.array(Train.covariate, copy=True)


    data_name = "synthetic"
    characteristics_kernel_func, treatment_kernel_func,covariate_kernel_func = get_kernel_func(data_name)
    characteristics_kernel_func.fit(Train.characteristics, )
    treatment_kernel_func.fit(Train.treatment, )
    covariate_kernel_func.fit(Train.covariate, )

    treatment_kernel = treatment_kernel_func.cal_kernel_mat(Train.treatment, Train.treatment)
    characteristics_kernel = characteristics_kernel_func.cal_kernel_mat(Train.characteristics, Train.characteristics)
    Covariate_kernel = covariate_kernel_func.cal_kernel_mat(Train.covariate, Train.covariate)

    all_kernel_mat = Covariate_kernel * treatment_kernel * characteristics_kernel


    n_data = data_size


    lam1 = np.linspace(0.001,0.0005,5)
    score = [cal_loocv(all_kernel_mat, outcome, reg) for reg in lam1]
    lam1 = lam1[np.argmin(score)]
    # print(lam1)


    lam2 = np.linspace(0.001,0.0005,5)
    score = [cal_loocv_emb(Covariate_kernel, characteristics_kernel, reg) for reg in lam2]
    lam2 = lam2[np.argmin(score)]
    # print(lam2)

    # lam1 = 0.0001
    # lam2 = 0.0001

    condition,cov_condition = check_conditions(Covariate_kernel, all_kernel_mat,n_data,lam1,lam2)
    # sparsity_list[count] = sparsity
    condition_list[count] = condition
    condition_list_cov[count] = cov_condition

    # rank_list[count] = rank

    with open(csv_path, 'a') as f:
        f.write(f"{data_size_vals[count]},{lam1},{condition},{lam2},{cov_condition}\n")

    count += 1  # Increment correctly







base_path = r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\Synthetic-CATE"


# Plot and save condition
plt.figure(figsize=(10, 6))
plt.plot(data_size_vals, condition_list, color='g')
plt.plot(data_size_vals,condition_list_cov,color = 'b')
plt.xlabel('Data Size')
plt.ylabel('Condition')
plt.title('Condition vs Data Size')
plt.grid(True)
condition_plot_path = os.path.join(base_path, 'condition_plot_datasize.png')
plt.savefig(condition_plot_path)
plt.close()  # Close the figure

