from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd

from typing import NamedTuple, Optional, Dict, Any, Tuple

import numpy as np

from scipy.spatial.distance import cdist


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
    # sub = data.loc[data["m"] > 0, :]
    # sub = sub.loc[sub["d"] >= 40, :]
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

lam = 1
# if isinstance(lam, list):
#     lam_score = [cal_loocv(treatment_kernel * characteristics_kernel, outcome, lam) for lam in lam]
#     lam = lam[np.argmin(lam_score)]

kernel_mat = treatment_kernel * characteristics_kernel + n_data * lam * np.eye(n_data)
mean_characteristics_kernel = np.mean(characteristics_kernel, axis=0)