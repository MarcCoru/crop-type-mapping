import torch
import torch.utils.data
import numpy
import tslearn
import numpy as np

from tslearn.datasets import UCR_UEA_datasets

class SyntheticDataset(torch.utils.data.Dataset):
    """
    A simple wrapper to insert the dataset in the torch.utils.data.DataLoader module
    that handles multi-threaded loading, sampling, batching and shuffling
    """

    def __init__(self, num_samples=200, T=100):

        self.nclasses = 2
        self.X, self.y = self.sample_dataset(num_samples, T)
        self.X -= self.X.mean()
        self.X /= self.X.std()

        self.T = T

    def sample_dataset(self, num_samples, T):

        def sigma(xdata, x0, k):  # p0 not used anymore, only its components (x0, k)
            y = np.exp(-k * (xdata - x0)) / (1 + np.exp(-k * (xdata - x0)))
            return y

        def noisy_sigma(x, x0, k, eps_x0=10, eps_k=0.1):
            eps_x0 = np.random.rand() * eps_x0
            eps_k = np.random.rand() * eps_k
            return sigma(x, x0 + eps_x0, k + eps_k)

        def sample(t=100):
            y = 0
            X = noisy_sigma(np.arange(t), 50, -0.1, eps_x0=10, eps_k=0.02)

            if np.random.choice([True, False]):
                X *= noisy_sigma(np.arange(t), 75, 0.3, eps_x0=10, eps_k=0.02)
                y = 1

            return X, y

        X = list()
        y = list()
        for i in range(num_samples):
            X_,y_ = sample(t=T)
            X.append(X_)
            y.append(y_)

        return np.vstack(X), np.vstack(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).type(torch.FloatTensor)
        y = torch.from_numpy(np.array(self.y[idx])).type(torch.LongTensor)

        # add 1d hight and width dimensions and copy y for each time
        return X.view(self.T,1,1,1),y.expand(self.T,1,1)

if __name__ == "__main__":

    ds = SyntheticDataset(num_samples=200,T=100)
    X, y = ds[0]
    pass