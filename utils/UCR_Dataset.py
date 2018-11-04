import torch
import torch.utils.data
import numpy
import tslearn
import numpy as np

from tslearn.datasets import UCR_UEA_datasets

class DatasetWrapper(torch.utils.data.Dataset):
    """
    A simple wrapper to insert the dataset in the torch.utils.data.DataLoader module
    that handles multi-threaded loading, sampling, batching and shuffling
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).type(torch.FloatTensor)
        y = torch.from_numpy(np.array([self.y[idx] - 1])).type(torch.LongTensor)

        # add 1d hight and width dimensions and copy y for each time
        return X.unsqueeze(-1).unsqueeze(-1), y.expand(X.shape[0]).unsqueeze(-1).unsqueeze(-1)

def list_UCR_datasets():
    return UCR_UEA_datasets().list_datasets()

class UCRDataset(torch.utils.data.Dataset):
    """
    A torch wrapper around tslearn UCR datasets
    """

    def __init__(self, name, partition="train", ratio=.75, randomstate=0, silent=False, augment_data_noise=0):
        r = np.random.RandomState(seed=randomstate)

        self.name = name
        self.dataset = UCR_UEA_datasets()

        self.augment_data_noise = augment_data_noise

        if name not in self.dataset.list_datasets():
            raise ValueError("Dataset not found: Please choose from "+", ".join(self.dataset.list_datasets()))

        X_trainvalid, y_trainvalid, X_test, y_test = self.dataset.load_dataset(name)

        train_mask = r.rand(len(X_trainvalid)) < ratio
        valid_mask = np.logical_not(train_mask)

        if partition == "train":
            self.X = X_trainvalid[train_mask]
            self.y = y_trainvalid[train_mask]
        elif partition == "valid":
            self.X = X_trainvalid[valid_mask]
            self.y = y_trainvalid[valid_mask]
        elif partition == "trainvalid":
            self.X = X_trainvalid
            self.y = y_trainvalid
        elif partition == "test":
            self.X = X_test
            self.y = y_test
        else:
            raise ValueError("Invalid partition! please provide either 'train','valid', 'trainvalid', or 'test'")

        self.classes = np.unique(np.append(y_trainvalid, y_test))
        self.nclasses = len(self.classes)
        self.sequencelength = X_trainvalid.shape[1]


        if not silent:
            msg = "Loaded dataset {}-{} T={}, classes={}: {}/{} samples"
            print(msg.format(name,partition,self.sequencelength,self.nclasses,len(self.X),len(X_trainvalid)+len(X_test)))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):


        X = self.X[idx]

        X += np.random.rand(*X.shape) * self.augment_data_noise

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(np.array([self.y[idx] - 1])).type(torch.LongTensor)

        # add 1d hight and width dimensions and copy y for each time
        return X.unsqueeze(-1).unsqueeze(-1), y.expand(X.shape[0]).unsqueeze(-1).unsqueeze(-1)

if __name__ == "__main__":

    for name in list_UCR_datasets():
        print(name)
        #traindataset = UCRDataset(name)