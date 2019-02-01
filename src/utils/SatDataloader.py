import torch
import torch.utils.data
import pandas as pd
import os

BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
NORMALIZING_FACTOR = 1e-4

class SatDataset(torch.utils.data.Dataset):

    def __init__(self, root):
        self.files = [os.path.join(root,f) for f in os.listdir(root)]

        self.X = list()
        self.y = list()
        self.ids = list()
        for f in self.files:
            sample = pd.read_csv(f,index_col=0)
            self.X.append((sample[BANDS] * NORMALIZING_FACTOR).values)
            self.ids.append(sample["id"].iloc[0])
            self.y.append(sample["label"].values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        id = self.ids[id]

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y, id

if __name__ == "__main__":
    ds = SatDataset("/data/crops/csv/KRUM_2018_MT_pilot")
    X,y = ds[0]

    torch.utils.data.DataLoader(ds)