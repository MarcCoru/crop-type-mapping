import torch.utils.data
import numpy as np
from tslearn.datasets import CachedDatasets
from models.DualOutputRNN import DualOutputRNN

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return X_train.shape[0]

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).type(torch.FloatTensor)
        y = torch.from_numpy(np.array([self.y[idx]-1])).type(torch.LongTensor)

        # add 1d hight and width dimensions and copy y for each time
        return X.unsqueeze(-1).unsqueeze(-1), y.expand(X.shape[0]).unsqueeze(-1).unsqueeze(-1)


batchsize=10
workers=0
learning_rate=1e-1
epochs=150
switch_epoch=50
earliness_factor=1e-1

X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

nclasses = len(set(y_train))

traindataset = DatasetWrapper(X_train, y_train)
testdataset = DatasetWrapper(X_test, y_test)

# handles multithreaded batching and shuffling
traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=workers)
testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=False,num_workers=workers)

model = DualOutputRNN(input_dim=1, nclasses=nclasses)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.NLLLoss(reduction='none') # reduction='none'

if torch.cuda.is_available():
    model = model.cuda()
    loss = loss.cuda()

stats = dict(
    loss=list(),
    loss_earliness=list(),
    loss_classif=list())

for epoch in range(epochs):

    for iteration, data in enumerate(traindataloader):
        optimizer.zero_grad()

        inputs, targets = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        predicted_probas, Pts = model.forward(inputs)

        loss_classif = torch.mean((Pts * loss(predicted_probas.permute(0,2,1,3,4), targets)))

        alphat = model._build_regularization_earliness(earliness_factor=earliness_factor, out_shape=Pts.shape)

        if torch.cuda.is_available():
            alphat = alphat.cuda()

        loss_earliness = torch.mean((Pts * alphat))

        if epoch < switch_epoch:
            l = loss_classif
        else:
            l = loss_classif + loss_earliness

        stats["loss"].append(l.detach().cpu().numpy())
        stats["loss_classif"].append(loss_classif.detach().cpu().numpy())
        stats["loss_earliness"].append(loss_earliness.detach().cpu().numpy())

        l.backward()
        optimizer.step()



    print("[End of training] Epoch:", '%04d' % (epoch + 1),
        "loss={loss:.9f}, loss_classif={loss_classif:.9f}, loss_earliness={loss_earliness:.9f}".format(
            loss=np.array(stats["loss"]).mean(),
            loss_classif=np.array(stats["loss_classif"]).mean(),
            loss_earliness = np.array(stats["loss_earliness"]).mean()
            )
        )

pass

