import torch
import numpy as np
from neural_network import GNN
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


ecg_data =np.loadtxt('processed_data/ecg.csv', delimiter=',',dtype=np.float32)
ppg_data = np.loadtxt('processed_data/ppg.csv', delimiter=',',dtype=np.float32)
qt_data =np.loadtxt('processed_data/qt.csv', delimiter=',',dtype=np.float32)
rr_data = np.loadtxt('processed_data/rr.csv', delimiter=',',dtype=np.float32)

ecg_data_train = torch.from_numpy(ecg_data[:,3000:])
ecg_data_test = torch.from_numpy(ecg_data[:,:3000])
ppg_data_train = torch.from_numpy(ppg_data[:,3000:])
ppg_data_test = torch.from_numpy(ppg_data[:,:3000])
qt_data_train = torch.from_numpy(qt_data[:,3000:])
qt_data_test = torch.from_numpy(qt_data[:,:3000])
rr_data_train = torch.from_numpy(rr_data[:,3000:])
rr_data_test = torch.from_numpy(rr_data[:,:3000])

ecg_dl = DataLoader(TensorDataset(ecg_data_train, ecg_data_test), shuffle=False, batch_size=16,drop_last=True)
ppg_dl = DataLoader(TensorDataset(ppg_data_train,ppg_data_test), shuffle=False, batch_size=16,drop_last=True)
rr_dl = DataLoader(TensorDataset(rr_data_train, rr_data_test), shuffle=False, batch_size=16,drop_last=True)
qt_dl = DataLoader(TensorDataset(qt_data_train,qt_data_test),shuffle=False,  batch_size=16,drop_last=True)

model = GNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = "cuda"
model = model.to(device)
"""
# Training loop
for epoch in range(1000):
    model.train()
    running_loss = 0.0

    for (x1, x2), (y1, y2) in zip(ecg_dl,rr_dl):
        x1, x2 = x1.to(device), x2.to(device)
        y1, y2 = y1.to(device), y2.to(device)

        # Forward pass
        y1_pred, y2_pred = model(x1, x2)

        # Compute individual losses and combine
        loss1 = criterion(y1_pred, y1)
        loss2 = criterion(y2_pred, y2)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{1000}, Loss: {running_loss / len(ecg_dl):.4f}")

torch.save(model.state_dict(), 'model.pth')
"""
model = GNN().cuda()
model.load_state_dict(torch.load('model.pth', weights_only=True))

ecg_inputs, ecg_outputs = next(iter(ecg_dl))
rr_inputs, rr_outputs = next(iter(rr_dl))

ecg_pred, rr_pred = model(ecg_inputs.cuda(), rr_inputs.cuda())


fig = plt.figure()
plt.plot(ecg_pred[0,:1000].cpu().detach(), label='prediction')
plt.plot(ecg_outputs[0,:1000].cpu().detach(), label='ground truth')
plt.legend()
plt.title("ECG output")
plt.savefig('ecg_pred.png')
plt.show()


fig = plt.figure()
plt.plot(rr_pred[0,:1000].cpu().detach(), label='prediction')
plt.plot(rr_outputs[0,:1000].cpu().detach(), label='ground truth')
plt.legend()
plt.title("Heartrate output")
plt.savefig('heartrate_pred.png')
plt.show()
