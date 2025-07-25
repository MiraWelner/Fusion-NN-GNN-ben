import torch
import numpy as np
from neural_network import GNN
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
signal_length = 2500
batch_size = 512

ecg_data =np.loadtxt('processed_data/ecg.csv', delimiter=',',dtype=np.float32)
rr_data = np.loadtxt('processed_data/rr.csv', delimiter=',',dtype=np.float32)

ecg_data_train = torch.from_numpy(ecg_data[:,signal_length//2:])
ecg_data_test = torch.from_numpy(ecg_data[:,:signal_length//2])
rr_data_train = torch.from_numpy(rr_data[:,signal_length//2:])
rr_data_test = torch.from_numpy(rr_data[:,:signal_length//2])

ecg_dl = DataLoader(TensorDataset(ecg_data_train, ecg_data_test), shuffle=False, batch_size=batch_size,drop_last=True)
rr_dl = DataLoader(TensorDataset(rr_data_train, rr_data_test), shuffle=False, batch_size=batch_size,drop_last=True)

model = GNN(signal_length=signal_length//2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = "cuda"
model = model.to(device)
"""
# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for (ecg_train, ecg_test), (rr_train, rr_test) in zip(ecg_dl,rr_dl):
        ecg_train, ecg_test, rr_train, rr_test = ecg_train.to(device), ecg_test.to(device), rr_train.to(device), rr_test.to(device)

        # Forward pass
        ecg_pred, rr_pred = model(ecg_train, rr_train)

        # Compute individual losses and combine
        loss1 = criterion(ecg_pred, ecg_test)
        loss2 = criterion(rr_pred, rr_test)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(ecg_dl):.4f}")

torch.save(model.state_dict(), 'model.pth')
"""
model = GNN(signal_length=signal_length//2).cuda()
model.load_state_dict(torch.load('model.pth', weights_only=True))

ecg_train, ecg_test = next(iter(ecg_dl))
rr_train, rr_test = next(iter(rr_dl))

ecg_pred, rr_pred = model(ecg_train.cuda(), rr_train.cuda())
fig = plt.figure()
plt.plot(ecg_pred[0,:].cpu().detach(), label='prediction')
plt.plot(ecg_test[0,:].cpu().detach(), label='ground truth')
plt.legend()
plt.title("ECG output")
plt.savefig('ecg_pred.png')
plt.show()


fig = plt.figure()
plt.plot(rr_pred[0,:].cpu().detach(), label='prediction')
plt.plot(rr_test[0,:].cpu().detach(), label='ground truth')
plt.legend()
plt.title("Heartrate output")
plt.savefig('heartrate_pred.png')
plt.show()
