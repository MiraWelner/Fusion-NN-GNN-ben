import torch
import numpy as np
from neural_network import GNN
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import sys
batch_size = 128

ecg_data =np.loadtxt('processed_data/ecg.csv', delimiter=',',dtype=np.float32)
heartrate_data = np.loadtxt('processed_data/rr.csv', delimiter=',',dtype=np.float32)

signal_length = ecg_data.shape[1]//2
ecg_test_split = int(ecg_data.shape[0]*0.8)
heartrate_test_split = int(heartrate_data.shape[0]*0.8)

ecg_data_train_input = torch.from_numpy(ecg_data[:ecg_test_split,signal_length:])
ecg_data_train_output = torch.from_numpy(ecg_data[:ecg_test_split,:signal_length])
ecg_data_test_input = torch.from_numpy(ecg_data[ecg_test_split:,signal_length:])
ecg_data_test_output = torch.from_numpy(ecg_data[ecg_test_split:,:signal_length])

heartrate_data_train_input = torch.from_numpy(heartrate_data[:heartrate_test_split,signal_length:])
heartrate_data_train_output = torch.from_numpy(heartrate_data[:heartrate_test_split,:signal_length])
heartrate_data_test_input = torch.from_numpy(heartrate_data[heartrate_test_split:,signal_length:])
heartrate_data_test_output = torch.from_numpy(heartrate_data[heartrate_test_split:,:signal_length])

if ecg_data_test_input.shape[0] <= batch_size or heartrate_data_test_input.shape[0] <= batch_size:
    print("batch size too large")
    print(f"batch size: {batch_size}")
    print(f"HEARTRATE:{heartrate_data_test_input.shape[0]}")
    print(f"ECG:{ecg_data_test_input.shape[0]}")
    sys.exit(1)
ecg_dl = DataLoader(TensorDataset(ecg_data_train_input, ecg_data_train_output), shuffle=False, batch_size=batch_size,drop_last=True)
rr_dl = DataLoader(TensorDataset(heartrate_data_train_input, heartrate_data_train_output), shuffle=False, batch_size=batch_size,drop_last=True)

model = GNN(signal_length=signal_length)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = "cuda"
model = model.to(device)

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


model = GNN(signal_length=signal_length).cuda()
model.load_state_dict(torch.load('model.pth', weights_only=True))

test_wave_ecg = random.randrange(ecg_test_split//4)
test_wave_heartrate = random.randrange(heartrate_test_split//4)

ecg_pred, heartrate_pred = model(ecg_data_test_input[test_wave_ecg,:].unsqueeze(0).cuda(),
                                 heartrate_data_test_input[test_wave_heartrate,:].unsqueeze(0).cuda())

_, axes = plt.subplots(2,1, figsize=(15,7), layout='constrained')
axes[0].plot(range(signal_length), ecg_data_test_input[test_wave_ecg,:].squeeze(), label='Input from Validation Set')
axes[0].plot(range(signal_length,signal_length*2), ecg_pred.cpu().detach().squeeze(), label='Prediction from Validation Set')
axes[0].plot(range(signal_length,signal_length*2), ecg_data_test_output[test_wave_ecg,:].cpu().detach(), label='Ground Truth from Validation Set')
axes[0].set_xticks(range(0,5001,500), range(0,11))
axes[0].set_xlabel("Time (s)")
axes[0].legend()
axes[0].set_title("ECG Prediction vs Ground Truth")

axes[1].plot(range(signal_length), heartrate_data_test_input[test_wave_heartrate,:].squeeze(), label='Input from Validation Set')
axes[1].plot(range(signal_length,signal_length*2), heartrate_pred.cpu().detach().squeeze(), label='Prediction from Validation Set')
axes[1].plot(range(signal_length,signal_length*2), heartrate_data_test_output[test_wave_heartrate,:].cpu().detach(), label='Ground Truth from Validation Set')
axes[1].set_xticks(range(0,5001,500), range(0,251,25))
axes[1].set_xlabel("Time (s)")
axes[1].legend()
axes[1].set_title("Heartrate Prediction vs Ground Truth")
plt.savefig('feed_forward_fusion_output.png')
plt.show()
