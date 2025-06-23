from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# from mamba import Mamba, MambaConfig,TCN
# Set random seed
# SEED = 1
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}.")

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

rcParams['figure.figsize'] = 13, 4

# Box
rcParams['axes.spines.top'] = False
rcParams['axes.spines.left'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.prop_cycle'] = cycler(color=['navy','goldenrod'])

# Grid and axis thickness, color
rcParams['axes.linewidth'] = 1
rcParams['axes.edgecolor'] = '#5B5859'
rcParams['axes.ymargin'] = 0
rcParams['axes.grid'] = True
rcParams['axes.grid.axis'] = 'y'
rcParams['axes.axisbelow'] = True
rcParams['grid.color'] = 'grey'
rcParams['grid.linewidth'] = 0.5

# xticks, yticks
rcParams['ytick.major.width'] = 0
rcParams['ytick.major.size'] = 0
rcParams['ytick.color'] = '#393433'
rcParams['xtick.major.width'] = 1
rcParams['xtick.major.size'] = 3
rcParams['xtick.color'] = '#393433'

# Line thickness
rcParams['lines.linewidth'] = 1.5

# Saving quality
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.dpi'] = 500
rcParams['savefig.transparent'] = True

csv_path = Path(f"E:\Project2024\MambaLithium-main\MambaLithium-main\CIBW.csv")
df = pd.read_csv(csv_path)
df["Date"] = pd.to_datetime(df["Timestamp"])
df.drop(['Project'], axis=1, inplace=True)


# Day in a month
df["Day_of_month"] = df.Date.apply(lambda x: x.day)
# Day in a week
df["Day_of_week"] = df.Date.apply(lambda x: x.dayofweek)
# 24-hour based
df["Hour"] = df.Date.apply(lambda x: x.hour)
# Week in a year
# df["Week"] = df.Date.apply(lambda x: x.week)

# Set "DateTime" column as row index
df.drop(['Timestamp'], axis=1, inplace=True)
df.drop(['Date'], axis=1, inplace=True)


batch_size = 64
out_seq_len=4
in_seq_len=24
hidden_size=16
print(int((in_seq_len+out_seq_len)/2))
input_size = df.shape[1]


num_epochs = 200
learning_rate = 1e-2
es_patience = 5
lr_patience = 5
model_save_path = "checkpoint_GRU.pth"


df['Outflow'] = np.log1p(df['Outflow'])
df['Temperature'] = np.log1p(df['Temperature'])

# scaler_qrate = StandardScaler()
# scaler_rain = StandardScaler()
# df['Outflow'] = scaler_qrate.fit_transform(df[['Outflow']])
# df['Temperature'] = scaler_rain.fit_transform(df[['Temperature']])
# Move target to the last column
target_feature = "Outflow"
df.insert(len(df.columns)-1, target_feature, df.pop(target_feature))
testNum = validationNum = 8750
total_rows = len(df)

train_ratio = 0.75
val_ratio = 0.25

# 计算每部分的行数
train_size = int(total_rows * train_ratio)
val_size = int(total_rows * val_ratio)
print("total_rows:", total_rows)
print("train_size:", train_size)
print("validation_size:", val_size)
# assert total_rows == testNum * 2 + (train_df_rows := total_rows - 2*testNum), "数据行数不支持这样的划分"
data_train = df.iloc[:train_size].copy()
data_val= df.iloc[train_size :].copy()
# data_train = df[:train_df_rows].copy()
# data_val = df[train_df_rows:train_df_rows+validationNum].copy()
# data_test = df[train_df_rows+validationNum:train_df_rows+validationNum+testNum].copy()
print("Training Shape:", data_train.shape)
print("Validation Shape:", data_val.shape)
# split a multivariate sequence past, future samples (X and y)
print(df.head())
def sequence_generator(arr, past_step=in_seq_len, future_step=out_seq_len):
    # instantiate X and y
    X, y = list(), list()
    for i in range(len(arr)):
        # find the end of the input, output sequence
        input_end = i + past_step
        output_end = input_end + future_step
        # check if we are beyond the dataset
        if output_end > len(arr):
            break
        else:
            # gather input and output of the pattern
            # seq_x, seq_y = X_arr[i : input_end], y_arr[input_end : output_end, -1]
            seq_x, seq_y = arr[i : input_end], arr[input_end : output_end]
            X.append(seq_x), y.append(seq_y)

    return np.array(X), np.array(y)

X_train, y_train = sequence_generator(data_train)
X_val, y_val = sequence_generator(data_val)

print("Training Shape:", X_train.shape, y_train.shape)
print("Validation Shape:", X_val.shape, y_val.shape)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        print(f"Type of X: {type(X)}")
        print(f"Type of y: {type(y)}")
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        # target = self.y[idx, :,-1].unsqueeze(-1)
        target = self.y[idx, :, :]
        return features, target

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
X, y = next(iter(train_loader))

print("Features shape:", X.size())
print("Target shape:", y.size())

class Net(nn.Module):
    def __init__(self, input_size,out_seq_len,hidden_size):
        super().__init__()
        self.input_size = input_size
        self.GRU=nn.LSTM(input_size,hidden_size=hidden_size, num_layers=1, batch_first=True)
        #self.BiLSTM = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.linear=nn.Linear(hidden_size,out_seq_len)

    def forward(self, x):
        x,_=self.GRU(x)#64,24,5
        x = x[:, -1, :]#64,24,64
        x=self.linear(x).unsqueeze(2)
        return x

model = Net(input_size, out_seq_len, hidden_size).to(device)
total_params = sum(p.numel() for p in model.parameters())
learn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print(f"\nTotal parameters: {total_params}")
print(f"Learnable parameters: {learn_params}")

loss_func = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=learning_rate)

# Early Stopping
# Stop training if validation loss does not improve
class EarlyStopping:

    def __init__(self, patience, model_save_path, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.model_save_path = model_save_path
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = 0
        self.early_stop = False


    def __call__(self, epoch, model, validation_loss):
        delta_loss = self.min_validation_loss - validation_loss
        # Check if val loss is smaller than min loss
        if delta_loss > self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # Save best model
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.model_save_path)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early Stopping.")
                print(f"Save best model at epoch {self.best_epoch}")
                self.early_stop = True

# ReduceLROnPlateau
# Reduce learning rate when validation loss stops improving
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=lr_patience, verbose=True)
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        output = model(X)
        target=y[:, :, -1].unsqueeze(2)
        loss = loss_function(output, target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_avg_loss = total_loss / num_batches

    return train_avg_loss


def val_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            output = model(X)
            total_loss += loss_function(output, y[:, :, -1].unsqueeze(2)).item()

    val_avg_loss = total_loss / num_batches

    return val_avg_loss
# Log losses for plotting
all_losses = []

# Initialize Early Stopping object
early_stopper = EarlyStopping(patience=es_patience, model_save_path=model_save_path)
for epoch in range(num_epochs):
    train_loss = train_model(train_loader, model, loss_func, opt)
    val_loss = val_model(val_loader, model, loss_func)
    all_losses.append([train_loss, val_loss])

    # Display
    print(f"\nEpoch [{epoch}/{num_epochs-1}]\t\tTrain loss: {train_loss:.6f} - Val loss: {val_loss:.6f}")

    # EarlyStopping
    early_stopper(epoch, model, val_loss)
    if early_stopper.early_stop:
        break
    # Adjust learning rate
    lr_scheduler.step(val_loss)

plt.title("Linear Model", size=18, y=1.1)
plt.plot(all_losses, label=["Train loss", "Val loss"])
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("MSE", fontsize=13)
plt.legend(loc="upper right", fontsize=10)
plt.show()
model.load_state_dict(torch.load(model_save_path))

def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_pred = model(X.to(device))
            output = torch.cat((output.to(device), y_pred.to(device)), 0)

    return output

y_pred = predict(val_loader, model).cpu().numpy()
print(y_pred[0,:,:])
print("y_pred Shape:", y_pred.shape)
y_pred = y_pred[:, :, -1]
y_pred = np.expand_dims(y_pred,axis=-1)
print("y_pred Shape:", y_pred.shape)

# Inverse the transformation

# y_pred_inv_flattened = scaler_qrate.inverse_transform(y_pred .reshape(-1, 1))
# y_pred_inv1 =y_pred_inv_flattened.reshape(-1,out_seq_len,1)
y_test=y_val[:,:,-1]
print("y_test Shape:", y_test.shape)
# y_test_inv_flattened = scaler_qrate.inverse_transform(y_test.reshape(-1, 1))
# y_test_inv1 =y_test_inv_flattened.reshape(-1,out_seq_len,1)
y_pred_inv = np.expm1(y_pred)
y_test_inv = np.expm1(y_val)

# Hours ahead to predict
forecast_length =out_seq_len-1
# 确保预测值和真实值的形状一致
truth = y_test_inv[:, :,-1]  # 取所有时间步长的真实值
forecast = y_pred_inv[:, :,-1]  # 取所有时间步长的预测值
print(truth.shape)
columns_truth = [f'truth_{i}' for i in range(out_seq_len)]
df_truth = pd.DataFrame(data = truth, columns = columns_truth)
df_truth.to_csv('truth_BiLSTM .csv', index=False)
columns_forecast = [f'forecast_{i}' for i in range(out_seq_len)]
df_forecast = pd.DataFrame(data = forecast, columns = columns_forecast)
df_forecast.to_csv('forecast_BiLSTM .csv', index=False)
# 计算误差
diff = np.subtract(truth, forecast)

# 计算指标
mae = np.mean(np.abs(diff))  # MAE
mse = np.mean(np.square(diff))  # MSE
rmse = np.sqrt(mse)  # RMSE

# NSE
num = np.sum(np.square(diff))
den = np.sum(np.square(np.subtract(truth, truth.mean())))
nse = 1 - (num / den)

# R^2
numerator = np.square(np.sum((truth - truth.mean()) * (forecast - forecast.mean())))
denominator = np.sum(np.square(truth - truth.mean())) * np.sum(np.square(forecast - forecast.mean()))
r_squared = numerator / denominator

# RSR
rsr = rmse / np.std(truth)

# Pbias
pbias = np.mean(diff / truth) * 100

# 输出结果
print(f"Overall forecast MAE : {mae:.4f}")
print(f"Overall forecast MSE: {rmse:.4f}")
print(f"Overall forecast NSE: {nse:.4f}")
print(f"Overall forecast R^2: {r_squared:.4f}")
print(f"Overall forecast RSR: {rsr:.4f}")
print(f"Overall forecast Pbias: {pbias:.2f}%")
print(y_val.shape)
print(y_pred.shape)

truth = y_test_inv[:, forecast_length,-1]
forecast = y_pred_inv [:,forecast_length, -1]

print(truth.shape)
plt.figure(figsize=(12, 4))
plt.title(f"linear-{forecast_length}-Hour Ahead Forecasting ", size=16, y=1.1)
plt.plot(truth, label="Ground Truth", color="teal")
plt.plot(forecast, label="Prediction", color="darkred")
plt.xlabel("Observation")
plt.legend(fontsize=10)
plt.show()
df_truth = pd.DataFrame(truth, columns=['Ground Truth'])
df_forecast = pd.DataFrame(forecast, columns=['Prediction'])
df_truth.to_csv('ground_truth_BiLSTM .csv', index=False)
df_forecast.to_csv('prediction_BiLSTM .csv', index=False)