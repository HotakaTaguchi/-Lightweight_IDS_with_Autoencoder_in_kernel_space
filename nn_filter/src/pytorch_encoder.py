import numpy as np
import argparse
import random
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import classification_report,accuracy_score,roc_curve, auc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import math

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e9)
    os.remove('temp.p')

def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e9
    os.remove('temp.p')
    return size

# モデルの定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.quant1 = torch.quantization.QuantStub()
        self.layer1 = nn.Linear(12, 10, bias=False)
        self.layer2 = nn.Linear(10, 8, bias=False)
        self.layer3 = nn.Linear(8, 6, bias=False)
        self.layer4 = nn.Linear(6, 8, bias=False)
        self.layer5 = nn.Linear(8, 10, bias=False)
        self.layer6 = nn.Linear(10, 12, bias=False)
        self.dequant1 = torch.quantization.DeQuantStub()
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.sigmoid1 = torch.nn.Hardsigmoid()
        self.sigmoid2 = torch.nn.Hardsigmoid()


    def forward(self, x):
        x = self.quant1(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        # x = self.sigmoid1(x)
        x = self.layer5(x)
        # x = self.sigmoid2(x)
        x = self.layer6(x)
        x = self.dequant1(x)
        return x

#TensorDatasetのカスタムクラス(全データセット)
class PacketDataset(Dataset):
    def __init__(self, data, labels, categories):
        self.data = data
        self.labels = labels
        self.categories = categories
        assert(len(self.data) == len(self.labels) == len(self.categories))

    def __getitem__(self, index):
        data, labels, categories = torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index]), torch.FloatTensor(self.categories[index])
        return data, labels, categories

    def __len__(self):
        return len(self.data)

#TensorDatasetのカスタムクラス(分割データ)
class PacketFlowDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert(len(self.data) == len(self.labels))

    def __getitem__(self, index):
        data, labels = torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index])
        return data, labels

    def __len__(self):
        return len(self.data)

#学習用エポック関数
def train_epoch(model:nn.Module, data_loader:DataLoader, optimizer:Adam, loss_fn:nn.MSELoss):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        optimizer (Adam)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average training loss on epoch
    """
    model.train(mode=True)
    num_batches = len(data_loader)

    loss = 0
    for x, y in data_loader:
        optimizer.zero_grad()
        y = y
        logits = model(x)
        batch_loss = loss_fn(logits, y)

        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
    return loss / num_batches


# 推論評価用エポック関数
def eval_epoch(model: nn.Module, data_loader:DataLoader, loss_fn:nn.MSELoss):
    """
    Evaluate epoch on validation data
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average validation loss 
    """
    model.eval()
    num_batches = len(data_loader)

    loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            pred_y = model(x)
            y = y.squeeze().long()
            batch_loss = loss_fn(pred_y, x)
            loss += batch_loss.item()
    return loss / num_batches

#訓練・テストデータ分割関数
def get_nth_split(dataset, n_fold, index):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
    train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
    return train_indices, test_indices

#convert data
def get_dataset(data,is_train=True, x=float('inf')):
    data_list = []
    for item in data:
        data_list.append(item)

    new_dataset = []
    new_labels = []

    new_dataset2 = []
    for item, y, y2 in tqdm(data_list):
        item = item.numpy()
        y = y.numpy().astype(np.int64)
        y2 = y2.numpy().astype(np.int64)

        average = np.zeros(3)
        deviation = np.zeros(3)
        time = 0
        for i in range(item.shape[0]):
            item[i,4] = item[i,4] * 1000
            #print(i,':',item[i,0])
            current_vector = item[i,:6]

            average += current_vector[3:]

            current_average = (average/(i+1))

            deviation += np.abs(current_vector[3:]-current_average)
            current_deviation = (deviation/(i+1))

            final_vector = np.concatenate((current_vector, current_average, current_deviation))
        #print('final')
        if is_train and y[0]==0:
            new_dataset.append(final_vector)
            new_labels.append(y[0])
        elif not is_train:
            new_dataset.append(final_vector)
            new_labels.append(y[0])
    # scaler = MinMaxScaler()
    # scaler.fit(new_dataset)
    # new_dataset = scaler.transform(new_dataset)

    return new_dataset, new_labels

#パラメータ取得
def quantize_model_params(model):
    state_dict = dict()
    model = torch.quantization.convert(model)

    for layer_idx in range(1, 7):
        weight = model.__getattr__(f"layer{layer_idx}").weight()
        s_w = model.__getattr__(f"layer{layer_idx}").weight().q_per_channel_scales().numpy()
        z_w = model.__getattr__(f"layer{layer_idx}").weight().q_per_channel_zero_points().numpy()
        s_y = model.__getattr__(f"layer{layer_idx}").scale
        z_y = model.__getattr__(f"layer{layer_idx}").zero_point
        if layer_idx == 1:
            s_x = model.__getattr__(f"quant{layer_idx}").scale.numpy()
            z_x = model.__getattr__(f"quant{layer_idx}").zero_point.numpy()
        else:
            s_x = model.__getattr__(f"layer{layer_idx-1}").scale
            z_x = model.__getattr__(f"layer{layer_idx-1}").zero_point
        

        state_dict[f'layer_{layer_idx}_weight'] = torch.int_repr(weight).numpy()
        state_dict[f'layer_{layer_idx}_s_x_inv'] = 1 / s_x
        state_dict[f'layer_{layer_idx}_s_w_inv'] = 1 / s_w
        state_dict[f'layer_{layer_idx}_z_w'] = z_w
        state_dict[f'layer_{layer_idx}_z_x'] = z_x
        state_dict[f'layer_{layer_idx}_s_y'] = s_y
        state_dict[f'layer_{layer_idx}_z_y'] = z_y
        state_dict[f'layer_{layer_idx}_S'] = s_x * s_w / s_y
        
        
    return state_dict

#パラメータ書き込み
def output_params(save_dir, filename, inp):
    saved_stats = torch.load(os.path.join(save_dir, filename))
    state_dict = saved_stats['state_dict']
    input = inp
    quant_input= 0
    output = 0
    with open('mlp_params.c', 'w') as f:
        for layer_idx in range(1, 7):
            name = f'layer_{layer_idx}_s_x_inv'
            s_x = state_dict[name]

            name = f'layer_{layer_idx}_z_x'
            z_x = state_dict[name]

            name = f'layer_{layer_idx}_s_w_inv'
            s_w = state_dict[name]

            name = f'layer_{layer_idx}_z_w'
            z_w = state_dict[name]

            name = f'layer_{layer_idx}_s_y'
            s_y = state_dict[name]

            name = f'layer_{layer_idx}_z_y'
            z_y = state_dict[name]

            name = f'layer_{layer_idx}_S'
            s = state_dict[name] 

            name = f'layer_{layer_idx}_weight'
            w = state_dict[name].T
            if layer_idx == 1:
                quant_input = np.round(input * s_x ) + z_x
            else:
                quant_input = torch.from_numpy(output)
            output = np.round(s * np.matmul(quant_input - z_x, w - z_w).numpy()) + z_y
            if layer_idx < 4:
                output = nn.functional.relu(torch.tensor(output)).numpy()
            print('input',layer_idx,quant_input)
            print('output',layer_idx,output,(output-z_y)*s_y)





#パラメータ書き込み
def create_params(save_dir, filename):
    saved_stats = torch.load(os.path.join(save_dir, filename))
    state_dict = saved_stats['state_dict']
    threshold = saved_stats['threshold']
    
    with open('mlp_params.c', 'w') as f:
        for layer_idx in range(1, 7):
            name = f'layer_{layer_idx}_s_x_inv'
            fxp_value = (state_dict[name] * (2**16))
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_z_x'
            fxp_value = (state_dict[name] * (2**16))
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_s_w_inv'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name}[{len(fxp_value)}] = {{")
            for idx in range(len(fxp_value)):
                f.write(f"{int(fxp_value[idx])}")
                if idx < len(fxp_value) - 1:
                     f.write(", ")
            f.write("};\n\n")

            name = f'layer_{layer_idx}_z_w'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name}[{len(fxp_value)}] = {{")
            for idx in range(len(fxp_value)):
                f.write(f"{int(fxp_value[idx])}")
                if idx < len(fxp_value) - 1:
                     f.write(", ")
            f.write("};\n\n")

            name = f'layer_{layer_idx}_s_y'
            fxp_value = (state_dict[name] * (2**16))
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_z_y'
            fxp_value = (state_dict[name] * (2**16))
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_S'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name}[{len(fxp_value)}] = {{")
            for idx in range(len(fxp_value)):
                f.write(f"{int(fxp_value[idx])}")
                if idx < len(fxp_value) - 1:
                     f.write(", ")
            f.write("};\n\n")


        for layer_idx in range(1, 7):
                name = f'layer_{layer_idx}_weight'
                param = state_dict[name].flatten()
                f.write(f"const int8_t {name}[{len(param)}] = {{")
                for idx in range(len(param)):
                    f.write(f"{param[idx]}")
                    if idx < len(param) - 1:
                        f.write(", ")
                f.write("};\n")
        
        f.write("\n\n")
        name = 'threshold'
        #追加部分
        fxp_value = 193
        #fxp_value = (threshold * (2**16)).round()
        f.write(f"const int {name} = {int(fxp_value)};\n\n")
    
