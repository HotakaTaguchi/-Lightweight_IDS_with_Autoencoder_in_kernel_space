import numpy as np
import argparse
import random
import os
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pytorch_encoder import PacketDataset, PacketFlowDataset, train_epoch ,eval_epoch, get_nth_split, get_dataset, print_size_of_model, get_size_of_model
import time
import pandas as pd
import json


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


    def forward(self, x):
        x = self.quant1(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.dequant1(x)
        return x

#パラメータ取得
def quantize_model_params(model):
    state_dict = dict()
    model = torch.quantization.convert(model)

    for layer_idx in range(1, 7):
        weight = model.__getattr__(f"layer{layer_idx}").weight()
        #s_w = model.__getattr__(f"layer{layer_idx}").weight().q_per_channel_scales().numpy()
        #z_w = model.__getattr__(f"layer{layer_idx}").weight().q_per_channel_zero_points().numpy()
        s_w = model.__getattr__(f"layer{layer_idx}").weight().q_scale()
        z_w = model.__getattr__(f"layer{layer_idx}").weight().q_zero_point()
        s_y = model.__getattr__(f"layer{layer_idx}").scale
        z_y = model.__getattr__(f"layer{layer_idx}").zero_point
        if layer_idx == 1:
            s_x = model.__getattr__(f"quant{layer_idx}").scale.numpy()
            z_x = model.__getattr__(f"quant{layer_idx}").zero_point.numpy()
        else:
            s_x = model.__getattr__(f"layer{layer_idx-1}").scale
            z_x = model.__getattr__(f"layer{layer_idx-1}").zero_point


        state_dict[f'layer_{layer_idx}_weight'] = torch.int_repr(weight).T.numpy()
        state_dict[f'layer_{layer_idx}_s_x_inv'] = 1 / s_x
        state_dict[f'layer_{layer_idx}_z_w'] = z_w
        state_dict[f'layer_{layer_idx}_z_x'] = z_x
        state_dict[f'layer_{layer_idx}_s_y'] = s_y
        state_dict[f'layer_{layer_idx}_z_y'] = z_y
        state_dict[f'layer_{layer_idx}_S'] = s_x * s_w / s_y


    return state_dict

#出力確認
def output_params(save_dir, filename, inp):
    saved_stats = torch.load(os.path.join(save_dir, filename))
    state_dict = saved_stats['state_dict']
    input = inp
    quant_input= 0
    output = 0
    for layer_idx in range(1, 7):
        name = f'layer_{layer_idx}_s_x_inv'
        s_x_inv = state_dict[name]

        name = f'layer_{layer_idx}_z_x'
        z_x = state_dict[name]

        name = f'layer_{layer_idx}_z_w'
        z_w = state_dict[name]

        name = f'layer_{layer_idx}_s_y'
        s_y = state_dict[name]

        name = f'layer_{layer_idx}_z_y'
        z_y = state_dict[name]

        name = f'layer_{layer_idx}_S'
        s = state_dict[name] 

        name = f'layer_{layer_idx}_weight'
        w = state_dict[name]
        if layer_idx == 1:
            quant_input = np.round(input * s_x_inv ) + z_x
        else:
            quant_input = torch.from_numpy(output)
        output = np.round(s * np.matmul(quant_input - z_x, w - z_w).numpy()) + z_y
        if layer_idx < 4:
            output = nn.functional.relu(torch.tensor(output)).numpy()
        print('input',layer_idx,quant_input)
        print('output',layer_idx,output,(output-z_y)*s_y)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

#パラメータ書き込み
def create_params(save_dir, filename):
    saved_stats = torch.load(os.path.join(save_dir, filename))
    state_dict = saved_stats['state_dict']
    threshold = saved_stats['threshold']
    data_min = saved_stats['data_min']
    data_max = saved_stats['data_max']
    ret = {}

    with open('mlp_params.c', 'w') as f:
        f.write(f"#include \"mlp_params.h\"\n\n")
        for layer_idx in range(1, 7):
            name = f'layer_{layer_idx}_s_x_inv'
            fxp_value = (state_dict[name] * (2**16))
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            ret[name] = []
            ret[name].append(int(fxp_value))

            name = f'layer_{layer_idx}_z_x'
            fxp_value = (state_dict[name])
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            ret[name] = []
            ret[name].append(int(fxp_value))

            name = f'layer_{layer_idx}_z_w'
            # fxp_value = (state_dict[name]).round()
            # f.write(f"const int {name}[{len(fxp_value)}] = {{")
            # for idx in range(len(fxp_value)):
            #     f.write(f"{int(fxp_value[idx])}")
            #     if idx < len(fxp_value) - 1:
            #         f.write(", ")
            # f.write("};\n\n")
            fxp_value = state_dict[name]
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            ret[name] = []
            ret[name].append(int(fxp_value))

            name = f'layer_{layer_idx}_s_y'
            fxp_value = (state_dict[name] * (2**16))
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            ret[name] = []
            ret[name].append(int(fxp_value))

            name = f'layer_{layer_idx}_z_y'
            fxp_value = (state_dict[name])
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            ret[name] = []
            ret[name].append(int(fxp_value))

            name = f'layer_{layer_idx}_S'
            fxp_value = (state_dict[name] * (2**16))
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            # fxp_value = (state_dict[name] * (2**16)).round()
            # f.write(f"const int {name}[{len(fxp_value)}] = {{")
            # for idx in range(len(fxp_value)):
            #     f.write(f"{int(fxp_value[idx])}")
            #     if idx < len(fxp_value) - 1:
            #         f.write(", ")
            # f.write("};\n\n")
            ret[name] = []
            ret[name].append(int(fxp_value))


        for layer_idx in range(1, 7):
            name = f'layer_{layer_idx}_weight'
            ret[name] = []
            param = state_dict[name].flatten()
            f.write(f"const int8_t {name}[{len(param)}] = {{")
            for idx in range(len(param)):
                f.write(f"{param[idx]}")
                ret[name].append(param[idx])
                if idx < len(param) - 1:
                    f.write(", ")
            f.write("};\n\n")
        
        name = 'data_min'
        ret[name] = []
        fxp_value = (data_min * (2**16)).round()
        f.write(f"const int64_t {name}[{len(fxp_value)}] = {{")
        for idx in range(len(fxp_value)):
            f.write(f"{int(fxp_value[idx])}")
            ret[name].append(int(fxp_value[idx]))
            if idx < len(fxp_value) - 1:
                f.write(", ")
        f.write("};\n\n")

        name = 'data_max'
        ret[name] = []
        fxp_value = (data_max * (2**16)).round()
        f.write(f"const int64_t {name}[{len(fxp_value)}] = {{")
        for idx in range(len(fxp_value)):
            f.write(f"{int(fxp_value[idx])}")
            ret[name].append(int(fxp_value[idx]))
            if idx < len(fxp_value) - 1:
                f.write(", ")
        f.write("};\n\n")
        
        name = 'threshold'
        fxp_value = (threshold * (2**16)).round()
        f.write(f"const int {name} = {int(fxp_value)};\n\n")
        ret[name] = []
        ret[name].append(int(fxp_value))
    
    with open("./json/mlp_params.json", "w") as f:
        json.dump(ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '), cls=NpEncoder)



if __name__ == '__main__':
    #setting commandline argument

    parser = argparse.ArgumentParser(description="Script for training a model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='dataset')
    parser.add_argument('--maxLength', type=int, default=100, help='max length')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=5)
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--fold', type=int, default=0, help='fold to use')
    parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
    parser.add_argument('--save_dir', help='save directory', default='./saved_models')
    parser.add_argument('--filename', help='filename', type=str, default='mlp_pktflw.th')
    parser.add_argument('--backends', help='backend name', type=str, default='fbgemm')

    


    # setting random seed
    args = parser.parse_args()
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.backends.quantized.engine = args.backends

    scaler = MinMaxScaler()

    # open dataset
    with open (os.path.join(args.data_dir, 'flowsCIC.pickle'), "rb") as f:
        all_data = pickle.load(f)

    # shuffle dataset
    all_data = [item[:args.maxLength,:] for item in all_data if np.all(item[:,4]>=0)]
    random.shuffle(all_data)

    # setting dataset
    x = [item[:, :-2] for item in all_data]
    y = [item[:, -1:] for item in all_data]
    categories = [item[:, -2:-1] for item in all_data]

    dataset = PacketDataset(x, y, categories)

    # 訓練データとテストデータの分割
    n_fold = args.nFold
    fold = args.fold
    train_indices, test_indices = get_nth_split(dataset, n_fold, fold)
    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    # convert dataset
    x_train, y_train = get_dataset(train_data, True)
    x_train = scaler.fit_transform(x_train)
    pickle.dump(scaler, open(f'{args.save_dir}/scaler_qat.sav', 'wb'))
    x_test, y_test = get_dataset(test_data, False)
    scaler = pickle.load(open(f'{args.save_dir}/scaler_qat.sav', 'rb'))
    x_test = scaler.transform(x_test)

    data_min = scaler.data_min_
    data_max = scaler.data_max_

    train_trainset = PacketFlowDataset(x_train, x_train)
    test_testset  = PacketFlowDataset(x_test, x_test)


    #トレーニングデータと検証データの分割
    split_r = args.train_val_split
    train_trainset, train_valset = random_split(train_trainset, [round(len(train_trainset)*split_r), round(len(train_trainset)*(1 - split_r))], generator=torch.Generator().manual_seed(SEED))

    #データローダー
    train_loader = DataLoader(train_trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(train_valset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_testset, batch_size=1, shuffle=False)

    # モデルのインスタンス化
    model_fp32 = MyModel().to('cpu')

    #量子化設定（QConfig）の取得
    model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig(args.backends)
    print(model_fp32.qconfig)
    # https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/qconfig.py#L374
    my_qconfig = torch.ao.quantization.qconfig.QConfig(
            activation=torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(observer=torch.ao.quantization.observer.MovingAverageMinMaxObserver,
                                                                                                               quant_min=0,
                                                                                                               quant_max=255,
                                                                                                               reduce_range=False),
            weight=torch.ao.quantization.fake_quantize.default_fused_wt_fake_quant)
            # weight=torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, quant_min=-127, quant_max=127))
    print(my_qconfig)
    model_fp32.qconfig = my_qconfig

    #relu関数の融合
    model_fp32_fused = torch.ao.quantization.fuse_modules_qat(model_fp32,  ['layer1',  'relu1'])
    model_fp32_fused = torch.ao.quantization.fuse_modules_qat(model_fp32_fused,  ['layer2',  'relu2'])
    model_fp32_fused = torch.ao.quantization.fuse_modules_qat(model_fp32_fused,  ['layer3',  'relu3'])


    # QATの準備
    model_fp32_fused.train()
    model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused.train())

    # 損失関数とオプティマイザの定義
    loss_fnc = nn.MSELoss()
    optimizer = optim.Adam(model_fp32_prepared.parameters())

    # モデルのトレーニング
    print('Training')
    all_preds = []
    all_targets = []
    model_fp32_prepared.train()
    for epoch in range(args.num_epochs):
        num_batches = len(train_loader)
        loss = 0
        for samples, y in train_loader:
            optimizer.zero_grad()
            preds = model_fp32_prepared(samples)
            batch_loss = loss_fnc(preds, samples)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        train_loss = loss / num_batches
        val_loss = eval_epoch(model_fp32_prepared, val_loader, loss_fnc)
        print(f"Epoch: {epoch  + 1} - train loss: {train_loss:.5f} validation loss: {val_loss:.5f}")
    model_fp32_prepared.eval()

    model_int8 = torch.ao.quantization.convert(model_fp32_prepared.eval())

    #threshold
    for samples, y in train_loader:
        preds = model_int8(samples)
        all_targets += [target for target in samples.detach().numpy().copy()]
        all_preds += [pred for pred in preds.detach().numpy().copy()]
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    mseN = np.mean(np.power(all_targets - all_preds, 2), axis=1)
    eta = 0.2
    threshold = np.mean(mseN) +  np.std(mseN)*eta

    #モデルパラメータの保存
    name = args.filename.replace('.th', '_qat.th')
    name_int8 = args.filename.replace('.th', '_qat_int8.th')
    name_int8_c = args.filename.replace('.th', '_qat_int8_c.th')
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({'state_dict': model_fp32_prepared.state_dict(),
                'threshold': threshold,
                'mean': np.mean(mseN),
                'std': np.std(mseN),
                'eta':eta
                },
               f"{args.save_dir}/{name}")

    torch.save({'state_dict': model_int8.state_dict(),
                'threshold': threshold,
                'mean': np.mean(mseN),
                'std': np.std(mseN),
                'eta':eta
                },
               f"{args.save_dir}/{name_int8}")

    state_dict = quantize_model_params(model_int8)
    torch.save({'state_dict': state_dict,
            'threshold': threshold,
            'mean': np.mean(mseN),
            'std': np.std(mseN),
            'eta':eta,
            'data_min':data_min,
            'data_max':data_max
            },
            f"{args.save_dir}/{name_int8_c}")


    torch.jit.save(torch.jit.script(model_int8), f"{args.save_dir}/int8_qat_model.pt")
    model_int8 = torch.jit.load(f"{args.save_dir}/int8_qat_model.pt")

    #モデルの推論評価
    all_preds = []
    all_targets = []
    print('Evaluate model on test data')
    inp = []
    firstLoop = True
    total_time = []
    model_int8.eval()
    with torch.no_grad():
        for samples, targets in tqdm(test_loader):
            if firstLoop:
                inp = targets
                firstLoop = False
            start_time = time.perf_counter_ns()
            preds = model_int8(samples)
            end_time = time.perf_counter_ns()
            all_targets += [target for target in samples.detach().numpy().copy()]
            all_preds += [pred for pred in preds.detach().numpy().copy()]
            # 推論時間を加算
            total_time.append(end_time - start_time)
    total_time = np.array(total_time)
    total_time = total_time / 1e6  # ミリ秒に変換

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    print('act',model_int8(inp))


    # テストデータを用いて再構築誤差を計算
    mseN_test = np.mean(np.power(all_targets - all_preds, 2), axis=1)
    # テストデータを用いて再構築誤差を計算
    eta_list = [-0.2, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.5, 1.75, 2, 2.5]
    for eta in eta_list:
        threshold = np.mean(mseN) +  np.std(mseN)*eta
        anomalies1 = 1 / (1 + np.exp(-1 * (mseN_test - threshold)))
        anomalies2 = mseN_test > threshold
        print("Threshold: ", eta, threshold, roc_auc_score(y_test, anomalies1))
        print(classification_report(y_test, anomalies2, digits=4))
    anomalies2 = mseN_test > threshold

    # #出力確認
    output_params(args.save_dir, "mlp_pktflw_qat_int8_c.th", inp)
    #
    # #モデルパラメータのmlp_params.cへの出力
    create_params(args.save_dir, "mlp_pktflw_qat_int8_c.th")





