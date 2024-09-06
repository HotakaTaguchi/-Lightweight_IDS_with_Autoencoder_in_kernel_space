import numpy as np
import argparse
import random
import os
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score,roc_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pytorch_encoder import MyModel, PacketDataset, PacketFlowDataset, train_epoch ,eval_epoch, get_nth_split, get_dataset, print_size_of_model, get_size_of_model
import time
import pandas as pd




if __name__ == '__main__':
    #setting commandline argument

    parser = argparse.ArgumentParser(description="Script for training a model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='dataset')
    parser.add_argument('--maxLength', type=int, default=100, help='max length')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=5)
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', help='batch size', type=int, default=50)
    parser.add_argument('--fold', type=int, default=0, help='fold to use')
    parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
    parser.add_argument('--save_dir', help='save directory', default='./saved_models')
    parser.add_argument('--filename', help='filename', type=str, default='mlp_pktflw.th')



    # setting random seed
    args = parser.parse_args()
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
    pickle.dump(scaler, open(f'./{args.save_dir}/scaler_float.sav', 'wb'))
    x_test, y_test = get_dataset(test_data, False)
    scaler = pickle.load(open(f'./{args.save_dir}/scaler_float.sav', 'rb'))
    x_test = scaler.transform(x_test)

    train_trainset = PacketFlowDataset(x_train, x_train)
    test_testset  = PacketFlowDataset(x_test, x_test)


    #トレーニングデータと検証データの分割
    split_r = args.train_val_split
    train_trainset, train_valset = random_split(train_trainset, [round(len(train_trainset)*split_r), round(len(train_trainset)*(1 - split_r))], generator=torch.Generator().manual_seed(SEED))

    # モデルのインスタンス化
    model = MyModel().to('cpu')

    # 損失関数とオプティマイザの定義
    loss_fnc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    #データローダー
    train_loader = DataLoader(train_trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(train_valset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_testset, batch_size=1, shuffle=False)

    # モデルのトレーニング
    print('Training')
    all_preds = []
    all_targets = []
    for epoch in range(args.num_epochs):
        model.train(mode=True)
        num_batches = len(train_loader)
        loss = 0
        for samples, y in train_loader:
            optimizer.zero_grad()
            preds = model(samples)
            batch_loss = loss_fnc(preds, samples)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        train_loss = loss / num_batches
        val_loss = eval_epoch(model, val_loader, loss_fnc)
        print(f"Epoch: {epoch  + 1} - train loss: {train_loss:.5f} validation loss: {val_loss:.5f}")
    
    #threashold
    model.eval()
    for samples, y in train_loader:
            preds = model(samples)
            all_targets += [target for target in samples.detach().numpy().copy()]
            all_preds += [pred for pred in preds.detach().numpy().copy()]
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    mseN = np.mean(np.power(all_targets - all_preds, 2), axis=1)

    eta = 0.2
    threshold = np.mean(mseN) +  np.std(mseN)*eta

    #モデルパラメータの保存
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(),
                'threshold': threshold,
                'mean': np.mean(mseN),
                'std': np.std(mseN),
                'eta': eta
                },
               f"{args.save_dir}/{args.filename}")
    torch.jit.save(torch.jit.script(model), f"{args.save_dir}/fp32_model.pt")
    model.eval()

