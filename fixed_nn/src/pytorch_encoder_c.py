import numpy as np
import argparse
import random
import os
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score,roc_curve, roc_auc_score
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_encoder import MyModel, PacketDataset, PacketFlowDataset, train_epoch ,eval_epoch, get_nth_split, get_dataset, print_size_of_model, get_size_of_model
from pytorch_encoder_c_nn import load_c_lib, run_mlp

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
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
    parser.add_argument('--filename', help='filename', type=str, default='mlp_pktflw_qat_int8_c.th')


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
    pickle.dump(scaler, open(f'{args.save_dir}/scaler_float.sav', 'wb'))
    x_test, y_test = get_dataset(test_data, False)
    print(x_test[0])
    scaler = pickle.load(open(f'{args.save_dir}/scaler_float.sav', 'rb'))
    x_test = scaler.transform(x_test)
    print(x_test[0])

    train_trainset = PacketFlowDataset(x_train, x_train)
    test_testset  = PacketFlowDataset(x_test, x_test)

    #データローダー
    test_loader = DataLoader(test_testset, batch_size=1, shuffle=False)

    # load c library
    c_lib = load_c_lib(library='./mlp.so')

    acc = 0
    all_preds = []
    all_targets = []
    li_elapsed = []
    for samples, targets in tqdm(test_loader):
        samples = (samples * (2 ** 16)).round() # convert to fixed-point 16
        x, preds, elapsed  = run_mlp(samples, c_lib)
        li_elapsed.append(elapsed)
        all_targets.append(x)
        all_preds.append(preds)

    all_targets = np.array(all_targets) / (2**16)
    all_preds = np.array(all_preds) / (2**16)

    # テストデータを用いて再構築誤差を計算
    mseN_test = np.mean(np.power(all_targets - all_preds, 2), axis=1)
    saved_stats = torch.load(os.path.join(args.save_dir, args.filename))
    threshold = saved_stats['threshold']
    anomalies1 = 1 / (1 + np.exp(-1 * (mseN_test - threshold)))
    anomalies2 = mseN_test > threshold
    #print("auc: ", roc_auc_score(y_test, anomalies1))
    #print(classification_report(y_test, anomalies2, digits=4))
