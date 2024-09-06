import numpy as np
import argparse
import random
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score,roc_curve, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import socket
import struct
import time

# CPU コア 0 にプロセスをバインドする
os.sched_setaffinity(0, {0})

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

def parse_packet(packet):
    # Ethernet header (14 bytes)
    eth_header = packet[:14]
    eth_unpacked = struct.unpack('!6s6sH', eth_header)
    eth_protocol = socket.ntohs(eth_unpacked[2])

    # IP header (20 bytes)
    if eth_protocol == 8:  # IP Protocol
        ip_header = packet[14:34]
        ip_unpacked = struct.unpack('!BBHHHBBH4s4s', ip_header)
        protocol = ip_unpacked[6]
        src_ip = socket.inet_ntoa(ip_unpacked[8])
        dst_ip = socket.inet_ntoa(ip_unpacked[9])

        # TCP or UDP
        if protocol == 6:  # TCP
            tcp_header = packet[34:54]
            tcp_unpacked = struct.unpack('!HHLLBBHHH', tcp_header)
            src_port = tcp_unpacked[0]
            dst_port = tcp_unpacked[1]
            packet_length = len(packet)
            return (src_port, dst_port, protocol, packet_length)
        
        #elif protocol == 17:  # UDP
        #    udp_header = packet[34:42]
        #    udp_unpacked = struct.unpack('!HHHH', udp_header)
        #    src_port = udp_unpacked[0]
        #    dst_port = udp_unpacked[1]
        #    packet_length = len(packet)
        #    return (src_port, dst_port, protocol, packet_length)

    return None

if __name__ == '__main__':
    #setting commandline argument

    parser = argparse.ArgumentParser(description="Script for training a model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    try:
        scaler = MinMaxScaler()
        scaler = pickle.load(open(f'{args.save_dir}/scaler_float.sav', 'rb'))

        conn = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
        conn.bind(('eno1', 0))

        last_packet_time = None
        last_packet_sport = None
        packet_num = 0
        average = np.zeros(3)
        deviation = np.zeros(3)
        ret = []

        # #モデルのロード
        saved_stats = torch.load(os.path.join(args.save_dir, args.filename))
        threshold = saved_stats['threshold']
        #mean = saved_stats['mean']
        #std = saved_stats['std']
        #eta = saved_stats['eta']
        #threshold = mean +  std * eta
        model = torch.jit.load(f"{args.save_dir}/fp32_model.pt")

        t_end1 = time.time() + 10
        t_end2 = time.time() + 1
        while time.time() < t_end1:
            packet, addr = conn.recvfrom(65565)
            current_time = time.time()
            
            packet_info = parse_packet(packet)
            if packet_info:
                sport, dport, protocol, packet_length = packet_info
                #if dport != 12345:
                #    continue
                if last_packet_time == None:
                    last_packet_time = current_time
                    last_packet_sport = sport
                interval = current_time - last_packet_time
                direction = (last_packet_sport == sport) 

                #print(f"送信ポート番号: {sport}")
                #print(f"受信ポート番号: {dport}")
                #print(f"プロトコル番号: {protocol}")
                #print(f"パケット長: {packet_length}")
                #print(f"方向: {direction}")
                #print(f"到着間隔時間: {interval:.6f} 秒")
                #print("-" * 50)
                
                last_packet_time = current_time
                last_packet_sport = sport

                x_test = []
                current_vector = np.array([ sport, dport, protocol, packet_length, interval, direction])
                average += current_vector[3:]
                current_average = (average/(packet_num + 1))
                deviation += np.abs(current_vector[3:]-current_average)
                current_deviation = (deviation/(packet_num + 1))
                final_vector = np.concatenate((current_vector, current_average, current_deviation))
                x_test.append(final_vector)

                x_test = scaler.transform(x_test)
                test_testset  = PacketFlowDataset(x_test, x_test)
                test_loader = DataLoader(test_testset, batch_size=1, shuffle=False)

                #モデルの推論評価
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for samples, targets in test_loader:
                        preds = model(samples.float())
                        all_targets += [target for target in samples.detach().numpy().copy()]
                        all_preds += [pred for pred in preds.detach().numpy().copy()]
                all_targets = np.array(all_targets)
                all_preds = np.array(all_preds)


                # テストデータを用いて再構築誤差を計算
                mseN = np.mean(np.power(all_targets - all_preds, 2), axis=1)
                #if mseN > threshold:
                    #print("drop")
                #else:
                    #print("pass")
                packet_num += 1
                if time.time() > t_end2:
                    t_end2 = time.time() + 1
                    ret.append(packet_num)
                    packet_num = 0
    finally:
        resdir = "./log"
        filename = f"{resdir}/rxpps_float.log"
        with open (filename, 'w') as f:
            for d in ret:
                f.write(f"{d}\n")    
        #pps = [1.2, 5.21, 36.22, 71.37, 100, 100, 100]
        #file_path = 'csv/cpu_usage_nomodel2.csv'
        #df = pd.read_csv('csv/cpu_usage_nomodel1.csv')
        #df['user'] = pps
        #df.to_csv(file_path, index=False)




  
