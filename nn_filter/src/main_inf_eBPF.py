#!/usr/bin/python3
# -*- coding: utf-8 -*-

from bcc import BPF
from bcc import lib, table
from pyroute2 import IPRoute
import sys
import time
import json
from socket import inet_ntop, ntohs, AF_INET, AF_INET6
from struct import pack
import ctypes as ct
import joblib
from datetime import datetime
import pandas as pd

def usage():
    print("Usage: {0} <ifdev> <output-dir> <flag>".format(sys.argv[0]))
    exit(1)

ipr = IPRoute()

bpf_text = """
#include <uapi/linux/bpf.h>
#include <linux/inet.h>
#include <linux/pkt_cls.h>
#include <linux/ip.h>
#include <uapi/linux/tcp.h>
#include <uapi/linux/udp.h>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define UINT8_MAX_VALUE 255
#define UINT8_MIN_VALUE 0
#define FXP_VALUE 16
#define ROUND_CONST (1 << (FXP_VALUE - 1))
#define N 1
#define INPUT_DIM 12
#define H1 10
#define H2 8
#define H3 6
#define H4 8
#define H5 10
#define H6 12

struct pkt_key_t {
  u32 protocol;
  u32 saddr;
  u32 daddr;
  u32 sport;
  u32 dport;
};

struct pkt_leaf_t {
  u32 num_packets;
  u64 last_packet_timestamp;
  u32 sport;
  u32 dport;
  u64 features[6];
};


BPF_TABLE("lru_hash", struct pkt_key_t, struct pkt_leaf_t, sessions, 1024);
BPF_PROG_ARRAY(jmp_table, 6);
BPF_PERCPU_ARRAY(out_input, int64_t, H6);
BPF_PERCPU_ARRAY(in_input, int64_t, INPUT_DIM);
BPF_ARRAY(layer_1_weight, int, LAYER_1_WEIGHT);
BPF_ARRAY(layer_2_weight, int, LAYER_2_WEIGHT);
BPF_ARRAY(layer_3_weight, int, LAYER_3_WEIGHT);
BPF_ARRAY(layer_4_weight, int, LAYER_4_WEIGHT);
BPF_ARRAY(layer_5_weight, int, LAYER_5_WEIGHT);
BPF_ARRAY(layer_6_weight, int, LAYER_6_WEIGHT);
BPF_ARRAY(layer_1_s_x_inv, int, 1);
BPF_ARRAY(layer_2_s_x_inv, int, 1);
BPF_ARRAY(layer_3_s_x_inv, int, 1);
BPF_ARRAY(layer_4_s_x_inv, int, 1);
BPF_ARRAY(layer_5_s_x_inv, int, 1);
BPF_ARRAY(layer_6_s_x_inv, int, 1);
BPF_ARRAY(layer_1_z_x, int, 1);
BPF_ARRAY(layer_2_z_x, int, 1);
BPF_ARRAY(layer_3_z_x, int, 1);
BPF_ARRAY(layer_4_z_x, int, 1);
BPF_ARRAY(layer_5_z_x, int, 1);
BPF_ARRAY(layer_6_z_x, int, 1);
BPF_ARRAY(layer_1_z_w, int, 1);
BPF_ARRAY(layer_2_z_w, int, 1);
BPF_ARRAY(layer_3_z_w, int, 1);
BPF_ARRAY(layer_4_z_w, int, 1);
BPF_ARRAY(layer_5_z_w, int, 1);
BPF_ARRAY(layer_6_z_w, int, 1);
BPF_ARRAY(layer_1_s_y, int, 1);
BPF_ARRAY(layer_2_s_y, int, 1);
BPF_ARRAY(layer_3_s_y, int, 1);
BPF_ARRAY(layer_4_s_y, int, 1);
BPF_ARRAY(layer_5_s_y, int, 1);
BPF_ARRAY(layer_6_s_y, int, 1);
BPF_ARRAY(layer_1_z_y, int, 1);
BPF_ARRAY(layer_2_z_y, int, 1);
BPF_ARRAY(layer_3_z_y, int, 1);
BPF_ARRAY(layer_4_z_y, int, 1);
BPF_ARRAY(layer_5_z_y, int, 1);
BPF_ARRAY(layer_6_z_y, int, 1);
BPF_ARRAY(layer_1_S, int, 1);
BPF_ARRAY(layer_2_S, int, 1);
BPF_ARRAY(layer_3_S, int, 1);
BPF_ARRAY(layer_4_S, int, 1);
BPF_ARRAY(layer_5_S, int, 1);
BPF_ARRAY(layer_6_S, int, 1);
BPF_ARRAY(threshold, int, 1);
BPF_ARRAY(data_min, int64_t, DATA_MIN);
BPF_ARRAY(data_max, int64_t, DATA_MAX);
BPF_HASH(dropcnt, int, u32);

int mse_cal(struct __sk_buff *skb) {
  unsigned int i, k, m, _i, _k, _m;
  int64_t in, out, dif;
  int dif_int, dif_frac, mse_value, th;
  uint32_t mse = 0;
  int _zero = 0;
  int64_t _zero64 = 0;
  
  // mse
  for (i = 0; i < H6; i++) {
    _i = i;
    in = *in_input.lookup_or_init(&_i, &_zero64);
    out = *out_input.lookup_or_init(&_i, &_zero64);
    //bpf_trace_printk("outputB,%d:%lld",i,out);
    dif = out - in;
    dif_int = (dif + ROUND_CONST) >> FXP_VALUE;
    dif_frac = dif - (dif_int << FXP_VALUE);
    mse_value = dif_int*dif_frac*2;
    mse_value += (dif_frac*dif_frac + ROUND_CONST) >> FXP_VALUE;
    mse_value +=  dif_int*dif_int << FXP_VALUE;
    mse += mse_value;
  }
  mse /= H6;
  //bpf_trace_printk("%u",mse);
  th = *threshold.lookup_or_init(&_zero, &_zero);
  u32 val = 0, *vp;
  vp = dropcnt.lookup_or_init(&_zero, &val);
  *vp += 1;
  if(th < mse){
    //return TC_ACT_SHOT;
    return TC_ACT_OK;
  }
  return TC_ACT_OK;
}

int nn5(struct __sk_buff *skb) {
  unsigned int i, k, m, _i, _k, _m;
  int64_t input, out, accumulator;
  int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac,  s_x_inv, z_x, z_w, scale_factor, z_y, s_y;
  int8_t weight;
  uint8_t y_q;
  int x_q[INPUT_DIM];
  int _zero = 0;
  int64_t _zero64 = 0;
  
  // quantize
  s_x_inv = *layer_6_s_x_inv.lookup_or_init(&_zero, &_zero);
  z_x = *layer_6_z_x.lookup_or_init(&_zero, &_zero);
  scale_factor_int = (s_x_inv + ROUND_CONST) >> FXP_VALUE;
  scale_factor_frac = s_x_inv - (scale_factor_int << FXP_VALUE);
  for (i = 0; i < H5; i++) {
    _i = i;
    input = *out_input.lookup_or_init(&_i, &_zero64);
    tensor_int = (input + ROUND_CONST) >> FXP_VALUE;
    tensor_frac = input - (tensor_int << FXP_VALUE);
    rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
    rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
    rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int + z_x;
    if (rounded_value > UINT8_MAX_VALUE){
      x_q[i] = (int)UINT8_MAX_VALUE;
    }else if(rounded_value < UINT8_MIN_VALUE){
      x_q[i] = (int)UINT8_MIN_VALUE; 
    }else{
      x_q[i] = (int)rounded_value;
    }
  }

  // mat_mult
  for (m = 0; m < H6; m++) {
    accumulator = 0;
    z_x = *layer_6_z_x.lookup_or_init(&_zero, &_zero);
    z_w = *layer_6_z_w.lookup_or_init(&_zero, &_zero);
    for (k = 0; k < H5; k++) {
      _k = k*H6 + m;
      weight = *(int8_t*)layer_6_weight.lookup_or_init(&_k, &_zero);
      accumulator += ((int64_t)x_q[k] - z_x)  * ((int64_t)weight - z_w);
    }

    _m = m;
    scale_factor = *layer_6_S.lookup_or_init(&_zero, &_zero);
    accumulator *= scale_factor;
    accumulator = ((accumulator + ROUND_CONST) >> FXP_VALUE);
    z_y = *layer_6_z_y.lookup_or_init(&_zero, &_zero);
    accumulator += z_y;
    out = (int64_t)accumulator;
    
    //relu
    out = MAX(out, 0);
    
    //y_cast
    if (out > UINT8_MAX_VALUE){
      y_q = (uint8_t)UINT8_MAX_VALUE;
    }else if(out < UINT8_MIN_VALUE){
      y_q = (uint8_t)UINT8_MIN_VALUE; 
    }else{
      y_q = (uint8_t)out;
    }
    
    //dequantize
    s_y = *layer_6_s_y.lookup_or_init(&_zero, &_zero);
    out =  (int64_t)y_q - z_y;
    out = out * s_y;
    //bpf_trace_printk("outputA,%d:%lld",m,out);
    out_input.update(&_m, &out);
  }
  jmp_table.call(skb, 5);
  return TC_ACT_OK;
}


int nn4(struct __sk_buff *skb) {
  unsigned int i, k, m, _i, _k, _m;
  int64_t input, out, accumulator;
  int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac,  s_x_inv, z_x, z_w, scale_factor, z_y, s_y;
  int8_t weight;
  uint8_t y_q;
  int x_q[INPUT_DIM];
  int _zero = 0;
  int64_t _zero64 = 0;
  
  // quantize
  s_x_inv = *layer_5_s_x_inv.lookup_or_init(&_zero, &_zero);
  z_x = *layer_5_z_x.lookup_or_init(&_zero, &_zero);
  scale_factor_int = (s_x_inv + ROUND_CONST) >> FXP_VALUE;
  scale_factor_frac = s_x_inv - (scale_factor_int << FXP_VALUE);
  for (i = 0; i < H4; i++) {
    _i = i;
    input = *out_input.lookup_or_init(&_i, &_zero64);
    tensor_int = (input + ROUND_CONST) >> FXP_VALUE;
    tensor_frac = input - (tensor_int << FXP_VALUE);
    rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
    rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
    rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int + z_x;
    if (rounded_value > UINT8_MAX_VALUE){
      x_q[i] = (int)UINT8_MAX_VALUE;
    }else if(rounded_value < UINT8_MIN_VALUE){
      x_q[i] = (int)UINT8_MIN_VALUE; 
    }else{
      x_q[i] = (int)rounded_value;
    }
  }

  // mat_mult
  for (m = 0; m < H5; m++) {
    accumulator = 0;
    z_x = *layer_5_z_x.lookup_or_init(&_zero, &_zero);
    z_w = *layer_5_z_w.lookup_or_init(&_zero, &_zero);
    for (k = 0; k < H4; k++) {
      _k = k*H5 + m;
      weight = *(int8_t*)layer_5_weight.lookup_or_init(&_k, &_zero);
      accumulator += ((int64_t)x_q[k] - z_x)  * ((int64_t)weight - z_w);
    }

    _m = m;
    scale_factor = *layer_5_S.lookup_or_init(&_zero, &_zero);
    accumulator *= scale_factor;
    accumulator = ((accumulator + ROUND_CONST) >> FXP_VALUE);
    z_y = *layer_5_z_y.lookup_or_init(&_zero, &_zero);
    accumulator += z_y;
    out = (int64_t)accumulator;
    
    //relu
    out = MAX(out, 0);
    
    //y_cast
    if (out > UINT8_MAX_VALUE){
      y_q = (uint8_t)UINT8_MAX_VALUE;
    }else if(out < UINT8_MIN_VALUE){
      y_q = (uint8_t)UINT8_MIN_VALUE; 
    }else{
      y_q = (uint8_t)out;
    }
    
    //dequantize
    s_y = *layer_5_s_y.lookup_or_init(&_zero, &_zero);
    out =  (int64_t)y_q - z_y;
    out = out * s_y;
    out_input.update(&_m, &out);
  }
  jmp_table.call(skb, 4);
  return TC_ACT_OK;
}


int nn3(struct __sk_buff *skb) {
  unsigned int i, k, m, _i, _k, _m;
  int64_t input, out, accumulator;
  int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac,  s_x_inv, z_x, z_w, scale_factor, z_y, s_y;
  int8_t weight;
  uint8_t y_q;
  int x_q[INPUT_DIM];
  int _zero = 0;
  int64_t _zero64 = 0;
  
  // quantize
  s_x_inv = *layer_4_s_x_inv.lookup_or_init(&_zero, &_zero);
  z_x = *layer_4_z_x.lookup_or_init(&_zero, &_zero);
  scale_factor_int = (s_x_inv + ROUND_CONST) >> FXP_VALUE;
  scale_factor_frac = s_x_inv - (scale_factor_int << FXP_VALUE);
  for (i = 0; i < H3; i++) {
    _i = i;
    input = *out_input.lookup_or_init(&_i, &_zero64);
    tensor_int = (input + ROUND_CONST) >> FXP_VALUE;
    tensor_frac = input - (tensor_int << FXP_VALUE);
    rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
    rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
    rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int + z_x;
    if (rounded_value > UINT8_MAX_VALUE){
      x_q[i] = (int)UINT8_MAX_VALUE;
    }else if(rounded_value < UINT8_MIN_VALUE){
      x_q[i] = (int)UINT8_MIN_VALUE; 
    }else{
      x_q[i] = (int)rounded_value;
    }
  }

  // mat_mult
  for (m = 0; m < H4; m++) {
    accumulator = 0;
    z_x = *layer_4_z_x.lookup_or_init(&_zero, &_zero);
    z_w = *layer_4_z_w.lookup_or_init(&_zero, &_zero);
    for (k = 0; k < H3; k++) {
      _k = k*H4 + m;
      weight = *(int8_t*)layer_4_weight.lookup_or_init(&_k, &_zero);
      accumulator += ((int64_t)x_q[k] - z_x)  * ((int64_t)weight - z_w);
    }

    _m = m;
    scale_factor = *layer_4_S.lookup_or_init(&_zero, &_zero);
    accumulator *= scale_factor;
    accumulator = ((accumulator + ROUND_CONST) >> FXP_VALUE);
    z_y = *layer_4_z_y.lookup_or_init(&_zero, &_zero);
    accumulator += z_y;
    out = (int64_t)accumulator;
    
    //relu
    out = MAX(out, 0);
    
    //y_cast
    if (out > UINT8_MAX_VALUE){
      y_q = (uint8_t)UINT8_MAX_VALUE;
    }else if(out < UINT8_MIN_VALUE){
      y_q = (uint8_t)UINT8_MIN_VALUE; 
    }else{
      y_q = (uint8_t)out;
    }
    
    //dequantize
    s_y = *layer_4_s_y.lookup_or_init(&_zero, &_zero);
    out =  (int64_t)y_q - z_y;
    out = out * s_y;
    out_input.update(&_m, &out);
  }
  jmp_table.call(skb, 3);
  return TC_ACT_OK;
}

int nn2(struct __sk_buff *skb) {
  unsigned int i, k, m, _i, _k, _m;
  int64_t input, out, accumulator;
  int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac,  s_x_inv, z_x, z_w, scale_factor, z_y, s_y;
  int8_t weight;
  uint8_t y_q;
  int x_q[INPUT_DIM];
  int _zero = 0;
  int64_t _zero64 = 0;
  
  // quantize
  s_x_inv = *layer_3_s_x_inv.lookup_or_init(&_zero, &_zero);
  z_x = *layer_3_z_x.lookup_or_init(&_zero, &_zero);
  scale_factor_int = (s_x_inv + ROUND_CONST) >> FXP_VALUE;
  scale_factor_frac = s_x_inv - (scale_factor_int << FXP_VALUE);
  for (i = 0; i < H2; i++) {
    _i = i;
    input = *out_input.lookup_or_init(&_i, &_zero64);
    tensor_int = (input + ROUND_CONST) >> FXP_VALUE;
    tensor_frac = input - (tensor_int << FXP_VALUE);
    rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
    rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
    rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int + z_x;
    if (rounded_value > UINT8_MAX_VALUE){
      x_q[i] = (int)UINT8_MAX_VALUE;
    }else if(rounded_value < UINT8_MIN_VALUE){
      x_q[i] = (int)UINT8_MIN_VALUE; 
    }else{
      x_q[i] = (int)rounded_value;
    }
  }

  // mat_mult
  for (m = 0; m < H3; m++) {
    accumulator = 0;
    z_x = *layer_3_z_x.lookup_or_init(&_zero, &_zero);
    z_w = *layer_3_z_w.lookup_or_init(&_zero, &_zero);
    for (k = 0; k < H2; k++) {
      _k = k*H3 + m;
      weight = *(int8_t*)layer_3_weight.lookup_or_init(&_k, &_zero);
      accumulator += ((int64_t)x_q[k] - z_x)  * ((int64_t)weight - z_w);
    }

    _m = m;
    scale_factor = *layer_3_S.lookup_or_init(&_zero, &_zero);
    accumulator *= scale_factor;
    accumulator = ((accumulator + ROUND_CONST) >> FXP_VALUE);
    z_y = *layer_3_z_y.lookup_or_init(&_zero, &_zero);
    accumulator += z_y;
    out = (int64_t)accumulator;
    
    //relu
    out = MAX(out, 0);
    
    //y_cast
    if (out > UINT8_MAX_VALUE){
      y_q = (uint8_t)UINT8_MAX_VALUE;
    }else if(out < UINT8_MIN_VALUE){
      y_q = (uint8_t)UINT8_MIN_VALUE; 
    }else{
      y_q = (uint8_t)out;
    }
    
    //dequantize
    s_y = *layer_3_s_y.lookup_or_init(&_zero, &_zero);
    out =  (int64_t)y_q - z_y;
    out = out * s_y;
    out_input.update(&_m, &out);
  }
  jmp_table.call(skb, 2);
  return TC_ACT_OK;
}


int nn1(struct __sk_buff *skb) {
  unsigned int i, k, m, _i, _k, _m;
  int64_t input, out, accumulator;
  int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac,  s_x_inv, z_x, z_w, scale_factor, z_y, s_y;
  int8_t weight;
  uint8_t y_q;
  int x_q[INPUT_DIM];
  int _zero = 0;
  int64_t _zero64 = 0;
  
  // quantize
  s_x_inv = *layer_2_s_x_inv.lookup_or_init(&_zero, &_zero);
  z_x = *layer_2_z_x.lookup_or_init(&_zero, &_zero);
  scale_factor_int = (s_x_inv + ROUND_CONST) >> FXP_VALUE;
  scale_factor_frac = s_x_inv - (scale_factor_int << FXP_VALUE);
  for (i = 0; i < H1; i++) {
    _i = i;
    input = *out_input.lookup_or_init(&_i, &_zero64);
    tensor_int = (input + ROUND_CONST) >> FXP_VALUE;
    tensor_frac = input - (tensor_int << FXP_VALUE);
    rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
    rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
    rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int + z_x;
    if (rounded_value > UINT8_MAX_VALUE){
      x_q[i] = (int)UINT8_MAX_VALUE;
    }else if(rounded_value < UINT8_MIN_VALUE){
      x_q[i] = (int)UINT8_MIN_VALUE; 
    }else{
      x_q[i] = (int)rounded_value;
    }
    //bpf_trace_printk("%d:%d",i,x_q[i]);
  }

  // mat_mult
  for (m = 0; m < H2; m++) {
    accumulator = 0;
    z_x = *layer_2_z_x.lookup_or_init(&_zero, &_zero);
    z_w = *layer_2_z_w.lookup_or_init(&_zero, &_zero);
    for (k = 0; k < H1; k++) {
      _k = k*H2 + m;
      weight = *(int8_t*)layer_2_weight.lookup_or_init(&_k, &_zero);
      accumulator += ((int64_t)x_q[k] - z_x)  * ((int64_t)weight - z_w);
    }

    _m = m;
    scale_factor = *layer_2_S.lookup_or_init(&_zero, &_zero);
    accumulator *= scale_factor;
    accumulator = ((accumulator + ROUND_CONST) >> FXP_VALUE);
    z_y = *layer_2_z_y.lookup_or_init(&_zero, &_zero);
    accumulator += z_y;
    out = (int64_t)accumulator;
    
    //relu
    out = MAX(out, 0);
    
    //y_cast
    if (out > UINT8_MAX_VALUE){
      y_q = (uint8_t)UINT8_MAX_VALUE;
    }else if(out < UINT8_MIN_VALUE){
      y_q = (uint8_t)UINT8_MIN_VALUE; 
    }else{
      y_q = (uint8_t)out;
    }
    
    //dequantize
    s_y = *layer_2_s_y.lookup_or_init(&_zero, &_zero);
    out =  (int64_t)y_q - z_y;
    out = out * s_y;
    //bpf_trace_printk("output2,%d:%lld",m,out);
    out_input.update(&_m, &out);
  }
  jmp_table.call(skb, 1);
  return TC_ACT_OK;
}


int nn_tc_drop_packet(struct __sk_buff *skb) {
  uint64_t ts = bpf_ktime_get_ns();
  ts = ts / 1000000;
  void* data_end = (void*)(long)skb->data_end;
  void* data = (void*)(long)skb->data;
  struct ethhdr *eth = data;
  u64 nh_off = sizeof(*eth);
  struct iphdr *iph;
  struct tcphdr *th;
  struct udphdr *uh;
  struct pkt_key_t pkt_key = {};
  int _zero = 0;
  int64_t _zero64 = 0;

  pkt_key.protocol = 0;
  pkt_key.saddr = 0;
  pkt_key.daddr = 0;
  pkt_key.sport = 0;
  pkt_key.dport = 0;

  ethernet: {
    if (data + nh_off > data_end) {
      bpf_trace_printk("i");
      return TC_ACT_SHOT;
    }
    switch(eth->h_proto) {
      case htons(ETH_P_IP): goto ip;
      default: return TC_ACT_OK;
    }
  }
  ip: {
    iph = data + nh_off;
    if ((void*)&iph[1] > data_end) {
      return TC_ACT_OK;
      }
    pkt_key.saddr    = iph->saddr;
    pkt_key.daddr    = iph->daddr;
    pkt_key.protocol = iph->protocol;

    switch(iph->protocol) {
      case IPPROTO_TCP: goto tcp;
      case IPPROTO_UDP: goto udp;
      default: return TC_ACT_OK;
    }
  }
  tcp: {
    th = (struct tcphdr *)(iph + 1);
    if ((void*)(th + 1) > data_end) {
      return TC_ACT_SHOT;
    }
    pkt_key.sport = ntohs(th->source);
    pkt_key.dport = ntohs(th->dest);
    //switch(ntohs(th->dest)) {
    //  case  12345: goto nn;
    //  default: return TC_ACT_OK;
    //}
    goto nn;
  }
  udp: {
    uh = (struct udphdr *)(iph + 1);
    if ((void*)(uh + 1) > data_end) {
      return TC_ACT_SHOT;
    }
    pkt_key.sport = ntohs(uh->source);
    pkt_key.dport = ntohs(uh->dest);
    switch(ntohs(th->dest)) {
      case  12345: goto nn;
      default: return TC_ACT_OK;
    }

    //goto nn;
  }
  nn: {
    struct pkt_leaf_t *pkt_leaf = sessions.lookup(&pkt_key);
    if (!pkt_leaf) {
      struct pkt_leaf_t zero = {};
      zero.sport = pkt_key.sport;
      zero.dport = pkt_key.dport;
      zero.num_packets = 0;
      zero.last_packet_timestamp = ts;
      sessions.update(&pkt_key, &zero);
      pkt_leaf = sessions.lookup(&pkt_key);
    }
    if (pkt_leaf != NULL) {
      int64_t x[INPUT_DIM];
      pkt_leaf->num_packets += 1;
      x[0] = pkt_leaf->sport;
      x[1] = pkt_leaf->dport;
      x[2] = iph->protocol;
      x[3] = ntohs(iph->tot_len);
      if (pkt_leaf->last_packet_timestamp > 0) {
        x[4] = ts - pkt_leaf->last_packet_timestamp;
      } else {
        x[4] = 0;
      }
      pkt_leaf->last_packet_timestamp = ts;
      x[5] = pkt_key.sport == x[0];

      x[0] <<= FXP_VALUE;
      x[1] <<= FXP_VALUE;
      x[2] <<= FXP_VALUE;
      x[3] <<= FXP_VALUE;
      x[4] <<= FXP_VALUE;
      x[5] <<= FXP_VALUE;

      pkt_leaf->features[0] += x[3];
      pkt_leaf->features[1] += x[4];
      pkt_leaf->features[2] += x[5];

      x[6] = pkt_leaf->features[0]/pkt_leaf->num_packets;
      x[7] = pkt_leaf->features[1]/pkt_leaf->num_packets;
      x[8] = pkt_leaf->features[2]/pkt_leaf->num_packets;

      pkt_leaf->features[3] += abs(x[3] - x[6]);
      pkt_leaf->features[4] += abs(x[4] - x[7]);
      pkt_leaf->features[5] += abs(x[5] - x[8]);

      x[9]  = pkt_leaf->features[3]/pkt_leaf->num_packets;
      x[10] = pkt_leaf->features[4]/pkt_leaf->num_packets;
      x[11] = pkt_leaf->features[5]/pkt_leaf->num_packets;

      unsigned int i, k, m, _i, _k, _m;
      int64_t _data_min, _data_max, range, in, out, accumulator;
      int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac,  s_x_inv, z_x, z_w, scale_factor, z_y, s_y;
      int8_t weight;
      uint8_t y_q;
      int x_q[INPUT_DIM];

      sessions.update(&pkt_key, pkt_leaf);
      // NORMALIZATION
      for (k = 0; k < INPUT_DIM; k++) {
        _k = k;
        _data_min = *data_min.lookup_or_init(&_k, &_zero64);
        _data_max = *data_max.lookup_or_init(&_k, &_zero64);
        range = _data_max - _data_min;
        x[k] -= _data_min;
        x[k] <<= 16;
        bool aneg = x[k] < 0;
        bool bneg = range < 0;
        uint64_t adiv = aneg ? -x[k] : x[k];
        uint64_t bdiv = bneg ? -range : range;
        uint64_t ou = adiv / bdiv;
        x[k] = (aneg != bneg) ? -ou : ou;
      }

      // quantize
      s_x_inv = *layer_1_s_x_inv.lookup_or_init(&_zero, &_zero);
      z_x = *layer_1_z_x.lookup_or_init(&_zero, &_zero);
      scale_factor_int = (s_x_inv + ROUND_CONST) >> FXP_VALUE;
      scale_factor_frac = s_x_inv - (scale_factor_int << FXP_VALUE);
      for (i = 0; i < INPUT_DIM; i++) {
        _i = i;
        in = x[i];  
        in_input.update(&_i, &in);
        //bpf_trace_printk("input,%d:%lld",i,x[i]);
        tensor_int = (x[i] + ROUND_CONST) >> FXP_VALUE;
        tensor_frac = x[i] - (tensor_int << FXP_VALUE);
        rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
        rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
        rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int + z_x;
        if (rounded_value > UINT8_MAX_VALUE){
              x_q[i] = (int)UINT8_MAX_VALUE;
        }else if(rounded_value < UINT8_MIN_VALUE){
              x_q[i] = (int)UINT8_MIN_VALUE; 
        }else{
              x_q[i] = (int)rounded_value;
        }
      }

      // mat_mult
      for (m = 0; m < H1; m++) {
        accumulator = 0;
        z_x = *layer_1_z_x.lookup_or_init(&_zero, &_zero);
        z_w = *layer_1_z_w.lookup_or_init(&_zero, &_zero);
        for (k = 0; k < INPUT_DIM; k++) {
          _k = k*H1 + m;
          weight = *(int8_t*)layer_1_weight.lookup_or_init(&_k, &_zero);
          
          accumulator += ((int64_t)x_q[k] - z_x)  * ((int64_t)weight - z_w);
        }

        
        _m = m;
        scale_factor = *layer_1_S.lookup_or_init(&_zero, &_zero);
        accumulator *= scale_factor;
        accumulator = ((accumulator + ROUND_CONST) >> FXP_VALUE);
        z_y = *layer_1_z_y.lookup_or_init(&_zero, &_zero);
        accumulator += z_y;

        out = (int64_t)accumulator;
        
        //relu
        out = MAX(out, 0);
        
        //y_cast
        if (out > UINT8_MAX_VALUE){
              y_q = (uint8_t)UINT8_MAX_VALUE;
        }else if(out < UINT8_MIN_VALUE){
              y_q = (uint8_t)UINT8_MIN_VALUE; 
        }else{
              y_q = (uint8_t)out;
        }
        
        //dequantize
        s_y = *layer_1_s_y.lookup_or_init(&_zero, &_zero);
        out =  (int64_t)y_q - z_y;
        out = out * s_y;
        //bpf_trace_printk("output1,%d:%lld",m,out);
        out_input.update(&_m, &out);
        
        
      }
      jmp_table.call(skb, 0);
    }
  }
  return TC_ACT_OK;
}

"""

def map_bpf_table(hashmap, values, c_type='int'):
  MAP_SIZE = len(values)
  assert len(hashmap.items()) == MAP_SIZE
  keys = (hashmap.Key * MAP_SIZE)()
  new_values = (hashmap.Leaf * MAP_SIZE)()

  if isinstance(hashmap, table.PerCpuArray):
      for i, (k, v) in enumerate(hashmap.items()):
          keys[i] = ct.c_int(i)
          for j, d in enumerate(v):
              if c_type == 'int':
                  v[j] = ct.c_int(values[i])
              elif c_type == 'int64_t':
                  v[j] = ct.c_longlong(values[i])
              else:
                  v[j] = ct.c_longlong(values[i])
          hashmap.__setitem__(k, v)
      # hashmap.items_update_batch(keys, new_values)
  else:
      for i in range(MAP_SIZE):
          keys[i] = ct.c_int(i)
          if c_type == 'int':
              new_values[i] = ct.c_int(values[i])
          elif c_type == 'int8_t':
              new_values[i] = ct.c_char(values[i])
          elif c_type == 'int64_t':
              new_values[i] = ct.c_longlong(values[i])
          else:
              new_values[i] = ct.c_longlong(values[i])
      hashmap.items_update_batch(keys, new_values)

SIMULATION_TIME = 10
if __name__ == '__main__':
  with open('./json/mlp_params.json') as f:
      params = json.load(f)


  bpf_text = bpf_text.replace('LAYER_1_WEIGHT',  str(len(params["layer_1_weight"])))
  bpf_text = bpf_text.replace('LAYER_2_WEIGHT',  str(len(params["layer_2_weight"])))
  bpf_text = bpf_text.replace('LAYER_3_WEIGHT',  str(len(params["layer_3_weight"])))
  bpf_text = bpf_text.replace('LAYER_4_WEIGHT',  str(len(params["layer_4_weight"])))
  bpf_text = bpf_text.replace('LAYER_5_WEIGHT',  str(len(params["layer_5_weight"])))
  bpf_text = bpf_text.replace('LAYER_6_WEIGHT',  str(len(params["layer_6_weight"])))
  bpf_text = bpf_text.replace('DATA_MIN',        str(len(params["data_min"])))
  bpf_text = bpf_text.replace('DATA_MAX',      str(len(params["data_max"])))

  INGRESS = "ffff:ffff2"
  EGRESS = "ffff:ffff3"

  device = sys.argv[1]
  #resdir = sys.argv[2]
  resdir = "./log"
  ret = []

  try:
      b = BPF(text=bpf_text, debug=0)
      fn = b.load_func("nn_tc_drop_packet", BPF.SCHED_CLS)
      idx = ipr.link_lookup(ifname=device)[0]

      for i in range(0, lib.bpf_num_functions(b.module)):
          func_name = lib.bpf_function_name(b.module, i)
          print(func_name, lib.bpf_function_size(b.module, func_name))

      ipr.tc("add", "clsact", idx);
      ipr.tc("add-filter", "bpf", idx, ":1", fd=fn.fd, name=fn.name, parent=INGRESS, classid=1, direct_action=True)

      jmp_table = b.get_table("jmp_table")
      nn1_fn = b.load_func("nn1", BPF.SCHED_CLS);
      nn2_fn = b.load_func("nn2", BPF.SCHED_CLS);
      nn3_fn = b.load_func("nn3", BPF.SCHED_CLS);
      nn4_fn = b.load_func("nn4", BPF.SCHED_CLS);
      nn5_fn = b.load_func("nn5", BPF.SCHED_CLS);
      mse_fn = b.load_func("mse_cal", BPF.SCHED_CLS);
      jmp_table[ct.c_int(0)] = ct.c_int(nn1_fn.fd)
      jmp_table[ct.c_int(1)] = ct.c_int(nn2_fn.fd)
      jmp_table[ct.c_int(2)] = ct.c_int(nn3_fn.fd)
      jmp_table[ct.c_int(3)] = ct.c_int(nn4_fn.fd)
      jmp_table[ct.c_int(4)] = ct.c_int(nn5_fn.fd)
      jmp_table[ct.c_int(5)] = ct.c_int(mse_fn.fd)


      layer_1_weight  = b.get_table("layer_1_weight")
      layer_2_weight  = b.get_table("layer_2_weight")
      layer_3_weight  = b.get_table("layer_3_weight")
      layer_4_weight  = b.get_table("layer_4_weight")
      layer_5_weight  = b.get_table("layer_5_weight")
      layer_6_weight  = b.get_table("layer_6_weight")
      layer_1_s_x_inv = b.get_table("layer_1_s_x_inv")
      layer_2_s_x_inv = b.get_table("layer_2_s_x_inv")
      layer_3_s_x_inv = b.get_table("layer_3_s_x_inv")
      layer_4_s_x_inv = b.get_table("layer_4_s_x_inv")
      layer_5_s_x_inv = b.get_table("layer_5_s_x_inv")
      layer_6_s_x_inv = b.get_table("layer_6_s_x_inv")
      layer_1_z_x = b.get_table("layer_1_z_x")
      layer_2_z_x = b.get_table("layer_2_z_x")
      layer_3_z_x = b.get_table("layer_3_z_x")
      layer_4_z_x = b.get_table("layer_4_z_x")
      layer_5_z_x = b.get_table("layer_5_z_x")
      layer_6_z_x = b.get_table("layer_6_z_x")
      layer_1_z_w = b.get_table("layer_1_z_w")
      layer_2_z_w = b.get_table("layer_2_z_w")
      layer_3_z_w = b.get_table("layer_3_z_w")
      layer_4_z_w = b.get_table("layer_4_z_w")
      layer_5_z_w = b.get_table("layer_5_z_w")
      layer_6_z_w = b.get_table("layer_6_z_w")
      layer_1_s_y = b.get_table("layer_1_s_y")
      layer_2_s_y = b.get_table("layer_2_s_y")
      layer_3_s_y = b.get_table("layer_3_s_y")
      layer_4_s_y = b.get_table("layer_4_s_y")
      layer_5_s_y = b.get_table("layer_5_s_y")
      layer_6_s_y = b.get_table("layer_6_s_y")
      layer_1_z_y = b.get_table("layer_1_z_y")
      layer_2_z_y = b.get_table("layer_2_z_y")
      layer_3_z_y = b.get_table("layer_3_z_y")
      layer_4_z_y = b.get_table("layer_4_z_y")
      layer_5_z_y = b.get_table("layer_5_z_y")
      layer_6_z_y = b.get_table("layer_6_z_y")
      layer_1_S = b.get_table("layer_1_S")
      layer_2_S = b.get_table("layer_2_S")
      layer_3_S = b.get_table("layer_3_S")
      layer_4_S = b.get_table("layer_4_S")
      layer_5_S = b.get_table("layer_5_S")
      layer_6_S = b.get_table("layer_6_S")
      data_min  = b.get_table("data_min")
      data_max  = b.get_table("data_max")
      threshold  = b.get_table("threshold")

      map_bpf_table(layer_1_weight,  params['layer_1_weight'],  'int')
      map_bpf_table(layer_2_weight,  params['layer_2_weight'],  'int')
      map_bpf_table(layer_3_weight,  params['layer_3_weight'],  'int')
      map_bpf_table(layer_4_weight,  params['layer_4_weight'],  'int')
      map_bpf_table(layer_5_weight,  params['layer_5_weight'],  'int')
      map_bpf_table(layer_6_weight,  params['layer_6_weight'],  'int')
      map_bpf_table(layer_1_s_x_inv, params['layer_1_s_x_inv'], 'int')
      map_bpf_table(layer_2_s_x_inv, params['layer_2_s_x_inv'], 'int')
      map_bpf_table(layer_3_s_x_inv, params['layer_3_s_x_inv'], 'int')
      map_bpf_table(layer_4_s_x_inv, params['layer_4_s_x_inv'], 'int')
      map_bpf_table(layer_5_s_x_inv, params['layer_5_s_x_inv'], 'int')
      map_bpf_table(layer_6_s_x_inv, params['layer_6_s_x_inv'], 'int')
      map_bpf_table(layer_1_z_x,     params['layer_1_z_x'],     'int')
      map_bpf_table(layer_2_z_x,     params['layer_2_z_x'],     'int')
      map_bpf_table(layer_3_z_x,     params['layer_3_z_x'],     'int')
      map_bpf_table(layer_4_z_x,     params['layer_4_z_x'],     'int')
      map_bpf_table(layer_5_z_x,     params['layer_5_z_x'],     'int')
      map_bpf_table(layer_6_z_x,     params['layer_6_z_x'],     'int')
      map_bpf_table(layer_1_z_w,     params['layer_1_z_w'],     'int')
      map_bpf_table(layer_2_z_w,     params['layer_2_z_w'],     'int')
      map_bpf_table(layer_3_z_w,     params['layer_3_z_w'],     'int')
      map_bpf_table(layer_4_z_w,     params['layer_4_z_w'],     'int')
      map_bpf_table(layer_5_z_w,     params['layer_5_z_w'],     'int')
      map_bpf_table(layer_6_z_w,     params['layer_6_z_w'],     'int')
      map_bpf_table(layer_1_s_y,     params['layer_1_s_y'],     'int')
      map_bpf_table(layer_2_s_y,     params['layer_2_s_y'],     'int')
      map_bpf_table(layer_3_s_y,     params['layer_3_s_y'],     'int')
      map_bpf_table(layer_4_s_y,     params['layer_4_s_y'],     'int')
      map_bpf_table(layer_5_s_y,     params['layer_5_s_y'],     'int')
      map_bpf_table(layer_6_s_y,     params['layer_6_s_y'],     'int')
      map_bpf_table(layer_1_z_y,     params['layer_1_z_y'],     'int')
      map_bpf_table(layer_2_z_y,     params['layer_2_z_y'],     'int')
      map_bpf_table(layer_3_z_y,     params['layer_3_z_y'],     'int')
      map_bpf_table(layer_4_z_y,     params['layer_4_z_y'],     'int')
      map_bpf_table(layer_5_z_y,     params['layer_5_z_y'],     'int')
      map_bpf_table(layer_6_z_y,     params['layer_6_z_y'],     'int')
      map_bpf_table(layer_1_S,       params['layer_1_S'],       'int')
      map_bpf_table(layer_2_S,       params['layer_2_S'],       'int')
      map_bpf_table(layer_3_S,       params['layer_3_S'],       'int')
      map_bpf_table(layer_4_S,       params['layer_4_S'],       'int')
      map_bpf_table(layer_5_S,       params['layer_5_S'],       'int')
      map_bpf_table(layer_6_S,       params['layer_6_S'],       'int')
      map_bpf_table(threshold,       params['threshold'],       'int')
      map_bpf_table(data_min,        params['data_min'],        'int64_t')
      map_bpf_table(data_max,        params['data_max'],        'int64_t')
      dropcnt  = b.get_table("dropcnt")

      start = datetime.now()
      while True:
          try:
              dropcnt.clear()
              time.sleep(1)
              for k, v in dropcnt.items():
                  ret.append(v.value)
              end = datetime.now()
              duration = (end - start).total_seconds()
              if duration > SIMULATION_TIME:
                  break
              #b.trace_print()
          except KeyboardInterrupt:
              break
  finally:
      if "idx" in locals():
          ipr.tc("del", "clsact", idx)
      filename = f"{resdir}/rxpps.log"
      with open (filename, 'w') as f:
          for d in ret:
              f.write(f"{d}\n")

      #データ保存
      #time_list = [1, 10, 100, 1e3, 1e4, 1e5, 5e5]
      #pps = [0, 0, 0, 0, 0, 0, 0]
      #file_path = 'csv/cpu_usage_nomodel1.csv'
      #data = {
      #    'time_list': time_list,
      #    'eBPF': pps
      #}
      #df = pd.DataFrame(data)
      #df.to_csv(file_path, index=False)
