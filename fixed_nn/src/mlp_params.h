/*******************************************************************
@file mlp_params.h
*  @brief variable prototypes for model parameters and amax values
*
*
*  @author Benjamin Fuhrer
*
*******************************************************************/
#ifndef MLP_PARAMS
#define MLP_PARAMS

#define INPUT_DIM 12
#define H1 10
#define H2 8
#define H3 6
#define H4 8
#define H5 10
#define H6 12
#include <stdint.h>


// quantization/dequantization constants
extern const int layer_1_s_x_inv;
extern const int layer_1_s_x;
extern const int layer_1_z_x;
extern const int layer_1_z_w;
extern const int layer_1_s_y;
extern const int layer_1_z_y;
extern const int layer_1_S;
extern const int layer_2_s_x_inv;
extern const int layer_2_s_x;
extern const int layer_2_z_x;
extern const int layer_2_z_w;
extern const int layer_2_s_y;
extern const int layer_2_z_y;
extern const int layer_2_S;
extern const int layer_3_s_x_inv;
extern const int layer_3_s_x;
extern const int layer_3_z_x;
extern const int layer_3_z_w;
extern const int layer_3_s_y;
extern const int layer_3_z_y;
extern const int layer_3_S;
extern const int layer_4_s_x_inv;
extern const int layer_4_s_x;
extern const int layer_4_z_x;
extern const int layer_4_z_w;
extern const int layer_4_s_y;
extern const int layer_4_z_y;
extern const int layer_4_S;
extern const int layer_5_s_x_inv;
extern const int layer_5_s_x;
extern const int layer_5_z_x;
extern const int layer_5_z_w;
extern const int layer_5_s_y;
extern const int layer_5_z_y;
extern const int layer_5_S;
extern const int layer_6_s_x_inv;
extern const int layer_6_s_x;
extern const int layer_6_z_x;
extern const int layer_6_z_w;
extern const int layer_6_s_y;
extern const int layer_6_z_y;
extern const int layer_6_S;

extern const int8_t layer_1_weight[120];
extern const int8_t layer_2_weight[80];
extern const int8_t layer_3_weight[48];
extern const int8_t layer_4_weight[48];
extern const int8_t layer_5_weight[80];
extern const int8_t layer_6_weight[120];

// MinMaxscaler params
extern const int64_t data_min[12];
extern const int64_t data_scale[12];

// threshold
extern const int64_t threshold;

#endif // end of MLP_PARAMS