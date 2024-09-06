#include "mlp_params.h"
#include "nn.h"
#include "nn_math.h"
void run_mlp(const int64_t *x, int64_t *class_indices) {
    int64_t out_encoder1[H1];
    linear_layer(x, layer_1_weight, out_encoder1, layer_1_S,
        layer_1_z_y,layer_1_z_w,layer_1_z_x, layer_1_s_x_inv, layer_1_s_y,
        INPUT_DIM, H1, 1);
    int64_t out_encoder2[H2];
    linear_layer(out_encoder1, layer_2_weight, out_encoder2, layer_2_S,
        layer_2_z_y,layer_2_z_w,layer_2_z_x, layer_2_s_x_inv, layer_2_s_y,
        H1, H2, 2);
    int64_t out_encoder3[H3];
    linear_layer(out_encoder2, layer_3_weight, out_encoder3, layer_3_S,
        layer_3_z_y,layer_3_z_w,layer_3_z_x, layer_3_s_x_inv, layer_3_s_y,
        H2, H3, 3);
    int64_t out_decoder1[H4];
    linear_layer(out_encoder3, layer_4_weight, out_decoder1, layer_4_S,
        layer_4_z_y,layer_4_z_w,layer_4_z_x, layer_4_s_x_inv, layer_4_s_y,
        H3, H4, 4);
    int64_t out_decoder2[H5];
    linear_layer(out_decoder1, layer_5_weight, out_decoder2, layer_5_S,
        layer_5_z_y,layer_5_z_w,layer_5_z_x, layer_5_s_x_inv, layer_5_s_y,
        H4, H5, 5);
    int64_t out_decoder3[H6];
    linear_layer(out_decoder2, layer_6_weight, out_decoder3, layer_6_S,
        layer_6_z_y,layer_6_z_w,layer_6_z_x, layer_6_s_x_inv, layer_6_s_y,
        H5, H6, 6);
  
    
    // get output
    get_output(out_decoder3, class_indices, H6);
}