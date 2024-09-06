#include "nn_math.h"

void mat_mult(const uint8_t *mat_l, const int8_t *mat_r, const int mat_l_z, const int mat_r_z, const int mat_y_z, const int scale_factor, int64_t *result, const unsigned int K, const unsigned int M)
{
  unsigned int k, m;
  unsigned int col;
  int64_t accumulator;

  for (m = 0; m < M; m++) {
    accumulator = 0;
    for (k = 0; k < K; k++) {
      col = k*M;
      accumulator += ((int64_t)mat_l[k] - mat_l_z)  * ((int64_t)mat_r[col + m] - mat_r_z);
    }
    accumulator *= scale_factor;
    accumulator = ((accumulator + ROUND_CONST) >> FXP_VALUE);
    accumulator += mat_y_z;
    #ifdef DEBUG_MODE
          //printf("out%d:%ld\n", m, accumulator); 
    #endif
    result[m] = (int64_t)accumulator;
  }
}


void relu(int64_t *tensor_q, const unsigned int size)
{
  unsigned int i;
  for (i = 0; i < size; i++) {
    tensor_q[i] = MAX(tensor_q[i], 0);
  }
}

void y_cast(int64_t *tensor_q, uint8_t *result, const unsigned int size)
{
  unsigned int i;
  for (i = 0; i < size; i++) {
    if (tensor_q[i] > UINT8_MAX_VALUE){
          result[i] = (uint8_t)UINT8_MAX_VALUE;
    }else if(tensor_q[i] < UINT8_MIN_VALUE){
          result[i] = (uint8_t)UINT8_MIN_VALUE; 
    }else{
          result[i] = (uint8_t)tensor_q[i];
    }
  }
}


void quantize(const int64_t *tensor_in, uint8_t *tensor_q, const int scale_factor_inv, const int zero_point,
     const unsigned int size)
{
  unsigned int i;
  int rounded_value, tensor_int, tensor_frac;
  // separation to integer and fraction parts
  int scale_factor_int = (scale_factor_inv + ROUND_CONST) >> FXP_VALUE;
  int scale_factor_frac = scale_factor_inv - (scale_factor_int << FXP_VALUE);
  for (i = 0; i < size; i++) {
    tensor_int = (tensor_in[i] + ROUND_CONST) >> FXP_VALUE;
    #ifdef DEBUG_MODE
            //printf("in:%ld\n", tensor_in[i]); 
    #endif
    tensor_frac = tensor_in[i] - (tensor_int << FXP_VALUE);
    // int * fxp = result is in fxp */
    rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
    // 第2小数点四捨五入 */
    rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
    // 第1小数点四捨五入
    rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int + zero_point;
    #ifdef DEBUG_MODE
         //printf("in%d:%d\n", i,rounded_value); 
    #endif
    if (rounded_value > UINT8_MAX_VALUE){
          tensor_q[i] = (uint8_t)UINT8_MAX_VALUE;
    }else if(rounded_value < UINT8_MIN_VALUE){
          tensor_q[i] = (uint8_t)UINT8_MIN_VALUE; 
    }else{
          tensor_q[i] = (uint8_t)rounded_value; /* store quantized value in output tensor */
    }
  }
}


void dequantize(uint8_t *tensor_q, int64_t *tensor_out, const int y_scale_factor, const int y_zero_point, const unsigned int  M)
{
  unsigned int i;
  int64_t rounded_value;
  for (i = 0; i < M; i++) {
    #ifdef DEBUG_MODE
         //printf("out%d:%ld\n", i,tensor_q[i]); 
    #endif
    rounded_value =  (int64_t)tensor_q[i] - y_zero_point;

    rounded_value = rounded_value * y_scale_factor;

    tensor_out[i] = rounded_value; /* store dequantized value in output tensor */
  }
}

void get_output(const int64_t *tensor_out, int64_t *indices, const unsigned int M)
{
  unsigned int m;
  for (m = 0; m < M; m++) {
    indices[m] = tensor_out[m];
  }
}

