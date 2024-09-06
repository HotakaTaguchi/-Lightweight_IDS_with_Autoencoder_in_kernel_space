#include "nn.h"
#include "nn_math.h"

void linear_layer(const int64_t *x, const int8_t *w, int64_t *output, const int scale_factor,
    const int y_zero_point, const int w_zero_point, const int x_zero_point, const int x_scale_factor_inv, const int y_scale_factor,
    const unsigned int  K, const unsigned int  M,
    const unsigned int  encode_layer)
{
  uint8_t x_q[K];
  int64_t y1_q[M];
  uint8_t y2_q[M];
  quantize(x, x_q, x_scale_factor_inv, x_zero_point,  K);
  mat_mult(x_q, w, x_zero_point, w_zero_point, y_zero_point, scale_factor, y1_q, K, M);
  if (encode_layer<=3) {
    relu(y1_q, M);
  }
  y_cast(y1_q, y2_q, M);
  dequantize(y2_q, output, y_scale_factor, y_zero_point, M);

}