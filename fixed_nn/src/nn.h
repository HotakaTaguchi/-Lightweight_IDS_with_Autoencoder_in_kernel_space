/*******************************************************************
@file nn.h
 *  @brief Function prototypes for neural network layers
 *
 *
 *  @author Benjamin Fuhrer
 *
*******************************************************************/
#ifndef NN_H
#define NN_H

#include <stdint.h>

void linear_layer(const int64_t *x, const int8_t *w, int64_t *output, const int scale_factor,
                  const int y_zero_point, const int w_zero_point, const int x_zero_point, const int x_scale_factor_inv, const int y_scale_factor,
                  const unsigned int K, const unsigned int M, const unsigned int encode_layer);
/**
 * @brief A neural network linear layer withthout bias  Y = ReLU(XW)
 *  x is quantized before multiplication with w and then dequantized per-row granulity prior to the activation function
 * 
 * @param x - NxK input matrix
 * @param w - KxM layer weight matrix
 * @param output - NxM output matrix
 * @param scale_factor
 * @param y_zero_point
 * @param w_zero_point
 * @param x_zero_point
 * @param y_scale_factor
 * @param K
 * @param M
 * @param encode_layer - boolean value if layer is a encode_layer (activation)
 * 
 * @return Void
 */


#endif 