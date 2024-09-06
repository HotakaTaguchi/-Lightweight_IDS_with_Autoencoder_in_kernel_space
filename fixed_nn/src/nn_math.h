/*******************************************************************
@file nn_math.h
 *  @brief Function prototypes for mathematical functions
 *
 *
 *  @author Benjamin Fuhrer
 *
*******************************************************************/
#ifndef NN_MATH_H
#define NN_MATH_H

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

#define NUM_BITS 8
#define UINT8_MAX_VALUE 255
#define UINT8_MIN_VALUE 0
#define INT16_MAX_VALUE 32767
#define FXP_VALUE 16
#define ROUND_CONST (1 << (FXP_VALUE - 1)) // = 0.5 四捨五入用
#define DEBUG_MODE
#ifdef DEBUG_MODE
#include <stdio.h>
#include <math.h>
#endif

#include <stdint.h>

void mat_mult(const uint8_t *mat_l, const int8_t *mat_r, const int mat_l_z, const int mat_r_z, const int mat_y_z, const int scale_factor, int64_t *result, const unsigned int K, const unsigned int M);
/**
 * @brief Calculates matrix multiplication as: Y = XW
 *  
 * 
 * @param mat_l - left matrix (X), size NxK
 * @param mat_r - right matrix (W), size (K+1)xM, the last row of W contains the bias vector
 * @param result - output matrix (Y), size NxM
 * @param N - number of rows in X
 * @param K - number of columns/rows in X/W
 * @param M - number of columns in W
 * @return Void
 */

void relu(int64_t *tensor_q, const unsigned int size);
/**
 * @brief ReLU activation function
 * 
 * @param tensor_q - input tensor
 * @param size - size of flattened tensor
 * @return Void
 */

void y_cast(int64_t *tensor_q, uint8_t *result, const unsigned int size);
/**
 * @brief y_cast activation function
 * 
 * @param tensor_q - input tensor
 * @param result - output tensor
 * @param size - size of flattened tensor
 * @return Void
 */

void quantize(const int64_t *tensor_in, uint8_t *tensor_q, const int scale_factor_inv, const int zero_point,
     const unsigned int size);
/**
 * @brief Scale quantization of a tensor by a single amax value
 * 
 * @param tensor_in - input tensor
 * @param tensor_q - output quantized tensor
 * @param scale_factor - 127 / amax
 * @param scale_factor_inv - 1 / scale_factor
 * @param size - size of flattened tensor
 * @return Void
 */

void dequantize(uint8_t *tensor_q, int64_t *tensor_out, const int y_scale_factor, const int y_zero_point, const unsigned int  M);
/**
 * @brief Scale dequantization with per-row granulity
 * Each row is multiplied by the corresponding column amax value
 * offline calculate reciprocal(amax) so we can replace division by multiplication
 * 
 * @param mat_in - NxM input matrix to dequantize
 * @param scale_factor_w_inv -1XM row vector of layer's weight matrix scale factor values
 * @param scale_factor_x_inv - input inverse scale factor
 * @param N
 * @param M
 * @return Void
*/

void get_output(const int64_t *tensor_out, int64_t *indices, const unsigned int M);
/**
 * @brief Calculate argmax per columns of an NxM matrix
 * 
 * @param mat_in - NxM input matrix
 * @param indices - 1xM indices to store argmax of each column
 * @param N
 * @param M
 * @return Void
 */


#endif //