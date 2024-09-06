/*******************************************************************
@file mlp.h
 *  @brief Function prototypes to create and run an MLP for inference
 *  with only integers (8-bit integers and 32-bit integers
 *  in fixed-point)
 *
 *  @author Benjamin Fuhrer
 *
*******************************************************************/
#ifndef MLP_H
#define MLP_H
#include <stdint.h>

void run_mlp(const int64_t *x, unsigned int *class_indices);
/**
 * @brief Function to run an mlp for classification
 * 
 * @param x - NxK input matrix
 * @param class_indices - Nx1 vector for storing class index prediction
 */


#endif 