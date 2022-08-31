#include <stdint.h>
#include <stdio.h>

#ifndef COMET_GENERIC_MKERNEL_H
#define COMET_GENERIC_MKERNEL_H

// generic arch-independent gemm microkernel reference implementation:
// https://github.com/flame/blis/blob/master/config/template/kernels/3/bli_gemm_template_noopt_mxn.c

// implements, C := beta * C + alpha * A * B
//   where C: m x n,
//         A: m x k,
//         B: k x n,
//         alpha: scalar,
//         beta: scalar.
//
//   param a: address of a micropanel of matrix A of dimension m x k, stored in column major order.
//   param b: address of a micropanel of matrix B of dimension k x n, stored in row major order.
//   param c: address of a matrix C of dimension m x n, stored according to rs_c and cs_c.
//   param rs_c: row stride of matrix C (i.e.,: the distance to the next row, in units of matrix elements).
//   param cs_c: column stride of matrix C (i.e.,: the distance to the next column, in units of matrix elements).
//               rs_c == 1 && cs_c == 0: means contiguous col-storage desired for C,
//               rs_c == 0 && cs_c == 1: means contiguous row-storage desired for C.

void dgemm_generic_noopt_mxn (
  int64_t m,
  int64_t n,  
  int64_t k,
  double* alpha,
  double* a, double* b, 
  double* beta,
  double* c, 
  int64_t rs_c, int64_t cs_c)
  {

    int64_t MR = m;
    int64_t NR = n;

    int64_t i, j, l;
    
    int64_t rs_ab = 1;
    int64_t cs_ab = MR; 

    double ai, bj; 
    double* abij;
    double ab[MR*NR];  // holds the computed values 
    for (i = 0; i < MR*NR; i++)  // initialization
      ab[i] = 0.0;
       
    /* Perform a series of k rank-1 updates into ab. */
    for (l = 0; l < k; ++l)
    {
      abij = ab;

      for (j = 0; j < NR; ++j)
      {
        bj = *(b + j);
        for (i = 0; i < MR; ++i)
        {
          ai = *(a + i);
          *abij += ai*bj;  // perform compute
                  
          abij += rs_ab;
        }
      }
      a += MR;  
      b += NR;  
    }  

    // scale by alpha
    for (i = 0; i < MR*NR; i++) 
    {
      ab[i] = (*alpha)*ab[i];
    }

    // Scale c by beta and then add the scaled result in ab.
    for (j = 0; j < NR; ++j) 
    {
      for (i = 0; i < MR; ++i)
      {
        c[i*rs_c + j*cs_c] = ab[i*rs_ab + j*cs_ab] + 
                                    c[i*rs_c + j*cs_c]*(*beta);
      }
    }

  };

#endif /** COMET_GENERIC_MKERNEL_H */
