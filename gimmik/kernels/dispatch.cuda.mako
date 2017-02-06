# -*- coding: utf-8 -*-

#include "cublas.h"

void
${funcn}(int m, int n, int k, const ${dtype} alpha, 
         const ${dtype}* __restrict__ a, int lda,
         const ${dtype}* __restrict__ b, int ldb,
         const ${dtype} beta, ${dtype}* __restrict__ c, int ldc,
         unsigned long id)
{

  int threads = 128;
  int blocks = (n + threads - 1) / threads;
  
  if (beta == 0.0)
  {
    switch(id)
    {
    % for id in mids:   
      case ${id}:
        ${'_'.join([funcn, str(id), 'b0'])}<<<blocks, threads>>>(n, b, ldb, c, ldc); break;
    % endfor

      default:
      % if order == 'row-major':
        cublasDgemm('N', 'N', n, m, k, alpha, B, ldb, A, lda, beta, c, ldc); break;
      % elif order == 'col-major':
        cublasDgemm('N', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); break;
      % endif
    }
  }
  else if (beta == 1.0)
  {
    switch(id)
    {
    % for id in mids:   
      case ${id}:
        ${'_'.join([funcn, str(id), 'b1'])}<<<blocks, threads>>>(n, b, ldb, c, ldc); break;
    % endfor
     
      default:
      % if order == 'row-major':
        cublasDgemm('N', 'N', n, m, k, alpha, B, ldb, A, lda, beta, c, ldc); break;
      % elif order == 'col-major':
        cublasDgemm('N', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); break;
      % endif
    }
  }
  else
  {
  % if order == 'row-major':
    cublasDgemm('N', 'N', n, m, k, alpha, B, ldb, A, lda, beta, c, ldc);
  % elif order == 'col-major':
    cublasDgemm('N', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  % endif
  }
}
