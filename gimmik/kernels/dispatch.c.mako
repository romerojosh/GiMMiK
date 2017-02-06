# -*- coding: utf-8 -*-

#include "cblas.h"

void
${funcn}(int m, int n, int k, const ${dtype} alpha, 
         const ${dtype}* restrict a, int lda,
         const ${dtype}* restrict b, int ldb,
         const ${dtype} beta, ${dtype}* restrict c, int ldc,
         unsigned long id)
{
  if (beta == 0.0)
  {
    switch(id)
    {
    % for id in mids:   
      case ${id}:
        ${'_'.join([funcn, str(id), 'b0'])}(n, b, ldb, c, ldc); break;
    % endfor

      default:
      % if order == 'row-major':
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
      % elif order == 'col-major':
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      % endif
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); break;
    }
  }
  else if (beta == 1.0)
  {
    switch(id)
    {
    % for id in mids:   
      case ${id}:
        ${'_'.join([funcn, str(id), 'b1'])}(n, b, ldb, c, ldc); break;
    % endfor
     
      default:
      % if order == 'row-major':
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      % elif order == 'col-major':
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      % endif
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); break;
    }
  }
  else
  {
  % if order == 'row-major':
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
  % elif order == 'col-major':
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
  % endif
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); 
  }
}
