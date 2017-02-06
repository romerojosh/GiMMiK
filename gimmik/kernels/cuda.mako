# -*- coding: utf-8 -*-

__global__ void
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    ${dtype} dotp;

    if (i < n)
    {
    % if order == 'row-major':
      % for j, jx in enumerate(mat):
          dotp = ${' + '.join('{kx}*b[i + {k}*ldb]'.format(k=k, kx=kx)
                              for k, kx in enumerate(jx) if kx != 0) or 0};
      % if beta == 0:
          c[i + ${j}*ldc] = dotp;
      % elif beta == 1:
          c[i + ${j}*ldc] += dotp;
      % else:
          c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
      % endif
    % endfor

    % elif order == 'col-major':
      % for j, jx in enumerate(mat):
          dotp = ${' + '.join('{kx}*b[i*ldb + {k}]'.format(k=k, kx=kx)
                              for k, kx in enumerate(jx) if kx != 0) or 0};
      % if beta == 0:
          c[i*ldc + ${j}] = dotp;
      % elif beta == 1:
          c[i*ldc + ${j}] += dotp;
      % else:
          c[i*ldc + ${j}] = dotp + ${beta}*c[i*ldc + ${j}];
      % endif
      % endfor
    % endif
    }
}
