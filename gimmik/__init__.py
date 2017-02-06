# -*- coding: utf-8 -*-

import pkgutil
import re

from mako.template import Template
import numpy as np
import glob 

from gimmik._version import __version__


def generate_mm(mat, dtype, platform, alpha=1.0, beta=0.0, funcn='gimmik_mm', order='row-major'):
    # Data type
    dtype = np.dtype(dtype).type
    if dtype == np.float32:
        dtype = 'float'
    elif dtype == np.float64:
        dtype = 'double'
    else:
        raise ValueError('Invalid floating point data type')

    if order not in ['row-major', 'col-major']:
        raise ValueError('Invalid order')

    # Multiply the matrix through by alpha
    mat = alpha*mat

    # Template arguments
    tplargs = {'dtype': dtype, 'mat': mat, 'beta': beta, 'funcn': funcn, 'order': order}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, 'kernels/{0}.mako'.format(platform))
    src = Template(tpl).render(**tplargs)

    # At single precision suffix all floating point constants by 'f'
    if dtype == 'float':
        src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                     r'\g<0>f', src)

    # Return the source
    return src

def generate_dispatch(matdir, dtype, platform, funcn='gimmik_mm', order='row-major'):

    matfiles = glob.glob(matdir + "/*.txt")

    src = ""
    mids = set()

    # Generate matrix-multiply functions
    for m in matfiles:
      f = open(m, 'r')

      mid = int(f.readline().strip())
      if mid in mids:
        raise ValueError('Non-unique id detected')
      elif mid == 0:
        raise ValueError('Cannot use 0 for matrix id number')

      mids.add(mid)

      mat = clean_mat(np.loadtxt(f, dtype = dtype))
      f.close()
      
      name = "_".join([funcn, str(mid), 'b0'])
      src += generate_mm(mat, dtype = dtype, alpha = 1.0, beta = 0.0, platform = platform, funcn = name, order = order)
      name = "_".join([funcn, str(mid), 'b1'])
      src += generate_mm(mat, dtype = dtype, alpha = 1.0, beta = 1.0, platform = platform, funcn = name, order = order)
      
    # Data type
    dtype = np.dtype(dtype).type
    if dtype == np.float32:
        dtype = 'float'
    elif dtype == np.float64:
        dtype = 'double'
    else:
        raise ValueError('Invalid floating point data type')

    # Setup dispatch function
    # Template arguments
    tplargs = {'dtype': dtype, 'mids': mids, 'funcn': funcn, 'order': order}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, 'kernels/dispatch.{0}.mako'.format(platform))
    src += Template(tpl).render(**tplargs)

    # At single precision suffix all floating point constants by 'f'
    if dtype == 'float':
        src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                     r'\g<0>f', src)

    # Return the source
    return src

def clean_mat(mat, tol=1e-10):

  arr = mat.copy()
  # Flush small elements to zero
  arr[np.abs(arr) < tol] = 0

  # Coalesce similar elements
  amfl = np.abs(arr.flat)
  amix = np.argsort(amfl)

  i, ix = 0, amix[0]
  for j, jx in enumerate(amix[1:], start=1):
    if amfl[jx] - amfl[ix] >= tol:
      if j - i > 1:
        amfl[amix[i:j]] = np.median(amfl[amix[i:j]])
        i, ix = j, jx

  if i != j:
    amfl[amix[i:]] = np.median(amfl[amix[i:]])

  # Fix up the signs and assign
  arr.flat = np.copysign(amfl, arr.flat)

  return arr 

