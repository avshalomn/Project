#### defferent inputs and targets ####
import numpy as np

def random_inputs(shape,complex = False):
  rows, cols = shape[0],shape[1]
  inps = np.random.rand(shape[0],shape[1])
  if complex:
    inps = np.array([[np.random.random + 0.00000001j for c in range(cols)] for r in range(rows)])
  return inps

def linear_targets(shape,complex = False)
  rows, cols = shape[0],shape[1]
  targets = np.array([np.arange(cols)/cols + (complex*0.0000001j) for r in range(rows)])
     
  return targets
