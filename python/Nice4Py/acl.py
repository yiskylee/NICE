import Nice4Py
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class ACL(object):
  def __init__(self, type, method, device):
    self.device = device
    self.params = {'q': 1, 'kernel': 'Gaussian', 'debug': 0.0,
                   'lambda': 1.0, 'sigma': 0.5, 'verbose': 0.0, 'max_time': 30,
                   'method': 'ISM'}

    # Call this so the mkl libarary is loaded before C++ is called
    pairwise_distances(np.zeros((4,4)), Y=None, metric='euclidean')

    if type == 'float':
      self.data_type = np.float32
      self.acl = Nice4Py.ACL(method, device)
    elif type == "double":
      self.data_type = np.float
      self.acl = Nice4Py.ACLDouble(method, device)

    self.acl.SetupParams(self.params)

    self.profiling = {}

  def set_params(self, c=None, q=None, kernel=None, debug=None, verbose=None,
                 Lambda=None, sigma=None, max_time=None, method=None,
                 vectorization=None):
    if c is not None:
      self.params['c'] = c
    if q is not None:
      self.params['q'] = q
    if kernel is not None:
      self.params['kernel'] = kernel
    if debug is not None:
      self.params['debug'] = debug
    if verbose is not None:
      self.params['verbose'] = verbose
    if Lambda is not None:
      self.params['lambda'] = Lambda
    if sigma is not None:
      self.params['sigma'] = sigma
    if max_time is not None:
      self.params['max_time'] = max_time
    if method is not None:
      self.params['method'] = method
    if vectorization is not None:
      self.params['vectorization'] = vectorization
    self.acl.SetupParams(self.params)

  def Fit(self, X=None, y=None):
    if X is None:
      self.acl.Fit()
    else:
      self.n = X.shape[0]
      self.d = X.shape[1]
      if y is None:
        self.acl.Fit(X, self.n, self.d)
      else:
        self.acl.Fit(X, self.n, self.d, y, y.shape[0], y.shape[1])

  def Predict(self):
    pred = np.empty((self.n, 1), dtype=self.data_type)
    self.acl.Predict(pred, self.n, 1)
    return pred.flatten()

  def GenProfile(self, per_iter_list):
    self.acl.GetProfiler(self.profiling)
    self.GetTimePerIter(per_iter_list)
    return self.profiling

  def GetTimePerIter(self, per_iter_list):
    num_iters = self.profiling['num_iters']
    time_per_iter = np.zeros(num_iters)
    for name in per_iter_list:
      self.acl.GetTimePerIter(time_per_iter, num_iters, name)
      self.profiling[name+'_per_iter'] = np.array(time_per_iter.copy())
