import Nice4Py
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class ACL(object):
    def __init__(self, data_type, method, device):
        self.device = device
        self.params = {'kernel': 'Gaussian', 'debug': 0.0,
                       'lambda': 1.0, 'verbose': 0.0, 'vectorization': 1.0}

        # Call this so the mkl libarary is loaded before C++ is called
        pairwise_distances(np.zeros((4, 4)), Y=None, metric='euclidean')

        if data_type == 'float':
            self.data_type = np.float32
            self.acl = Nice4Py.ACL(method, device)
        elif data_type == "double":
            self.data_type = np.float
            self.acl = Nice4Py.ACLDouble(method, device)

        self.acl.SetupParams(self.params)

        self.profiling = {}

    def set_params(self, c=None, q=None, kernel=None, debug=None, verbose=None,
                   Lambda=None, sigma=None, max_time=None, vectorization=None,
                   thresh1=None, thresh2=None):
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
        if vectorization is not None:
            self.params['vectorization'] = vectorization
        if thresh1 is not None:
            self.params['threshold1'] = thresh1
        if thresh2 is not None:
            print thresh2
            self.params['threshold2'] = thresh2

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

    def GenProfile(self, each_iter_list=None):
        self.acl.GetProfiler(self.profiling)
        if each_iter_list is not None:
            self.GetTimeInEachIter(each_iter_list)

        return self.profiling

    def GetTimeInEachIter(self, each_iter_list):
        num_iters = self.profiling['num_iters']
        time_in_each_iter = np.zeros(num_iters)
        for name in each_iter_list:
            self.acl.GetTimeInEachIter(time_in_each_iter, num_iters, name)
            self.profiling[name +
                           '_time_in_each_iter'] = np.array(time_in_each_iter.copy())

    def OutputConfigs(self):
        self.acl.OutputConfigs()
