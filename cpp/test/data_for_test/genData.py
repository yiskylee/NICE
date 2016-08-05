#!/usr/bin/python
import numpy as np 
import sys

if len(sys.argv) != 5:
  print 'Usage: python genKmeansData.py numCenters numPoints dim variance'
  sys.exit(-1)
numCenters = int(sys.argv[1])
numPoints = int(sys.argv[2])
dim = int(sys.argv[3])
dia = int(sys.argv[4])
print 'Number of Centers: ', numCenters
print 'Number of Points: ', numPoints
print 'Number of Dimensions: ', dim
print 'Variance: ', dia

sideLength = 500
cov = np.identity(dim) * dia
data = np.empty([numCenters*numPoints, dim])

for i in np.arange(numCenters):
  mean = np.random.uniform(0, sideLength, dim)
  data[i*numPoints, :] = mean
  points = np.random.multivariate_normal(mean, cov, numPoints-1)
  data[i*numPoints+1:(i+1)*numPoints, :] = points

np.savetxt('data_k'+str(numCenters)+'_p'+str(numPoints)+'_d'+str(dim)+'_c'+str(dia)+'.txt', data, delimiter=' ')
