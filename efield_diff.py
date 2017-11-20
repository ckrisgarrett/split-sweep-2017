import numpy
import matplotlib.pyplot as plt
import math
import sys
import scipy


file1 = sys.argv[1]
file2 = sys.argv[2]
data1 = numpy.loadtxt(file1)
data2 = numpy.loadtxt(file2)
diff = data1 - data2

averagePhiDiff = numpy.average(diff[1:10,0])
diff[:,0] = diff[:,0] - averagePhiDiff;

print diff[:,2]

print 'Average Phi Diff:', averagePhiDiff
print 'Max Diff (Phi, E1, E2):', max(abs(diff[:,0])), max(abs(diff[:,1])), max(abs(diff[:,2]))
