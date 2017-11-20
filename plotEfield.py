import numpy
import matplotlib.pyplot as plt
import math
import sys
import scipy


gamma = -0.1533
filename = sys.argv[1]
data = numpy.loadtxt(filename)


TIME_STEPS = data[0]
NUM_EFIELD = data[1]
NUM_SWEEP = data[2]
NUM_EULER = data[3]
NUM_DATA = 12

data1 = data[4:TIME_STEPS*NUM_DATA+4]

time        = data1[0:TIME_STEPS*NUM_DATA:NUM_DATA]
l2efield1   = data1[1:TIME_STEPS*NUM_DATA:NUM_DATA]
l2efield2   = data1[2:TIME_STEPS*NUM_DATA:NUM_DATA]
mass        = data1[3:TIME_STEPS*NUM_DATA:NUM_DATA]
momentum1   = data1[4:TIME_STEPS*NUM_DATA:NUM_DATA]
momentum2   = data1[5:TIME_STEPS*NUM_DATA:NUM_DATA]
energy      = data1[6:TIME_STEPS*NUM_DATA:NUM_DATA]
l2norm      = data1[7:TIME_STEPS*NUM_DATA:NUM_DATA]
minECFL     = data1[8:TIME_STEPS*NUM_DATA:NUM_DATA]
maxECFL     = data1[9:TIME_STEPS*NUM_DATA:NUM_DATA]
meanECFL    = data1[10:TIME_STEPS*NUM_DATA:NUM_DATA]
xCFL        = data1[11:TIME_STEPS*NUM_DATA:NUM_DATA]


if l2efield2[0] <= 0:
    l2efield2 = abs(l2efield2) + 1e-12

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.semilogy(time, l2efield1, time, l2efield1[0] * numpy.exp(gamma*time))
ax2.semilogy(time, l2efield2, time, l2efield2[0] * numpy.exp(gamma*time))
plt.savefig('plot-efield.pdf')
plt.show()

