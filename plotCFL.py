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

print 'xCFL:', max(xCFL), 'ECFL:', max(maxECFL)

plt.plot(time, xCFL, time, maxECFL, time, meanECFL, time, minECFL)
plt.legend(('X CFL', 'Max E CFL', 'Mean E CFL', 'Min E CFL'))
plt.savefig('plot-cfl.pdf')
plt.show()

