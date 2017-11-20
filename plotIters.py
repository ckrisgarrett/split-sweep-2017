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

data1 = data[TIME_STEPS*NUM_DATA+4:]
efieldIters = data1[0:NUM_EFIELD]
sweepIters = data1[NUM_EFIELD:NUM_EFIELD+NUM_SWEEP]
eulerIters = data1[NUM_EFIELD+NUM_SWEEP:]

print 'Average E-Field Iters: ', numpy.mean(efieldIters)
print 'Average Sweep Iters: ', numpy.mean(sweepIters)
print 'Average Euler Iters: ', numpy.mean(eulerIters)


fig, (ax1,ax2,ax3) = plt.subplots(3,1)
fig.subplots_adjust(hspace=0.5)
ax1.plot(efieldIters, '*')
ax2.plot(sweepIters, '*')
ax3.plot(eulerIters, '*')
ax1.set_title('E-Field Iters')
ax2.set_title('Sweep Iters')
ax3.set_title('Euler Iters')
plt.savefig('plot-iters.pdf')
plt.show()

