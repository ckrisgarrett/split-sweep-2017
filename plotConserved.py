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


print 'Mass: ', mass[0]
print 'momentum1: ', momentum1[0]
print 'energy: ', energy[0]
print 'l2norm: ', l2norm[0]
print 'Mass (max-min)', max(mass) - min(mass)


# Show conserved quanitites
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2)
fig.subplots_adjust(hspace=0.5)
ax1.plot(time, mass)
ax2.plot(time, momentum1)
ax3.plot(time, energy)
ax4.plot(time, l2norm)
ax5.plot(time, momentum2)
ax1.set_title('mass')
ax2.set_title('momentum1')
ax3.set_title('energy')
ax4.set_title('l2norm')
ax5.set_title('momentum2')
plt.savefig('plot-conserved.pdf')
plt.show()


## Show difference from initial
##mass = numpy.maximum(abs(mass - mass[0]), 1e-30)
#momentum = abs(momentum - momentum[0])
#energy = abs(energy - energy[0]) / abs(energy[0])
#l2norm = abs(l2norm - l2norm[0]) / abs(l2norm[0])

##plt.figure()
#fig, (ax1,ax2,ax3) = plt.subplots(3,1)
#ax1.semilogy(time, momentum)
#ax2.semilogy(time, energy)
#ax3.semilogy(time, l2norm)
#ax1.set_title('momentum (Absolute Error)')
#ax2.set_title('energy (Relative Error)')
#ax3.set_title('l2norm (Relative Error)')
##plt.semilogy(time, mass, time, momentum, time, energy, time, l2norm)
##plt.legend(('Mass ', 'Momentum ', 'Energy ', 'L2 Norm '))
#plt.savefig('plot-conserved-diff.pdf')
#plt.show()


