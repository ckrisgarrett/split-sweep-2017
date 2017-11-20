import numpy
import matplotlib.pyplot as plt
import math
import sys
import scipy


filename = sys.argv[1]
data = numpy.loadtxt(filename)

NX = int(data[0])
NV = int(data[1])
NB = int(data[2])
AX = float(data[3])
BX = float(data[4])
AV = float(data[5])
BV = float(data[6])

f = data[7:len(data)]

print 'NX = ', NX
print 'NV = ', NV
print 'NB = ', NB
print 'AX = ', AX
print 'BX = ', BX
print 'AV = ', AV
print 'BV = ', BV
print 'len(f) = ', len(f)


dx = (BX - AX) / NX
dv = (BV - AV) / NV
x = numpy.arange(AX + dx/2.0, BX, dx)
v = numpy.arange(AV + dv/2.0, BV, dv)


# Plot f
#for i in range(0,NX*NV):
#    f[i] = 0.25 * (fdata[i*4+0] + fdata[i*4+1] + fdata[i*4+2] + fdata[i*4+3])
print 'Max/Min of f0: ', max(f[0:len(f):3]), min(f[0:len(f):3])
print 'Max/Min of f1: ', max(f[1:len(f):3]), min(f[1:len(f):3])
print 'Max/Min of f2: ', max(f[2:len(f):3]), min(f[2:len(f):3])
f = f[0:len(f):3]
f = f.reshape([NX,NV])
f = numpy.transpose(f)

plt.imshow(f, origin='lower',extent=[AX,BX,AV,BV],aspect='auto')#, interpolation='none')
#plt.clim([0.0,1.0])
#print numpy.shape(f1)
#f1 = numpy.transpose(f1)
#plt.pcolormesh(x1,v1,f1)
plt.colorbar()
#plt.title('f')




# Save figure and show
plt.savefig('plot.pdf')
plt.show()




