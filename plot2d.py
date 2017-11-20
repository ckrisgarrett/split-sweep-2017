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
print 'Max/Min of f0: ', max(f[0:len(f):NB]), min(f[0:len(f):NB])
print 'Max/Min of f1: ', max(f[1:len(f):NB]), min(f[1:len(f):NB])
print 'Max/Min of f2: ', max(f[2:len(f):NB]), min(f[2:len(f):NB])
print 'Max/Min of f3: ', max(f[3:len(f):NB]), min(f[3:len(f):NB])
print 'Max/Min of f4: ', max(f[4:len(f):NB]), min(f[4:len(f):NB])

f = f[0:len(f):NB]
f = f.reshape([NX,NX,NV,NV])

#f11 = f[:,NX/2,:,NV/2]
#f12 = f[:,NX/2,NV/2,:]
#f21 = f[NX/2,:,:,NV/2]
#f22 = f[NX/2,:,NV/2,:]

f11 = numpy.zeros((NX,NV))
f12 = numpy.zeros((NX,NV))
f21 = numpy.zeros((NX,NV))
f22 = numpy.zeros((NX,NV))
for i in range(0,NX):
    for j in range(0,NV):
        f11 = f11 + f[:,i,:,j] * dx * dv
        f12 = f12 + f[:,i,j,:] * dx * dv
        f21 = f21 + f[i,:,:,j] * dx * dv
        f22 = f22 + f[i,:,j,:] * dx * dv

#f11 = f11.reshape([NX,NV])
#f12 = f12.reshape([NX,NV])
#f21 = f21.reshape([NX,NV])
#f22 = f22.reshape([NX,NV])

f11 = numpy.transpose(f11)
f12 = numpy.transpose(f12)
f21 = numpy.transpose(f21)
f22 = numpy.transpose(f22)

print 'Max/Min of f11: ', f11.max(), f11.min()
print 'Max/Min of f12: ', f12.max(), f12.min()
print 'Max/Min of f21: ', f21.max(), f21.min()
print 'Max/Min of f22: ', f22.max(), f22.min()

minf = min(f11.min(), f12.min(), f21.min(), f22.min())
maxf = max(f11.max(), f12.max(), f21.max(), f22.max())


#Plots
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
im = ax1.imshow(f11, origin='lower',extent=[AX,BX,AV,BV],aspect='auto', vmin=minf, vmax=maxf)
#ax1.colorbar()
ax2.imshow(f12, origin='lower',extent=[AX,BX,AV,BV],aspect='auto', vmin=minf, vmax=maxf)
#ax2.colorbar()
ax3.imshow(f21, origin='lower',extent=[AX,BX,AV,BV],aspect='auto', vmin=minf, vmax=maxf)
#ax3.colorbar()
im = ax4.imshow(f22, origin='lower',extent=[AX,BX,AV,BV],aspect='auto', vmin=minf, vmax=maxf)
#ax4.colorbar()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# Save figure and show
plt.savefig('plot2d.pdf')
plt.show()




