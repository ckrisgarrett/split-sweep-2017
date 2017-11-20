import numpy
import matplotlib.pyplot as plt
import math
import sys
import scipy
import sys


filename1 = sys.argv[1]
filename2 = sys.argv[2]
data1 = numpy.loadtxt(filename1)
data2 = numpy.loadtxt(filename2)

NX1 = int(data1[0]);    NX2 = int(data2[0])
NV1 = int(data1[1]);    NV2 = int(data2[1])
NB1 = int(data1[2]);    NB2 = int(data2[2])
AX1 = float(data1[3]);  AX2 = float(data2[3])
BX1 = float(data1[4]);  BX2 = float(data2[4])
AV1 = float(data1[5]);  AV2 = float(data2[5])
BV1 = float(data1[6]);  BV2 = float(data2[6])

if NX1 < NX2:
    print 'NX1 < NX2'
    sys.exit()
if NV1 < NV2:
    print 'NV1 < NV2'
    sys.exit()
if NB1 != NB2:
    print 'NB1 != NB2'
    sys.exit()
if AX1 != AX2:
    print 'AX1 != AX2'
    sys.exit()
if BX1 != BX2:
    print 'BX1 != BX2'
    sys.exit()
if AV1 != AV2:
    print 'AV1 != AV2'
    sys.exit()
if BV1 != BV2:
    print 'BV1 != BV2'
    sys.exit()


f1 = data1[7:len(data1)]
f2 = data2[7:len(data2)]

f1 = f1[0:len(f1):3]
f1 = f1.reshape([NX1,NV1])
f2 = f2.reshape([NX2,NV2,3])
f2fine = numpy.zeros([NX1,NV1])

LX = NX1 / NX2
LV = NV1 / NV2
DX1 = (BX1 - AX1) / NX1
DV1 = (BV1 - AV1) / NV1
DX2 = (BX2 - AX2) / NX2
DV2 = (BV2 - AV2) / NV2
for i in range(0,NX1):
    for j in range(0,NV1):
        i2 = i / LX
        j2 = j / LV
        dx = -1.0 + 1.0 / LX + 2.0 / LX * (i % LX)
        dv = -1.0 + 1.0 / LV + 2.0 / LV * (j % LV)
        f2fine[i,j] = f2[i2,j2,0] + f2[i2,j2,1] * dx + f2[i2,j2,2] * dv

print numpy.linalg.norm(f1 - f2fine) / numpy.linalg.norm(f1)




