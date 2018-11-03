import numpy as np
from math import cos
PI=3.1415926
import sys
from scipy.signal import butter,filtfilt

import soundfile
import pysptk
import pyreaper
import pyworld
import numpy as np
import matplotlib.pyplot as plt

PLOT=0

wav,fs=soundfile.read(sys.argv[1])
_f0,t=pyworld.dio(wav,fs)
f0=pyworld.stonemask(wav,_f0,t,fs)

t*=fs

tt=np.linspace(0,1,wav.shape[0])
#f0=np.interp(tt,t,f0)
f0=np.repeat(f0,fs*0.005)
print(f0.shape,wav.shape)




def highpass(x,fs,cutoff,axis=0):
    def butter_highpass(cutoff,fs=fs,order=3):
        nyq=0.5*fs
        normal_cutoff=cutoff/nyq
        b,a=butter(order,normal_cutoff,btype='high',analog=False)
        return b,a
    b,a=butter_highpass(cutoff)
    high=filtfilt(b,a,x,axis=axis)
    return high

mix=1.0

xm=[1]*wav.shape[0]

def smooth_f0(f0):
    f0=f0[f0>0]
    return np.mean(f0)

SMOOTH_RANGE=int(fs*0.1)

for k,h in zip([2,3,4],[0.75,0.7,0.7]):
    ph=PI/2
    preph=PI/2
    for t in range(0,wav.shape[0]):
        if t>500 and f0[t-500:t].all():
            if t<SMOOTH_RANGE:
                s_f0=smooth_f0(f0[:SMOOTH_RANGE])
            else:
                s_f0=smooth_f0(f0[t-SMOOTH_RANGE:t+SMOOTH_RANGE])
            ph=PI/2+2*PI*(s_f0/k+(np.random.rand(1)[0]*2-1)*0)/fs*t-preph
            prph=ph
            xm[t]+=h*0.9*cos(ph)
        else:
            ph=PI/2


if PLOT>0:
    print("plot")
    plt.figure()
    plt.plot(tt,xm)
    plt.show()

#xm=smooth(np.array(xm),int(0.005*fs))

y=[0]*wav.shape[0]
for i in range(0,wav.shape[0]):
    if 1:
        y[i]=wav[i]*xm[i]
    else:
        y[i]=wav[i]

ysub=y-wav

print(type(ysub))
wav=0.5*wav+mix*highpass(ysub,fs,1000)

soundfile.write("test.wav",wav,fs)


