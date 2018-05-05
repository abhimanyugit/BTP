from scipy import signal
import matplotlib.pyplot as plt 
import numpy as np

# set sampling frequency in timeframe 0 to 1 at 50hz
#fs = np.linspace(0,1,50, endpoint=False)
fs = np.linspace(0,1,6300, endpoint=False)

#generate a square wave at 100Hz sampled at 6300
fa = 200
k = 1.0/(4*fa)
fb = 600

wave1 = (signal.square(2*np.pi*fa*fs) + 1)/2
ax = plt.subplot(2, 1, 1)
ax.set_title("wave1")
plt.plot(fs,wave1,'-') #r+ is red plusses for plot
plt.ylim(-0.5,1.5)
plt.xlim(0,0.05)

#plt.figure()
wave2 = (signal.square(2*np.pi*fb*fs) + 1)/2
ax2 = plt.subplot(2, 1, 2)
ax2.set_title("wave1 + wave2")
plt.plot(fs,wave1 + wave2)
plt.ylim(-0.5,2.5)
plt.xlim(0,0.05)
plt.close()

x1 = wave1 #+ wave2

plt.figure()
s_n = signal.square(2*np.pi*fa*fs)
ax3 = plt.subplot(2, 1, 1)
ax3.set_title("s_n")
plt.plot(fs,s_n)
plt.ylim(-1.5,1.5)
plt.xlim(0,0.05)

c_n = signal.square(2*np.pi*fa*(fs + k))
ax4 = plt.subplot(2, 1, 2)
ax4.set_title("c_n")
plt.plot(fs,c_n)
plt.ylim(-1.5,1.5)
plt.xlim(0,0.05)
plt.close()

plt.figure()
ax5 = plt.subplot(2, 1, 1)
ax5.set_title("x_n * s_n")
plt.plot(fs, x1 * s_n)
plt.ylim(-1.5,1.5)
plt.xlim(0,0.05)

ax6 = plt.subplot(2, 1, 2)
ax6.set_title("x_n * c_n")
plt.plot(fs,x1 * c_n)
plt.ylim(-1.5,1.5)
plt.xlim(0,0.05)
plt.close()

b = np.ones(120)/120
a = [1]
print b,a

plt.figure()
w, h = signal.freqz(b,a)
plt.title('Digital filter frequency response')
plt.plot(w, 20*np.log10(np.abs(h)))
plt.title('Digital filter frequency response')
plt.ylabel('Amplitude Response [dB]')
plt.xlabel('Frequency (rad/sample)')
plt.grid()
plt.ylim(-90,0)
plt.close()

plt.figure()
b1, a1 = signal.butter(6, 0.0016, 'low', analog=False)
w, h = signal.freqz(b1, a1)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xlim(0,3.5)
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.close()
#sig1 = np.sin(2 * np.pi * 106 * fs)
#sig2 = np.cos(2 * np.pi * 600 * fs
#pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)
#plt.subplot(2, 1, 1)
#plt.plot(fs, sig1)
#plt.subplot(2, 1, 2)
#plt.plot(fs, sig2)
#plt.ylim(-1.5, 1.5)

rxs = x1 * s_n
rxc = x1 * c_n

I_n = signal.filtfilt(b,a,rxs)
Q_n = signal.filtfilt(b,a,rxc)

alpha = I_n * I_n + Q_n * Q_n
V = np.pi * 0.5 * np.sqrt(alpha)
plt.figure()
#y = signal.filtfilt(b,a,sig1+sig2)
ax_1 = plt.subplot(3,1,1)
ax_1.set_title("I_n")
plt.plot(fs,I_n)
#plt.xlim(0,1)
ax_2=plt.subplot(3,1,2)
ax_2.set_title("Q_n")
plt.plot(fs,Q_n)
ax_3=plt.subplot(3,1,3)
ax_3.set_title("V")
plt.plot(fs,V)
#plt.close()

plt.figure()
plt.subplot(2,2,1)
plt.plot(abs(np.fft.rfft(rxs)), label = "rxs")
plt.xlim(-100,7000)
plt.legend()
plt.subplot(2,2,2)
plt.plot(abs(np.fft.rfft(s_n)), label = "s_n")
plt.xlim(-100,7000)
plt.legend()
plt.subplot(2,2,3)
plt.plot(abs(np.fft.rfft(x1)), label = "x1")
plt.xlim(-100,7000)
plt.legend()
plt.subplot(2,2,4)
plt.plot(abs(np.fft.rfft(I_n)), label = "I_n")
plt.xlim(-100,7000)
plt.legend()
#print x1.shape
#plt.close()

plt.figure()
f1, Pxx_den = signal.periodogram(x1, 6300, 'flattop',scaling='spectrum')
plt.semilogy(f1, Pxx_den)
plt.xlim(-100,3500)
plt.close()
plt.show()