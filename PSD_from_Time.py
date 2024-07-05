# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:08:56 2023

@author: acc2105
"""
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


def Mide_FFT_PSD(datalist,fActual):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %[freq, xdft, psdx, phase] = FFT_PSD_Spectrogram(datalist,fActual)
    % Given a dataset this will calculate the FFT PSD and phase
    %
    % Inputs:
    %   datalist = two column array with time in first column, data to analyze
    %       in second
    %   fActual = sample rate of the data in Hertz
    %
    % Outputs:
    %   freq = frequency bins for FFT and PSD
    %   xdft = amplitude of FFT in native units of datalist
    %   phase = phase response of FFT in radians
    %   psdx = amplitude of FFT in native units of datalist squared divided by
    %       Hz
    %
    %MATLAB may run out of memory for large files
    %
    %Mide Technology
    %Date: 06.08.2016
    %Rev: 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    Fs = fActual
    x = datalist[:,1]
    N = len(x)
    freq = np.arange(N//2)*Fs/N
    xdft = np.fft.fft(x)
    xdft = xdft[0:N//2]
    psdx = (1/(Fs*N)) * abs(xdft)**2
    psdx[1:-1] = 2*psdx[1:-1]
    xdft = 1/N*xdft
    xdft[1:-1] = 2*xdft[1:-1]
    cdft = xdft
    phase = np.angle(xdft)
    xdft = abs(xdft)
    
    return freq, xdft, psdx, phase, cdft


"""
##### Creating a sample array for testing the function
sigfreq = 2*np.pi*10
time = np.arange(920)/1e3
mysig = np.sin(time*sigfreq)
datalist = np.array([time,mysig]).T
Fs = 1/(time[1]-time[0])
f,x,p,phase = Mide_FFT_PSD(datalist,Fs)
"""

# Declaration of Standard Spectrum of PSD for ISO TRUCK TEST
THM = np.array([[10 , 0.2],
       [13, 0.2],
       [19, 0.2],
       [20, 0.2],
       [50, 0.01],
       [100, 0.001],
       [500, 0.001],
       [2000, 0.0001],
     ])


# Loading data from file - Change the filepath if needed
loaded_data = np.loadtxt(r'E:\Programming\PSD (python)\data\random_z004.csv', skiprows=23, delimiter=',')

# Sampling frequency, as read from file
Fs = 8192.00 # from file
# Construction of time array
time = np.arange(len(loaded_data))/Fs
# Description of file contents (shared by MAGNA through email)
description = {
    0: 'Plate Z',
    1: 'Comp. Cover X',
    2: 'Comp. Cover Y',
    3: 'Comp. Cover Z',
    4: 'Comp. Lateral X',
    5: 'Comp. Lateral Y',
    6: 'Comp. Lateral Z',
    }

# Selection of data to plot over time
i_ = 0
fg, myax = plt.subplots(figsize=(15,7))
myax.plot(time, loaded_data[:,i_], label=description[i_])
myax.set(
    # xlim = (10, 2000),
    # ylim = (1e-5, 10),
    xlabel = 'Time (s)',
    ylabel = 'Accel (g)',
    # xscale = 'log',
    # yscale = 'log',
    )
myax.grid()
myax.legend()


# Creation of time-mask to build the PSD for comparison
mask = (time>60)&(time<80)
fg, myax = plt.subplots(figsize=(15,7))
for i_ in description.keys():
    datalist = np.array([time[mask],loaded_data[mask,i_]]).T
    freq, xfft, xpsd, xpha, _ = Mide_FFT_PSD(datalist,Fs)
    myax.plot(freq, xpsd, label=description[i_])
# Plotting the ISO Truck standard spectrum over data plots
myax.plot(THM[:,0],THM[:,1], label='THM')
myax.plot(THM[:,0],THM[:,1]*1.5, label='1.5dB tol')
myax.plot(THM[:,0],THM[:,1]/1.5)
myax.set(
    xlim = (10, 2000),
    ylim = (1e-5, 10),
    xlabel = 'Frequency (Hz)',
    ylabel = 'Accel (g²/Hz)',
    xscale = 'log',
    yscale = 'log',
    )
myax.grid()
myax.legend()




#%%
g = 9.8166
t_start = 60
t_finish = 100
f_start = 20
f_finish = 100
mask = (time>t_start)&(time<t_finish)
plate_acc = np.array([time[mask],loaded_data[mask,0]]).T
freq, plate_fft, plate_psd, plate_pha, plate_cft = Mide_FFT_PSD(plate_acc,Fs)
plate_cft = plate_cft[1:]-plate_cft[0]

print(f'Analysis from {t_start}s to {t_finish}s ({f_start}<f<{f_finish} Hz)')

freq = freq[1:]
plate_dis = 1000*g*abs(plate_cft[(freq>f_start)&(freq<f_finish)]/(2*np.pi*freq[(freq>f_start)&(freq<f_finish)])**2)
print(f'Plate Z (ref.): {max(plate_dis):.3f}mm')

fg, myax = plt.subplots(figsize=(15,7))
for i_ in description.keys():
    desired_acc = np.array([time[mask],loaded_data[mask,i_]]).T
    freq, desired_fft, desired_psd, desired_pha, desired_cft = Mide_FFT_PSD(desired_acc,Fs)
    
    freq = freq[1:]
    
    desired_cft = desired_cft[1:]-desired_cft[0]
    
    desired_frf = 1000*g*abs(desired_cft[(freq>f_start)&(freq<f_finish)]-plate_cft[(freq>f_start)&(freq<f_finish)])/(2*np.pi*freq[(freq>f_start)&(freq<f_finish)])**2
    myax.plot(freq[(freq>f_start)&(freq<f_finish)], desired_frf, label=description[i_])
    
    print(f'{description[i_]} (relative to plate): {max(desired_frf):.3f}mm')

myax.set(
    xlim = (f_start, f_finish),
    # ylim = (1e-5, 1e5),
    xlabel = 'Frequency (Hz)',
    ylabel = 'Disp. FRF [mm/mm]',
    xscale = 'log',
    yscale = 'log',
    )
myax.grid()
myax.legend()


    
    
#%%
# fg, myax = plt.subplots(figsize=(15,7))
# myax.plot(freq, 1000*g*abs(covrz_cft/(plate_cft*(2*np.pi*freq)**2)), label='fft')
# # myax.plot(freq, covrz_psd, label='cover-z')
# myax.set(
#     # xlim = (1, 100),
#     # ylim = (1e-5, 1e5),
#     xlabel = 'Frequency (Hz)',
#     ylabel = 'Accl. FRF [g/g]',
#     xscale = 'log',
#     yscale = 'log',
#     )
# myax.grid()
# myax.legend()


#%%
g = 9.8661
mask = (time>0)&(time<140)
plate_acc = g*loaded_data[mask,0]
plate_vel = integrate.cumtrapz(plate_acc, time[mask], initial = 60)
plate_dis = integrate.cumtrapz(plate_vel, time[mask], initial = 60)

fg, myax = plt.subplots(figsize=(15,7))
myax.plot(time[mask],plate_dis)
myax.set(
    xlim = (61, 81),
    ylim = (-5,1),
    )

