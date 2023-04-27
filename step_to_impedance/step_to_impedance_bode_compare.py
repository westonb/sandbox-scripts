from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import numpy as np
import pandas
import control


# script for fitting the measured data from analog discovery 2 to a two pole transfer function
# written for EE254 lab 2


#
# You may need to change these things (unlikely except for input file name)
#

input_file = "STEP.CSV" #name of input data. Relative to location of script

TIME_STOP = 20E-3 #step period is ~50ms
I_STEP = 5.0/10.2 

FILTER_CUTOFF = 50E3
FILTER_ORDER = 10

#load data from file
all_data = pandas.read_csv(input_file,comment='#')


meas_time = all_data["in s"].to_numpy()
ref_sig = all_data["C3 in V"].to_numpy()
output_sig = all_data["C2 in V"].to_numpy() * -1.0 #signal is inverted based on s


index_trigger = np.argmin(abs(meas_time))
index_end = np.argmin(abs(meas_time - TIME_STOP))

seq_length = index_end - index_trigger

index_start = index_trigger - seq_length


step_sig = output_sig[index_start:index_end]
time_sig = meas_time[index_start:index_end]

time_step = time_sig[1] - time_sig[0]
f_sample = 1/time_step

N = len(step_sig)
print("Sequence Length: %d" % (N))

#filter signal
sos = signal.butter(FILTER_ORDER, FILTER_CUTOFF, 'low', fs=f_sample, output='sos')

step_sig_filtered = signal.sosfilt(sos,step_sig) #butter filter adds time delay, causes phase error. Do not use
plt.xlabel("Time", fontsize=18)
plt.title("Step Response", fontsize=22)
plt.ylabel("Volts", fontsize=18)
plt.plot(time_sig, step_sig*-1, label="Captured Step")
plt.plot(time_sig, step_sig_filtered*-1, label="Filtered Step")
plt.legend()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("step_response.png", dpi=300)
plt.show()



step_sig_windowed = signal.windows.hann(N) * step_sig

step_sig_fft = fft(step_sig_windowed)



fft_freqs = fftfreq(N, time_step)

#array to differentiate signal
fft_deriv = 2*np.pi*1.0j*fft_freqs 

#array to time shift signal 
T0 = time_step * seq_length
fft_time_shift = np.exp(-2j*np.pi*fft_freqs*T0)

#multiply step fft by deriv to get impulse
fft_impulse = step_sig_fft * fft_deriv * fft_time_shift




#scale factor for fft to transfer function
#time step is fft scale, 2 is only positive freqs scale, Istep is impulse scaling
scale_factor = time_step / I_STEP 
#clip to prevent taking log of zero 
#only dealing with positive frequencies
fft_impulse_normed = np.clip( np.abs(fft_impulse[0:N//2])*scale_factor, 1E-6, 1E6)
fft_impulse_phase = np.angle(fft_impulse[0:N//2], deg=True)

min_db = -40.0
max_db = 20.0

mag_db = np.clip(20*np.log10(fft_impulse_normed), min_db, max_db)

positive_freqs = fft_freqs[:N//2]

freq_min = 80.0
freq_max = 20E3 

freq_l_index = np.argmin(np.abs(positive_freqs - freq_min))
freq_h_index = np.argmin(np.abs(positive_freqs - freq_max))


# load data 
bode_file = "out_bode.CSV"
bode_data = pandas.read_csv(bode_file,comment='#')

bode_freq = bode_data["Frequency in Hz"].to_numpy()
bode_mag = bode_data["Gain in dB"].to_numpy() + 6.0 - 0.26
bode_angle_unwrapped = bode_data["Phase in °"].to_numpy() + 180 #test setup is inverted

wrap_phase = np.vectorize( lambda x: x-360 if x>180 else x + 180 if x<-180 else x)
bode_angle = wrap_phase(bode_angle_unwrapped)


bode_index_start = np.argmin(np.abs(bode_freq-freq_min))
bode_index_end = np.argmin(np.abs(bode_freq - freq_max))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(positive_freqs[freq_l_index:freq_h_index], mag_db[freq_l_index:freq_h_index], label="from step")
ax1.plot(bode_freq[bode_index_start:bode_index_end], bode_mag[bode_index_start:bode_index_end], label="from FRA")
ax1.set_ylabel("Magnitude (dB)", fontsize=16, labelpad=12)
ax1.set_xscale("log")
ax1.margins(x=0)
ax1.set_ylim(-40,-10)
ax1.legend()
ax1.grid(True)




ax2.plot(positive_freqs[freq_l_index:freq_h_index],fft_impulse_phase[freq_l_index:freq_h_index], label="from step")
ax2.plot(bode_freq[bode_index_start:bode_index_end], bode_angle[bode_index_start:bode_index_end], label="from FRA")
ax2.set_ylabel("Angle (deg)", fontsize=16, labelpad=12)
ax2.grid(True)

ax2.set_xlabel(r"Frequency (Hz)", fontsize=18)
ax2.yaxis.set_ticks([-90, -45, 0, 45, 90])
ax2.set_ylim(-90,90)
ax2.margins(x=0)
fig.suptitle("Output Impedance", fontsize=22)





plt.tight_layout()
plt.margins(x=0)
plt.savefig("output_impedance_compare.png", dpi=300)
plt.show()





