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

step_sig_filtered = signal.sosfilt(sos,step_sig)

plt.plot(time_sig, step_sig)
plt.plot(time_sig, step_sig_filtered)
plt.show()


step_sig_windowed = signal.windows.hann(N) * step_sig_filtered

step_sig_fft = fft(step_sig_windowed)



fft_freqs = fftfreq(N, time_step)

#array to differentiate signal
fft_deriv = 2*np.pi*1.0j*fft_freqs 

#multiply step fft by deriv to get impulse
fft_impulse = step_sig_fft * fft_deriv


#scale factor for fft to transfer function
#time step is fft scale, 2 is only positive freqs scale, Istep is impulse scaling
scale_factor = time_step / I_STEP 

#clip to prevent taking log of zero 
#only dealing with positive frequencies
fft_impulse_normed = np.clip( np.abs(fft_impulse[0:N//2])*scale_factor, 1E-6, 1E6)

min_db = -40.0
max_db = 20.0

mag_db = np.clip(20*np.log10(fft_impulse_normed), min_db, max_db)

positive_freqs = fft_freqs[:N//2]

freq_min = 80.0
freq_max = 20E3 

freq_l_index = np.argmin(np.abs(positive_freqs - freq_min))
freq_h_index = np.argmin(np.abs(positive_freqs - freq_max))

plt.plot(positive_freqs[freq_l_index:freq_h_index], mag_db[freq_l_index:freq_h_index])

plt.title("Output Impedance", fontsize=22)
plt.xscale("log")
plt.grid(True)
plt.ylabel("dB", rotation=0, fontsize=18, labelpad=12)
plt.xlabel(r"Frequency (Hz)", fontsize=18)
plt.tight_layout()
plt.margins(x=0)
plt.savefig("output_impedance.png", dpi=300)
plt.show()



print("Low frequency points")
print(positive_freqs[0:10])


