#!/usr/bin/python3

import numpy as np

# Set the sampling rate and duration
fs = 2*61.44e6 # Sampling rate (Hz)
duration = 0.1 # Duration (s)

# Set the frequency of the sine wave
f = 5e6 # Frequency (Hz)

# Generate the time vector
t = np.arange(0, duration, 1/fs)

# Generate the complex sine wave
sine_wave = np.exp(1j * 2 * np.pi * f * t)

# Frequency shift the entire spectrum to cancel the LO offset in the FPGA
f_shifted = 10e6 # Frequency shift (Hz)
carrier_wave = np.exp(1j * 2 * np.pi * f_shifted * t)
sine_wave_shifted = sine_wave * carrier_wave

# Convert the complex values to signed 16-bit integers
iq_s16 = (sine_wave_shifted * (2**15 - 1)).astype(np.int16)

# Write the sine wave to a binary file
with open('sine_wave.sc16', 'wb') as f:
    f.write(sine_wave_int.tobytes())