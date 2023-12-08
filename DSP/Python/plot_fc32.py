import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
from pathlib import Path
import os

# Plot an RTSA phosphor spectrum
def fft_intensity_plot(samples: np.ndarray, fft_len: int = 256, fft_div: int = 2, mag_steps: int = 100, cmap: str = 'viridis'):
    
    num_ffts = math.floor(len(samples)/fft_len)
    
    fft_array = []
    for i in range(num_ffts):
        temp = np.fft.fftshift(np.fft.fft(samples[i*fft_len:(i+1)*fft_len]))
        temp_mag = 20.0 * np.log10(np.abs(temp))
        fft_array.append(temp_mag)
        
    max_mag = np.amax(fft_array)
    min_mag = np.abs(np.amin(fft_array))
    
    norm_fft_array = fft_array
    for i in range(num_ffts):
        norm_fft_array[i] = (fft_array[i]+(min_mag))/(max_mag+(min_mag)) 
        
    mag_step = 1/mag_steps

    hitmap_array = np.random.random((mag_steps+1,int(fft_len/fft_div)))*np.exp(-10)

    for i in range(num_ffts):
        for m in range(fft_len):
            hit_mag = int(norm_fft_array[i][m]/mag_step)
            hitmap_array[hit_mag][int(m/fft_div)] = hitmap_array[hit_mag][int(m/fft_div)] + 1

    hitmap_array_db = 20.0 * np.log10(hitmap_array+1)
    
    figure, axes = plt.subplots(figsize=(8,6))
    axes.imshow(hitmap_array_db, origin='lower', cmap=cmap, interpolation='bilinear')
    axes.set(xlabel='Frequency',
             ylabel='PSD (dB)')
    return(figure)

def plot_waterfall(samples: np.ndarray, fs: int, f_center: float, fft_len: int = 256, cmap: str = 'viridis'):
    iq = samples

    num_rows = len(iq) // fft_len # // is an integer division which rounds down
    spectrogram = np.zeros((num_rows, fft_len))
    for i in range(num_rows):
        spectrogram[i,:] = np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq[i*fft_len:(i+1)*fft_len])))**2)

    figure, axes = plt.subplots()
    figure.set_size_inches(12,12)

    caxes = axes.imshow((spectrogram),
                        aspect='auto',
                        extent = [(fs/-2+f_center)/1e6, (fs/2+f_center)/1e6, len(iq)/fs, 0])

    cbar = figure.colorbar(caxes)
    cbar.set_label('Intensity dB')

    axes.set(xlabel='Frequency (MHz)',
            ylabel='Time (s)')
    
    return figure


parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

if not Path(args.file).is_file():
    print(f'{args.file} is not a file')
    exit()

filename = os.path.basename(args.file)
dirname = os.path.dirname(os.path.abspath(args.file))

print(f'Plotting the spectrum and waterfall of {filename}')
print(f'Saving plots to {dirname}')

nfft = 4096
iq_fc32 = iq_fc32 = np.fromfile(args.file, 'complex64')

fig = fft_intensity_plot(iq_fc32, nfft, 4, 200, 'viridis')
fig.set_size_inches(12,6)
plt.title(filename)
plt.savefig(f'{dirname}/{filename}-spectrum.png')
#plt.show()

fs = 28e6
f_center = 0
plot_waterfall(iq_fc32, fs, f_center, nfft)
plt.title(filename)
plt.savefig(f'{dirname}/{filename}-waterfall.png')
#plt.show()