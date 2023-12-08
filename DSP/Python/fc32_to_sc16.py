import numpy as np
import argparse
from pathlib import Path
import os
from scipy.signal import resample_poly
from fractions import Fraction

def fc32_to_sc16(iq_fc32: np.ndarray):
    # Normalize to a maximum magnitude of 1 for real and imaginary components
    iq_f32 = iq_fc32.view('float32')
    max_val = np.max(iq_f32)
    if max_val > 1:
        print(f'Input fc32 IQ data has values > 1 ({max_val}). Rescaling to a max of 1.)')
        iq_f32 /= max_val
    
    print(f'Converting from fc32 to sc16 and rescaling 1 -> 2**15 - 1')
    iq_s16 = (iq_f32 * (2**15 - 1)).astype(np.int16)
    return iq_s16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("input_rate", type=float)
    parser.add_argument("output_rate", type=float, default=122.88e6)
    parser.add_argument("shift", type=float, default=27e6)
    args = parser.parse_args()

    if not Path(args.file).is_file():
        print(f'{args.file} is not a file')
        exit()

    filename = os.path.basename(args.file)
    dirname = os.path.dirname(os.path.abspath(args.file))

    print(f'Converting {filename} to sc16 format')
    print(f'Saving data to {dirname}')

    # read at most one second of samples
    nsamps = int(args.input_rate)
    iq_fc32 = np.fromfile(filename, 'complex64', count=nsamps)

    if args.output_rate != args.input_rate:
        print(f'Resampling from {args.input_rate/1e6} MS/s to {args.output_rate/1e6} MS/s')

        f = Fraction(int(args.output_rate), int(args.input_rate)) \
            .limit_denominator(1000)
        up = f.numerator
        down = f.denominator
        print(f'Upsampling by {up} then Decimating by {down}')

        iq_resampled = resample_poly(iq_fc32, up, down)
        iq_fc32 = iq_resampled

    if args.shift != 0:
        # Frequency shift the entire spectrum to cancel the LO offset in the FPGA
        t = np.arange(0, int(len(iq_fc32)/args.output_rate), 1/args.output_rate)
        carrier_wave = np.exp(1j * 2 * np.pi * args.shift * t)
        iq_shifted = carrier_wave * iq_fc32
        iq_fc32 = iq_shifted

    iq_s16 = fc32_to_sc16(iq_fc32)

    with open(f'{filename}.sc16', 'wb') as f:
        f.write(iq_s16.tobytes())