import numpy as np

filename = 'ascii_fc32.csv'

data = np.genfromtxt(filename, delimiter=';')
iq_fc32 = data.astype(np.float32).view(np.complex64)

with open(f'{filename}.fc32', 'wb') as f:
    f.write(iq_fc32.tobytes())
