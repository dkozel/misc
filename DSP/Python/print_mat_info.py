import sys
from scipy.io import loadmat                                                                                                                                                                                                     

file = sys.argv[1]

# Regular Matlab file
annots = loadmat(file)
print(annots)
exit()

# Older Matlab files
import h5py

f = h5py.File(file)
list(f.keys())
for k in f.keys():
   print(len(f[k]))