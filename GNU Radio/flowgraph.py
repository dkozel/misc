import numpy as np
from gnuradio import gr
from time import sleep

# Define modulation parameters
N = 1_000  # Number of symbols
M = 4  # Number of constellation points (QAM-4)
bits_per_symbol = int(np.log2(M))
bit_combinations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.bool_)
constellation_points = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
bit_to_symbol_map = {tuple(b): p for b, p in zip(bit_combinations, constellation_points)}

class BitSource(gr.sync_block):
    """Generates a stream of random bits."""
    def __init__(self):
        gr.sync_block.__init__(self, name='BitSource', in_sig=None, out_sig=[np.int8])
    
    def work(self, input_items, output_items):
        output_items[0][:] = np.random.randint(0, 2, len(output_items[0]), dtype=np.int8)
        return len(output_items[0])

class Modulator(gr.decim_block):
    """Maps bits to QAM symbols."""
    def __init__(self, constellation=bit_to_symbol_map, bits_per_symbol=bits_per_symbol):
        gr.decim_block.__init__(self, name='Modulator', in_sig=[np.int8], out_sig=[np.complex64], decim=bits_per_symbol)
        self.constellation = constellation
        self.bits_per_symbol = bits_per_symbol
    
    def work(self, input_items, output_items):
        in0 = input_items[0][:len(input_items[0])//self.bits_per_symbol*self.bits_per_symbol]
        in0 = np.reshape(in0, (-1, self.bits_per_symbol))
        output_items[0][:] = np.array([self.constellation[tuple(b)] for b in in0]).flatten()
        return len(in0)

class AwgnChannel(gr.sync_block):
    """Adds AWGN to the signal based on SNR."""
    def __init__(self, snr):
        gr.sync_block.__init__(self, name='AWGN Channel', in_sig=[np.complex64], out_sig=[np.complex64])
        self.snr = snr
    
    def work(self, input_items, output_items):
        noise = (np.random.randn(len(input_items[0])) + 1j*np.random.randn(len(input_items[0]))) / (self.snr * np.sqrt(2))
        output_items[0][:] = input_items[0] + noise
        return len(input_items[0])

class Demodulator(gr.interp_block):
    """Maps received symbols back to bit sequences."""
    def __init__(self, constellation=bit_to_symbol_map, bits_per_symbol=bits_per_symbol):
        gr.interp_block.__init__(self, name='Demodulator', in_sig=[np.complex64], out_sig=[np.int8], interp=bits_per_symbol)
        self.constellation = constellation
        self.bits_per_symbol = bits_per_symbol
        self.constellation_points = np.array(list(constellation.values())).reshape((1, -1))
        self.bit_combinations = np.array(list(constellation.keys())).reshape((-1, bits_per_symbol))
    
    def work(self, input_items, output_items):
        distances = np.abs(input_items[0][:, None] - self.constellation_points)
        output_items[0][:] = self.bit_combinations[np.argmin(distances, axis=1)].flatten()
        return len(output_items[0])

class BitErrorCounter(gr.sync_block):
    """Counts bit errors for BER computation."""
    def __init__(self):
        gr.sync_block.__init__(self, name='Bit Error Counter', in_sig=[np.int8, np.int8], out_sig=None)
        self.total, self.errors = 0, 0
    
    def work(self, input_items, output_items):
        length = min(len(input_items[0]), len(input_items[1]))
        self.total += length
        self.errors += np.sum(np.logical_xor(input_items[0][:length], input_items[1][:length]))
        return length
    
    def get_BER(self):
        return self.errors / self.total if self.total > 0 else 0

class TopBlock(gr.top_block):
    """Defines the full GNU Radio flowgraph."""
    def __init__(self, snr):
        gr.top_block.__init__(self, f'Flowgraph SNR={snr}')
        self.snr = snr
        self.bitsource = BitSource()
        self.modulator = Modulator()
        self.channel = AwgnChannel(snr)
        self.demodulator = Demodulator()
        self.biterrors = BitErrorCounter()
        
        self.connect((self.bitsource, 0), (self.modulator, 0))
        self.connect((self.modulator, 0), (self.channel, 0))
        self.connect((self.channel, 0), (self.demodulator, 0))
        self.connect((self.demodulator, 0), (self.biterrors, 1))
        self.connect((self.bitsource, 0), (self.biterrors, 0))
    
    def print_bit_error_rate(self):
        print(f'SNR: {self.snr}, BER: {self.biterrors.get_BER()}')

if __name__ == '__main__':
    """Run the flowgraph for multiple SNR values and print BER results."""
    tb = TopBlock(snr=2)
    tb.start()
    sleep(1)  # Run for 1 second
    tb.stop()
    tb.wait()
    tb.print_bit_error_rate()
