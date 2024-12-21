import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
y, sr = librosa.load('./U-belong-with-me.mp3')

n_fft = 2048
ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))
plt.plot(ft);
plt.title('Spectrum');
plt.xlabel('Frequency Bin');
plt.ylabel('Amplitude');
plt.savefig('fft.png');