import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
y, sr = librosa.load('./U-belong-with-me.mp3')
spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB');
plt.title('Spectrogram');
plt.savefig('spectogram.png');