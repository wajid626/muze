import librosa
import librosa.display
import matplotlib.pyplot as plt
y, sr = librosa.load('./U-belong-with-me.mp3')
plt.plot(y);
plt.title('Signal');
plt.xlabel('Time (samples)');
plt.ylabel('Amplitude');
plt.savefig('mp32singnal.png')