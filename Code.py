import librosa
import librosa.display
import matplotlib.pyplot as plt

data, sampling_rate = librosa.load('Test.wav')
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)
plt.show()