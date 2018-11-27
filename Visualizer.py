import librosa
import librosa.display
import matplotlib.pyplot as plt

audioInput = ["clean_p232_006.wav"]

audio, sampling_rate = librosa.load(audioInput[0])
plt.figure(figsize=(12, 4))
plt.title("Clean data")
plt.xlabel("t")
plt.ylabel("y")
librosa.display.waveplot(audio, sr=sampling_rate)
plt.show()