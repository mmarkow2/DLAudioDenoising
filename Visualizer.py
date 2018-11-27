import librosa
import librosa.display
import matplotlib.pyplot as plt

audioInput = ["DLAudioDenoising/Result/Build 1 - 0.0001 lambda/Outputp232_006.wav", "DLAudioDenoising/Result/Build 2 - 0.0005 lambda/Outputp232_006.wav", "DLAudioDenoising/Result/Build 4 - 0.0007 lambda/Outputp232_006.wav", "DLAudioDenoising/Result/Build 3 - 0.001 lambda/Outputp232_006.wav"]

audio, sampling_rate = librosa.load(audioInput[3])
plt.figure(figsize=(12, 4))
plt.title("Plot For lambda = 0.001")
plt.xlabel("t")
plt.ylabel("y")
librosa.display.waveplot(audio, sr=sampling_rate)
plt.show()