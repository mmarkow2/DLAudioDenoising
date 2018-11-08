import os
import math
import numpy
import librosa
import librosa.display
import matplotlib.pyplot as plt

def normGrad(alpha):
    result = numpy.copy(alpha)
    result[result<0] = -1
    result[result>=0] = 1
    return result

def LARSGradient(D, xt, alpha, lambdaParm):
    return numpy.matmul(-numpy.transpose(D), xt) + numpy.matmul(numpy.matmul(numpy.transpose(D), D), alpha) + lambdaParm * normGrad(alpha)


trainingSet = "Data/"
alphaIterations = 5000
dictIterations = 500

#sampling rate for our dataset is 22050
dataset_sampling_rate = 22050
atoms = dataset_sampling_rate * 2
A = numpy.zeros((atoms, atoms))
B = numpy.zeros((dataset_sampling_rate, atoms))
D = numpy.zeros((dataset_sampling_rate, atoms))

lambdaParm = 1

for filename in os.listdir(trainingSet):
    if filename.endswith(".wav"): 
        audio, sampling_rate = librosa.load(trainingSet + filename)
        for i in range(math.ceil(audio.shape[0] / sampling_rate)):
            #extract an xt for each second of audio, padding shorter audio to reach a second
            xt = audio[(i*sampling_rate):((i+1)*sampling_rate)]
            xt = numpy.pad(xt, pad_width=(0,sampling_rate - xt.shape[0]), mode='constant', constant_values=0)
            
            #convert to column vector
            xt = numpy.transpose(xt)
            
            #LARS alpha computation
            alpha = numpy.zeros((atoms, 1))
            for t in range(1, alphaIterations + 1):
                #theoretical learning_rate = 2/(alpha*t)
                learning_rate = 2 / (min(numpy.linalg.eigvals(numpy.matmul(numpy.transpose(D), D))) * t)
                alpha = alpha - learning_rate * LARSGradient(D, xt, alpha, lambdaParm)
            
            A += numpy.matmul(alpha, numpy.transpose(alpha))
            B += numpy.matmul(xt, numpy.transpose(alpha))
            
            #Dictionary Update
            for t in range(1, dictIterations + 1):
                for j in range(0, D.shape[1]):
                    uj = 1/A[j, j] * (B[:, j] - numpy.matmul(D, A[:, j])) + D[:, j]
                    D[:, j] = 1/ max(numpy.linalg.norm(uj), 1) * uj
            
#plt.figure(figsize=(12, 4))
#librosa.display.waveplot(data, sr=sampling_rate)
#librosa.output.write_wav('Test2.wav', data, sampling_rate)
#plt.show()