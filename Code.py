import os
import math
import numpy
import librosa
import librosa.display
from sklearn.linear_model import LassoLars
import matplotlib.pyplot as plt


dataSet = "Data/"
resultSet = "Result/"

#sampling rate for our dataset is 22050
sampling_rate_target = 5000
dictionary_size = 50
atoms = dictionary_size * 10

#make it so the average norm of the difference in atoms is less than 0.0001
dictThreshold = atoms * 0.0001
maxDictIterations = 1000


A = numpy.zeros((atoms, atoms),  numpy.float32)
B = numpy.zeros((dictionary_size, atoms),  numpy.float32)
D = numpy.random.randn(dictionary_size, atoms)
D = D.astype(numpy.float32)

#enforce condition that all columns have less than unit norm
D = D / numpy.linalg.norm(D, 2, 0, True)

#LARS alpha initialization
reg = LassoLars(alpha=0.0005, fit_intercept=False, copy_X=True)

count = 0
#bool to track when the A matrix has no zeros on the diagonal
aNonNull = False
for filename in os.listdir(dataSet):
    if filename.endswith(".wav"): 
        print("Sound: " + str(count))
        count += 1
        audio, sampling_rate = librosa.load(dataSet + filename, sampling_rate_target)
        audioOutput = numpy.array([], numpy.float32)
        for i in range(math.ceil(audio.shape[0] / dictionary_size)):
            #extract an xt for each chunk of audio, padding shorter audio to reach a second
            xt = numpy.copy(audio[(i*dictionary_size):((i+1)*dictionary_size)])
            xt = numpy.pad(xt, pad_width=(0, dictionary_size - xt.shape[0]), mode='constant', constant_values=0)

            #LARS alpha computation
            reg.fit(D, xt)
            alpha = reg.coef_
            alpha = alpha.astype(numpy.float32)

            #reshape parameters
            xt = xt.reshape((xt.shape[0], 1))
            alpha = alpha.reshape((alpha.shape[0], 1))

            print("Finished Alpha Update")
            A += numpy.matmul(alpha, numpy.transpose(alpha))
            B += numpy.matmul(xt, numpy.transpose(alpha))

            aNonNull = True
            for i in range(A.shape[0]):
                if (A[i,i] == 0):
                    aNonNull = False
                
                

            #add the second to the audio output
            audioOutput = numpy.concatenate([audioOutput, numpy.matmul(D, alpha).reshape(-1)])
            
            if (aNonNull):
                #Dictionary Update
                dictCounter = 0
                while True:
                    print("Finished " + str(dictCounter) + "/" + str(maxDictIterations) + " Dictionary Iterations", end="\r")
                    dictCounter += 1
                    totalDiff = 0
                    for j in range(D.shape[1]):
                        uj = 1/A[j, j] * (B[:, j:j+1] - numpy.matmul(D, A[:, j:j+1])) + D[:, j:j+1]
                        dUpdate = 1/ max(numpy.linalg.norm(uj), 1) * uj
                        totalDiff += numpy.linalg.norm(dUpdate - D[:, j:j+1])
                        D[:, j:j+1] = dUpdate
                    if(totalDiff < dictThreshold or dictCounter > maxDictIterations):
                        break
                print("Finished Dictionary Update")
        librosa.output.write_wav(resultSet + 'Output' + filename, audioOutput, sampling_rate)
        print("Audio file processed")
            
#plt.figure(figsize=(12, 4))
#librosa.display.waveplot(data, sr=sampling_rate)
#plt.show()