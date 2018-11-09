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

def LARSGradient(negDXt, Hessian, xt, alpha, lambdaParm):
    return  negDXt + numpy.matmul(Hessian, alpha) + lambdaParm * normGrad(alpha)


trainingSet = "trainingData/"
testingSet = "testingData/"
alphaIterations = 1000
dictIterations = 50

#sampling rate for our dataset is 22050
sampling_rate_target = 5000
dictionary_size = 200
atoms = dictionary_size * 5
A = numpy.zeros((atoms, atoms),  numpy.float32)
B = numpy.zeros((dictionary_size, atoms),  numpy.float32)
D = numpy.random.randn(dictionary_size, atoms)
D = D.astype(numpy.float32)

lambdaParm = 1
for dataLocation in [trainingSet, testingSet]:
    count = 0
    for filename in os.listdir(dataLocation):
        if filename.endswith(".wav"): 
            print("Sound: " + str(count))
            count += 1
            audio, sampling_rate = librosa.load(dataLocation + filename, sampling_rate_target)
            if (dataLocation == testingSet):
                audioOutput = numpy.array([], numpy.float32)
            for i in range(math.ceil(audio.shape[0] / dictionary_size)):
                #extract an xt for each second of audio, padding shorter audio to reach a second
                xt = numpy.copy(audio[(i*dictionary_size):((i+1)*dictionary_size)])
                xt = numpy.pad(xt, pad_width=(0, dictionary_size - xt.shape[0]), mode='constant', constant_values=0)
                xt = xt.reshape((xt.shape[0], 1))

                #LARS alpha computation
                alpha = numpy.zeros((atoms, 1), numpy.float32)

                #precompute values from the unchanging dictionary for efficiency
                hessian = numpy.matmul(numpy.transpose(D), D)
                strConvexAlpha = min(numpy.linalg.eigvalsh(hessian))
                #-D * xt is used in the gradient but is precomputed since it doesn't depend on alpha
                negDXt = numpy.matmul(-numpy.transpose(D), xt)

                for t in range(1, alphaIterations + 1):
                    #theoretical learning_rate
                    if (strConvexAlpha > 0 ):
                        #when strongly-convex, use 2/(alpha*t)
                        learning_rate = 2 / (strConvexAlpha * t)
                    else:
                        #when not strongly-convex, use 1/(sqrt(t) * norm(subradient))
                        learning_rate = 1 / (math.sqrt(t) * numpy.linalg.norm(LARSGradient(negDXt, hessian, xt, alpha, lambdaParm)))
                    alpha = alpha - learning_rate * LARSGradient(negDXt, hessian, xt, alpha, lambdaParm)

                print("Finished alpha update")
                if (dataLocation == trainingSet):
                    A += numpy.matmul(alpha, numpy.transpose(alpha))
                    B += numpy.matmul(xt, numpy.transpose(alpha))

                    #Dictionary Update
                    for t in range(dictIterations):
                        for j in range(D.shape[1]):
                            uj = 1/A[j, j] * (B[:, j] - numpy.matmul(D, A[:, j])) + D[:, j]
                            D[:, j] = 1/ max(numpy.linalg.norm(uj), 1) * uj
                else:
                    #add the second to the audio output
                    audioOutput = numpy.concatenate([audioOutput, numpy.matmul(D, alpha).reshape(-1)])
            if (dataLocation == trainingSet):
                print("Training audio file processed")
            else:
                #output the representation
                librosa.output.write_wav('Output' + filename, audioOutput, sampling_rate)
                print("Testing audio file processed")
            
#plt.figure(figsize=(12, 4))
#librosa.display.waveplot(data, sr=sampling_rate)
#plt.show()