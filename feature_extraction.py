import librosa
import numpy as np
import sklearn

def normalize(x, axis=0):
  return sklearn.preprocessing.minmax_scale(x, axis=axis)

def features(x, sr):
    result = np.array([])

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=x).T, axis=0)
    result=np.hstack((result, zcr)) 

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc)) 

    # Chroma_stft
    stft = np.abs(librosa.stft(x))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft)) 

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sr).T, axis=0)
    result = np.hstack((result, mel)) 

    # Spectral Centroid
    speCen = np.mean(librosa.feature.spectral_centroid(y=x, sr=sr)[0])
    result = np.hstack((result, speCen))
    
    # Spectral Rolloff
    specRollOff = np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr))
    result = np.hstack((result, specRollOff))

    # Fourier Tempogram
    fourierTempgram = np.mean(librosa.feature.fourier_tempogram(y=x, sr=sr))
    result = np.hstack((result, fourierTempgram))
 
    return result
