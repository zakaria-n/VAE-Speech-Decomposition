import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
import soundfile as sf

import librosa 
import parselmouth 
from parselmouth import praat


def plot_specgram_from_wave(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)


def plot_specgram(spec, sample_rate, title="Spectrogram", xlim=None):
  num_freq, num_frames = spec.shape
  time_axis = np.arange(0, num_frames) / num_frames
  freq_axis = np.arange(0, num_freq) * sample_rate/2/num_freq
  figure, axes = plt.subplots(1, 1)
  axes.pcolormesh(time_axis, freq_axis, spec[:,:], cmap='viridis')
  axes.set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")




## Features extraction 

def specToAudio(x):
    return librosa.feature.inverse.mel_to_audio(x) # to test 

def f0(x, gender): 
    '''
    Measure voice pitch
    '''
    if gender == 'male':
        f0min = 50
        f0max = 200 
        
    else: 
        f0min = 150
        f0max = 350
    
    audio = specToAudio(x)
    sound = parselmouth.Sound(audio) # read the sound
    pitch = praat.call(sound, "To Pitch", 0.0, f0min, f0max) # create a praat pitch object
    pitch_values = pitch.selected_array['frequency']

    return pitch_values 

def f0_array(x_train, gender):
    f0_list = [] # list of list, since pitch_values is a list
    for i in range(x_train.shape[0]):
        pitch_values = f0(x_train[i], gender)
        f0_list.append(pitch_values)
    
    return np.array(f0_list)



def extract_formants(x, gender):
    '''
    Extract 3 first formants with Praat's functions thanks to Python's parselmouth
    '''
    time_step = 0.0025 # time between the centres of consecutive analysis frames 
    max_nb_formants = 5 # number of formants extracted per frame
    # formant_ceiling: maximum frequency of the formant search range, in Hertz
    window_length = 0.025 # duration of the analysis window (s)
    pre_emphasis = 50
    
    if gender == 'male':
        f0min = 100
        f0max = 150
        formant_ceiling = 5000 # An average adult male speaker has a vocal tract length that requires an average ceiling of 5000 Hz
        
    else: # gender == 'female'
        f0min = 200
        f0max = 300
        formant_ceiling = 5500 # An average adult female speaker has a vocal tract length that requires an average ceiling of 5500 Hz 
    
    audio = specToAudio(x) 

    # Transform the file into a parselmouth object sound
    sound = parselmouth.Sound(audio) 

    # First, compute the occurrences of periodic instances in the signal:
    pointProcess = praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    # Then, compute the formants: 
    formants = praat.call(sound, "To Formant (burg)", time_step, max_nb_formants, formant_ceiling, window_length, pre_emphasis) 

    # And finally assign formant values with times where they make sense (periodic instances):
    numPoints = praat.call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    for point in range(0, numPoints):
        point += 1
        t = praat.call(pointProcess, "Get time from index", point)
        f1 = praat.call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = praat.call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = praat.call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)

    # Create a matrix of formants 
    formant_tuple = (np.array(f1_list), np.array(f2_list), np.array(f3_list))
    formants = np.vstack(formant_tuple)
    formants = formants[~np.isnan(formants)] # We should check that nan values appear in the same places 
    
    return formants


def formants_array(x_train, gender):
    formants_list = [] # list of list, since pitch_values is a list
    for i in range(x_train.shape[0]):
        formants_list.append(extract_formants(x_train[i], gender))
    
    return np.array(formants_list)

    