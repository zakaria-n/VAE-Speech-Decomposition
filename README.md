# Multi-task auto-encoder for source-filter decomposition of speech

The source-filter model for speech is based on the fact that speech is produced by a waveform from the glottis (the source) that is filtered through a time-varying transfer function of the vocal tract (the filter).

This filter is equivalent to a (element-wise) multiplication in the spectral domain between the source spectrum X(k) and the filter spectrum H(k), i.e. 

Y(k) = X(k)H(k)

which is equal to a summation in the log-spectral domain:

log(Y(k)) = log(X(k)) + log(H(k))

This means that a spectrogram, which is composed of stacked log-power spectra, can be decomposed as the sum of a source spectrogram and a filter spectrogram. 

The task is to train an auto-encoder that takes a spectrogram as input and decomposes it into a source and a filter spectrogram, that when summed should equal the input. This could for example be accomplished using multi-task learning, where the auto-encoder network, in addition to re-constructing the input spectrogram, also is trained to predict explicit parameters specific to source and filter respectively (for example pitch and formant values) from separate latent vectors.
