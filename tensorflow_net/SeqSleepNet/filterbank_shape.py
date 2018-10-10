import numpy as np
import os

class FilterbankShape(object):

    def tri_filter_shape(self, ndim, nfilter):
        f = np.arange(ndim)
        f_high = f[-1]
        f_low = f[0]
        H = np.zeros((nfilter, ndim))

        M = f_low + np.arange(nfilter+2)*(f_high-f_low)/(nfilter+1)
        for m in range(nfilter):
            k = np.logical_and(f >= M[m], f <= M[m+1])   # up-slope
            H[m][k] = 2*(f[k]-M[m]) / ((M[m+2]-M[m])*(M[m+1]-M[m]))
            k = np.logical_and(f >= M[m+1], f <= M[m+2]) # down-slope
            H[m][k] = 2*(M[m+2] - f[k]) / ((M[m+2]-M[m])*(M[m+2]-M[m+1]))

        H = np.transpose(H)
        H.astype(np.float32)
        return H

    def hz2mel(self, hz):
        """Convert a value in Hertz to Mels
        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * np.log10(1+hz/700.)

    def mel2hz(self, mel):
        """Convert a value in Mels to Hertz
        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700*(10**(mel/2595.0)-1)

    def mel_tri_filter_shape(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = self.hz2mel(lowfreq)
        highmel = self.hz2mel(highfreq)
        melpoints = np.linspace(lowmel,highmel,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*self.mel2hz(melpoints)/samplerate)

        fbank = np.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        fbank = np.transpose(fbank)
        fbank.astype(np.float32)
        return fbank

    def lin_tri_filter_shape(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a linear-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        #lowmel = self.hz2mel(lowfreq)
        #highmel = self.hz2mel(highfreq)
        #melpoints = np.linspace(lowmel,highmel,nfilt+2)
        hzpoints = np.linspace(lowfreq,highfreq,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*hzpoints/samplerate)

        fbank = np.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        fbank = np.transpose(fbank)
        fbank.astype(np.float32)
        return fbank

