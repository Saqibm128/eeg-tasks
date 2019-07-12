from scipy.signal import butter, lfilter

# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lp_filter(data, lowcut, fs, order=5):
    nyq = 0.5 * fs #just get the highest freq possible (nyquist, and bandgap it!)
    hc = nyq * 0.9 #can't accept exactly nyq
    print("hc:", hc)
    return butter_bandpass_filter(data, lowcut, hyc, fs, order)


def butter_bandgap_filter(data, lowcut, highcut, fs, order=5):
    if highcut is None:
        return butter_bandgap_filter(data, lowcut, fs, order)
    toRemove = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    return data - toRemove
