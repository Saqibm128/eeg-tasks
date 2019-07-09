COMMON_FREQ = 125
COMMON_DELTA = 1.0/COMMON_FREQ
#gamma, theta, alpha, beta bands
FREQ_BANDS=[0,3.5,7.5,14,40]
SMALLEST_COLUMN_SUBSET=['EEG F4-REF','EEG C4-REF','EEG O2-REF',] #as per https://www.ers-education.org/lrmedia/2016/pdf/298830.pdf
SYMMETRIC_COLUMN_SUBSET=['EEG F4-REF','EEG C4-REF','EEG O2-REF','EEG F3-REF','EEG C3-REF','EEG O1-REF',] #SMALLEST_COLUMN_SUBSET only hits a single hemisphere, this hits two
