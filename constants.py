COMMON_FREQ = 250
COMMON_DELTA = 1.0/COMMON_FREQ
#gamma, theta, alpha, beta bands
FREQ_BANDS=[0,3.5,7.5,14,40]
SMALLEST_COLUMN_SUBSET=['EEG F4-REF','EEG C4-REF','EEG O2-REF',] #as per https://www.ers-education.org/lrmedia/2016/pdf/298830.pdf
SYMMETRIC_COLUMN_SUBSET=['EEG F4-REF','EEG C4-REF','EEG O2-REF','EEG F3-REF','EEG C3-REF','EEG O1-REF',] #SMALLEST_COLUMN_SUBSET only hits a single hemisphere, this hits two


SIMPLE_CONV2D_MAP = [
    [0           , 0           ,"EEG FP1-REF", 0           ,"EEG FP2-REF",  0           , 0           ],
    [0           , "EEG F7-REF", "EEG F3-REF", "EEG FZ-REF", "EEG F4-REF", "EEG F8-REF", 0           ],
    ["EEG A1-REF", "EEG T3-REF", "EEG C3-REF", "EEG CZ-REF", "EEG C4-REF", "EEG T4-REF", "EEG A2-REF"],
    [0           , "EEG T5-REF", "EEG P3-REF", "EEG PZ-REF", "EEG P4-REF", "EEG T6-REF", 0           ],
    [0           ,         0   , "EEG O1-REF", 0           , "EEG O2-REF", 0           , 0           ],
]

SEIZURE_SUBTYPES = ["bckg", "fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz", "cnsz", "tcsz", "atsz", "mysz"]

MONTAGE_COLUMNS = ['C3-CZ','CZ-C4','FP1-F7','F8-T4','F7-T3','C4-T4','FP2-F4','T5-O1','C4-P4','F3-C3','T3-T5','A1-T3','T4-A2','FP2-F8','FP1-F3','T3-C3','C3-P3','T4-T6','P4-O2','P3-O1','T6-O2','F4-C4']
