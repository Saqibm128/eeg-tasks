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
MONTAGE_COLUMN_TUPLES = [(1, 6),(6, 10),(16, 14),(4, 12),(14, 18),(10, 12),(20, 13),(0, 11),(10, 3),(5, 1),(18, 0),(2, 18),(12, 8),(20, 4),(16, 5),(18, 1),(1, 9),(12, 19),(3, 7),(9, 11),(19, 7),(13, 10)]

MNE_CHANNEL_EDF_MAPPING = {'EEG FP1-REF': 'Fp1',
 'EEG FP2-REF': 'Fp2',
 'EEG F7-REF': 'F7',
 'EEG F3-REF': 'F3',
 'EEG FZ-REF': 'Fz',
 'EEG F4-REF': 'F4',
 'EEG F8-REF': 'F8',
 'EEG C3-REF': 'C3',
 'EEG CZ-REF': 'Cz',
 'EEG C4-REF': 'C4',
 'EEG P3-REF': 'P3',
 'EEG PZ-REF': 'Pz',
 'EEG P4-REF': 'P4',
 'EEG O1-REF': 'O1',
 'EEG O2-REF': 'O2',
 'EEG T3-REF': 'T3',
 'EEG T5-REF': 'T5',
 'EEG T4-REF': 'T4',
 'EEG T6-REF': 'T6',
 'EEG A1-REF': 'A1',
 'EEG A2-REF': 'A2'}
