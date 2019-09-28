import re
from data_reader import convert_edf_path_to_txt, get_all_clinical_notes, get_all_token_file_names, get_token_file_names
import data_reader as read
from sklearn.model_selection import train_test_split
from os import path
import argparse
import pandas as pd
from addict import Dict
from functools import lru_cache

def train_test_split_on_combined(edfTokens, labels, test_size=0.2, warn_conflicts=False, stratify=True):
    patients = Dict()
    for i, token in enumerate(edfTokens):
        data_split, patient, session, tokenName = read.parse_edf_token_path_structure(token)
        if patient in patients.keys() and patients[patient].label != labels[i] and warn_conflicts:
            print("WARNING! Patient has conflicting labels! ", patient, session, token)
        if patient not in patients.keys():
            patients[patient].tokens = []
        patients[patient].label = labels[i]
        patients[patient].tokens.append(token)
    patientlabels = []
    patientId = []
    for p in patients.keys():
        patientlabels.append(patients[p].label)
        patientId.append(p)
    if stratify:
        patientIdsTrain, patientIdsTest = train_test_split(patientId, test_size=test_size, stratify=patientlabels)
    else:
        patientIdsTrain, patientIdsTest = train_test_split(patientId, test_size=test_size)

    edfTokenTrain = []
    labelTrain = []
    edfTokenTest = []
    labelTest = []
    for trainId in patientIdsTrain:
        for edfToken in patients[trainId].tokens:
            edfTokenTrain.append(edfToken)
            labelTrain.append(patients[trainId].label)
    for testId in patientIdsTest:
        for edfToken in patients[testId].tokens:
            edfTokenTest.append(edfToken)
            labelTest.append(patients[testId].label)
    return edfTokenTrain, edfTokenTest, labelTrain, labelTest

def get_clinical_notes_by_section(path, is_edf_path=True):
    """Gets clinical notes and splits based on the subheading into a somewhat structured dictionary

    Parameters
    ----------
    path : string
        path to edf file or text file
    is_edf_path : bool
        if true, then path is to edf file, we then find text file location based on that

    Returns
    -------
    Dict
        keys are headers, values are the info from the clinical notes
    """
    pass

def demux_to_tokens(dataDictItems):
    """utility method to deal with fact that labels are on session level, but
    data is on token level (all getXandFileNames methods give out on session level)

    Parameters
    ----------
    dataDictItems : list
        list of tuples of session clinical text filenames, and extracted label value

    Returns
    -------
    array
        list of edf tokens
    array
        list of repeated labels, "demuxed" to tokens
    """
    clinicalTxtPaths = [dataDictItem[0]
                        for dataDictItem in dataDictItems]
    singLabels = [dataDictItem[1] for dataDictItem in dataDictItems]
    tokenFiles = []
    labels = []  # duplicate/demux single labels depending on number of tokens per session
    for i, txtPath in enumerate(clinicalTxtPaths):
        session_dir = path.dirname(txtPath)
        session_tkn_files = sorted(get_token_file_names(session_dir))
        tokenFiles += session_tkn_files
        labels += [singLabels[i] for tkn_file in session_tkn_files]
    return tokenFiles, labels

@lru_cache(10)
def getGenderAndFileNames(split, ref, convert_gender_to_num=False,):
    all_token_fns = get_all_token_file_names(split, ref)
    num_hits = []
    genders = {}
    for token_fn in all_token_fns:
        clinical_fn = convert_edf_path_to_txt(token_fn)
        if clinical_fn in genders:
            continue
        else:
            genders[clinical_fn] = None
        try:
            txt = get_all_clinical_notes(token_fn)
            gender = None
            match = re.search(r'female', txt)
            if match is not None:
                gender = 'f'
            elif re.search(r'woman', txt) is not None:
                gender = 'f'
            elif re.search(r'man\W', txt) is not None:
                gender = 'm'
            elif re.search(r'male\W', txt) is not None:
                gender = 'm'
            if gender is not None:
                genders[clinical_fn] = gender
        except:
            print("Could not read {}".format(token_fn))
    toDels = []
    for key, val in genders.items():
        if val is None:
            toDels.append(key)
        if convert_gender_to_num:
            genders[key] = 1 if val == 'm' else 0
    for toDel in toDels:
        del genders[toDel]
    return list(genders.items())

@lru_cache(10)
def getBPMAndFileNames(split, ref):
    all_token_fns = get_all_token_file_names(split, ref)
    num_hits = []
    bpms = {}
    for token_fn in all_token_fns:
        clinical_fn = convert_edf_path_to_txt(token_fn)
        if clinical_fn in bpms:
            continue
        else:
            bpms[clinical_fn] = None
        try:
            txt = get_all_clinical_notes(token_fn)
            match = re.search(r'(\d+)\s*b\W*p\W*m', txt)
            if match is None:
                match = re.search(r'(\d+)\s*h\W+r\W+', txt)
                if match is None:
                    match = re.search(r'heart\s*rate\s*\W*\s*(\d+)', txt)
                    if match is None:
                        num_hits.append(0)
                        # print(txt)
                        continue
            num_hits.append(len(match.groups()))
            if len(match.groups()) != 0:
                bpms[clinical_fn] = int(match.group(1))
        except BaseException:
            print("Could not read clinical txt for {}".format(token_fn))
    toDels = []
    for key, val in bpms.items():
        if val is None:
            toDels.append(key)
    for toDel in toDels:
        del bpms[toDel]
    return list(bpms.items())

@lru_cache(10)
def getAgesAndFileNames(split, ref):
    all_token_fns = get_all_token_file_names(split, ref)
    num_hits = []
    ages = {}
    for token_fn in all_token_fns:
        clinical_fn = convert_edf_path_to_txt(token_fn)
        if clinical_fn in ages:
            continue
        else:
            ages[clinical_fn] = None
        try:
            txt = get_all_clinical_notes(token_fn)
            txt = txt.lower()
            match = re.search(r'(\d+)\s*-*\s*years*\s*-*\s*old', txt)
            if match is None:
                match = re.search(r'(\d+)\s*years*\s*old', txt)
                if match is None:
                    match = re.search(r'(\d+)\s*y\.\s*o\.', txt)
                    if match is None:
                        match = re.match(r'(\d+)\s*(yr|YR)s*', txt)
                        if match is None:
                            num_hits.append(0)
    #                         print(txt)
                            continue
            num_hits.append(len(match.groups()))
            if len(match.groups()) != 0:
                ages[clinical_fn] = int(match.group(1))
        except BaseException:
            print("Could not read clinical txt for {}".format(token_fn))
    toDels = []
    for key, val in ages.items():
        if val is None: #if there was a token we couldn't get an age for.
            toDels.append(key)
    for toDel in toDels:
        del ages[toDel]
    return list(ages.items())

if __name__ == "__main__":
    splits = ["dev_test", "train"]
    file_stats = pd.DataFrame(columns=["sessionName", "patientName", "reference_system", "split", "gender", "age", "heart_rate", "data_length"])
