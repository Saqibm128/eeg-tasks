import re
from data_reader import convert_edf_path_to_txt, get_all_clinical_notes, get_all_token_file_names

def getGenderAndFileNames(split, ref):
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
            elif re.search(r'man', txt) is not None:
                gender = 'm'
            elif re.search(r'male', txt) is not None:
                gender = 'm'
            if gender is not None:
                genders[clinical_fn] = gender
        except:
            print("Could not read {}".format(token_fn))
    toDels = []
    for key, val in genders.items():
        if val is None:
            toDels.append(key)
    for toDel in toDels:
        del genders[toDel]
    return list(genders.items())

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
