'''
Used to create a new combined split that we use to create test, train splits, since dev_test split is imbalanced (75% males!).
'''
import os, sys
import util_funcs
import data_reader as read
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Used to create a mega directory")
    parser.add_argument("new_split_directory", help="which new directory to use as a split inside the data_dir_root (from config.json)")
    parser.add_argument("--ref", help="which reference system to use", default="01_tcp_ar")
    parser.add_argument("--apply", help="apply changes to directory. otherwise, does dry run", action="store_true")
    parser.add_argument("--dont_use_symlinks", help="instead do a copy", action="store_true")

    args = parser.parse_args()
    use_symlinks = not args.dont_use_symlinks
    if not use_symlinks:
        raise Exception("Not implemented")

    filesDict = {}
    trainPatientsFullPath = read.get_patient_dir_names("train", args.ref)
    trainPatients = read.get_patient_dir_names("train", args.ref, full_path=False)
    testPatientsFullPath = read.get_patient_dir_names("dev_test", args.ref)
    testPatients = read.get_patient_dir_names("dev_test", args.ref, full_path=False)

    allPatientsFullPath = trainPatientsFullPath + testPatientsFullPath
    allPatients = set(testPatients).union(set(trainPatients))
    if len(allPatients) != len(allPatientsFullPath):
        print("WARNING! Original test set and train set had patients in common")

    config = util_funcs.read_config()

    for patient in allPatients:
        filesDict[patient] = {}

        newPatientDir = os.path.join(config["data_dir_root"],  args.new_split_directory, args.ref,patient[0:3], patient)
        print("Making directory: ", newPatientDir)
        if args.apply:
            os.makedirs(newPatientDir)

    for patientDir in allPatientsFullPath:
        patient = os.path.basename(patientDir)
        sessionPaths = os.listdir(patientDir)
        newPatientDir = os.path.join(config["data_dir_root"],  args.new_split_directory, args.ref, patient[0:3], patient)

        for sessionPath in sessionPaths:
            sessionPath = os.path.join(patientDir, sessionPath)
            session = os.path.basename(sessionPath)
            if session in filesDict[patient].keys():
                print("WARNING! Repeated session!") # not only repeated patients but repeated sessions in between dev_test and train
            else:
                filesDict[patient][session] = sessionPath
                newSessionPath = os.path.join(newPatientDir, session)
                # print("Making new directory: {}".format(newSessionPath))
                print("Making symlink from {} to new directory {}".format(sessionPath, newSessionPath))

                if args.apply:
                    os.symlink(sessionPath, newSessionPath)






    print("Finished")
