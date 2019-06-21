# dbmi_eeg_clustering
Uses the TUH dataset:  https://www.isip.piconepress.com/projects/tuh_eeg/

This dataset is segmented into a train set and test set and includes annotations.
In addition, there are subsets of the data based on the exact reference node of the EEG.

## Sacred
Uses sacred to control experiments and provides logging
https://sacred.readthedocs.io/en/latest/

## Project Setup
Notebooks are messy and are currently used for scratch code.

[environmentSetup.sh](environmentSetup.sh) can be used to create a conda environment that can run this code

[data_reader.py](data_reader.py) is used to read data from edf files.
The EdfDataset returns raw data in the form of Pd.DataFrame
The EdfFFTDatasetTransformer returns the same data transformed.

[env.yaml](env.yaml)
the conda environment i'm using

[script_runner.py](script_runner.py)
Used to run the initial_clustering.py file with multiple parameters

[util_funcs.py](util_funcs.py)
Contains a few utility functions, including a MongoClient provider for access to
a db to store sacred experiment results.

## Data Format
data should be accessible using the EdfDataset and EdfFFTDatasetTransformer
EdfDataset and EdfFFTDatasetTransformer is array-like, and will return tuples.
The first elem is the actual data, second is a timeseries by annotation array
showing the assigned probabilities for various annotations

## Setup
Reads the directory of the data using config.json file

``` config.json
{
  "train_01_tcp_ar": "/mnt/c/Users/sawer/src/dbmi/tuh/v1.5.0/edf/train/01_tcp_ar/",
  "dev_test_01_tcp_ar": "/mnt/c/Users/sawer/src/dbmi/tuh/v1.5.0/edf/dev_test/01_tcp_ar/",
  "train_02_tcp_le": "/mnt/c/Users/sawer/src/dbmi/tuh/v1.5.0/edf/train/02_tcp_le/",
  "dev_test_02_tcp_le": "/mnt/c/Users/sawer/src/dbmi/tuh/v1.5.0/edf/dev_test/02_tcp_le/",
  "train_03_tcp_ar_a": "/mnt/c/Users/sawer/src/dbmi/tuh/v1.5.0/edf/dev_test/03_tcp_ar_a/",
  "dev_test_03_tcp_ar_a": "/mnt/c/Users/sawer/src/dbmi/tuh/v1.5.0/edf/train/03_tcp_ar_a/"
}
```
