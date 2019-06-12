# dbmi_eeg_clustering
Uses the TUH dataset:  https://www.isip.piconepress.com/projects/tuh_eeg/

This dataset is segmented into a train set and test set and includes annotations.
In addition, there are subsets of the data based on the exact reference node of the EEG.

# Setup
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
