# Dataset Setup

Download the dataset zip file from kaggle here: https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

Extract the contents of the zip into a folder called `tuberculosis` and copy the `labels.json` into it.

The final file structure should look as such,

```
data
├── gen_data_csv.py
├── readme.md
└── tuberculosis
    ├── labels.json
    ├── Normal
    ├── Normal.metadata.xlsx
    ├── README.md.txt
    ├── Tuberculosis
    └── Tuberculosis.metadata.xlsx
```