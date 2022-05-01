# Dataset Setup

Download the dataset zip file from kaggle here: https://www.kaggle.com/datasets/ruizgara/socofing

Extract the contents of the zip into a folder called `SOCOFing` and copy the `labels.json` into it.

The final file structure should look as such,

```
data
├───...
└───SOCOFing
    ├───labels.json
    ├───Altered
    │   ├───Altered-Easy
    │   ├───Altered-Hard
    │   └───Altered-Medium
    └───Real
```

Then, `cd` into the data directory and run the `gen_data_csv.py` file.