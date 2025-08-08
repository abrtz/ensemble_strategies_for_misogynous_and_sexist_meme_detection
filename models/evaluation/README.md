
The gold labels from validation and test splits after stratified sampling are stored in this directory, as well as the predictions from the models. These files are used to calculate the different metrics implemented in this thesis following the ones used in the respective shared tasks.

This directory is organized as follows:

```
.
├── golds
│   ├── EXIST2024
│   └── MAMI
├── overlapping_classes
│   ├── golds
│   │   ├── EXIST2024
│   │   └── MAMI
│   └── predictions
│       ├── EXIST2024
│       └── MAMI
└── predictions
    ├── EXIST2024
    └── MAMI
```

The files are created in the notebook `gold_labels.ipynb` under the preprocessing directory above.