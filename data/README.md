Once the files were requested and downloaded, their image captions were generated with BLIP-2 vision-language model. They were then split with stratified sampling into trainng, dev and test. The final splits are stored in this directory, with the following structure:

```
data
├── EXIST2024
│   ├── EXIST2024_test.json
│   ├── EXIST2024_training.json
│   └── EXIST2024_validation.json
├── MAMI
│   ├── MAMI_test.json
│   ├── MAMI_training.json
│   └── MAMI_validation.json
└── overlapping_classes
    ├── EXIST2024
    │   ├── EXIST2024_test.json
    │   ├── EXIST2024_training.json
    │   └── EXIST2024_validation.json
    └── MAMI
        ├── MAMI_test.json
        ├── MAMI_training.json
        └── MAMI_validation.json
```

The EXIST 2024 splits were created from the original training split as the gold labels in the test set are not publicly available.