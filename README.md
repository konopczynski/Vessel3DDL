# README #

Vessel3DDL - Automated Multiscale 3D Feature Learning for Vessels Segmentation in Thorax CT Images

### Structure ###

The data may be downloaded from: https://grand-challenge.org/site/vessel12/
and should be stored in the Data folder. Should look like this:

└─ Data
   ├── Serialized
   │   ├── Output
   │   ├── saved_classifiers
   │   ├── saved_dict
   │   ├── saved_measures
   │   ├── saved_patches
   │   └── saved_xy
   │       └── Parallel
   ├── VESSEL12_01-05
   ├── VESSEL12_01-20_Lungmasks
   ├── VESSEL12_06-10
   ├── VESSEL12_11-15
   ├── VESSEL12_16-20
   └── VESSEL12_ExampleScans
       ├── Annotations
       ├── Lungmasks
       └── Scans

### Structure ###

* Dictionary learning (Unsupervised step). First the dictionary has to be learned on a number of given volumes. The volumes don't have to be annotated. 
* Classifier learning (Supervised step). Based on the learned features, train the classifier of choice.
* Testing module. Apply filters from the dictionary and use a classifier.
* Some additional functionality: 3d patch extraction, 3d Gaussian pyramids, loading/saving data.

└── scripts
    ├── config.py
    ├── HelpFunctions.py
    ├── LearnClassifier
    │   ├── ConcatenateXy.py
    │   ├── ExtractXy_multithread.py
    │   ├── MakeMeasurments.py
    │   ├── TrainClassifier.py
    │   └── VolumesToXy.py
    ├── LearnDictionary
    │   ├── ExtractPatches.py
    │   └── LearnDictionary.py
    ├── patches_3d.py
    ├── pyramids_3d.py
    └── UseClassifier
        ├── UseClassifier.py
        └── ViewResults.py

The dictionaries and classifier weights are serialized.
