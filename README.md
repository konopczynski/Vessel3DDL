## Automated Multiscale 3D Feature Learning for Vessels Segmentation in Thorax CT Images ##

### Structure ###

The data may be downloaded from: https://grand-challenge.org/site/vessel12/
and should be stored in the Data folder.
 <br />
└─ Data <br />
   ├── Serialized <br />
   │   ├── Output <br />
   │   ├── saved_classifiers <br />
   │   ├── saved_dict <br />
   │   ├── saved_measures <br />
   │   ├── saved_patches <br />
   │   └── saved_xy <br />
   │       └── Parallel <br />
   ├── VESSEL12_01-05 <br />
   ├── VESSEL12_01-20_Lungmasks <br />
   ├── VESSEL12_06-10 <br />
   ├── VESSEL12_11-15 <br />
   ├── VESSEL12_16-20 <br />
   └── VESSEL12_ExampleScans <br />
       ├── Annotations <br />
       ├── Lungmasks <br />
       └── Scans <br />

### Structure ###

* Dictionary learning (Unsupervised step). First the dictionary has to be learned on a number of given volumes. The volumes don't have to be annotated. 
* Classifier learning (Supervised step). Based on the learned features, train the classifier of choice.
* Testing module. Apply filters from the dictionary and use a classifier.
* Some additional functionality: 3d patch extraction, 3d Gaussian pyramids, loading/saving data.
 <br />
└── scripts <br />
    ├── config.py <br />
    ├── HelpFunctions.py <br />
    ├── LearnClassifier <br />
    │   ├── ConcatenateXy.py <br />
    │   ├── ExtractXy_multithread.py <br />
    │   ├── MakeMeasurments.py <br />
    │   ├── TrainClassifier.py <br />
    │   └── VolumesToXy.py <br />
    ├── LearnDictionary <br />
    │   ├── ExtractPatches.py <br />
    │   └── LearnDictionary.py <br />
    ├── patches_3d.py <br />
    ├── pyramids_3d.py <br />
    └── UseClassifier <br />
        ├── UseClassifier.py <br />
        └── ViewResults.py <br />

The dictionaries and classifier weights are serialized.
