# Vessel3DDL

Automated Multiscale 3D Feature Learning for Vessels Segmentation in Thorax CT Images

## Instalation 

The VESSEL 12 data may be downloaded from: https://grand-challenge.org/site/vessel12/
and should be stored at ./Data/VESSEL12/

TBD

## Structure

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

### LearnDictionary
Execute the scripts in following order:
1. ExtractPatches.py
2. LearnDictionary.py
### LearnClassifier
Execute the scripts in following order:
1. ExtractXy_multithread.py
2. ConcatenateXy.py
3. TrainClassifier.py or MakeMeasurments.py
### Usage
Once the dictionary and classifier are learned,
one can use them on a given volume.
Execute the scripts in following order:
1. UseClassifier.py
2. ViewResults.py

## Reference
TODO

