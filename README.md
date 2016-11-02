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
The dictionaries and classifier weights are serialized in the ./Data/Serialized directory.

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

