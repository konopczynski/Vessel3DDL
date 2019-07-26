# Vessel3DDL

Automated Multiscale 3D Feature Learning for Vessels Segmentation in Thorax CT Images

## Data

The VESSEL 12 data may be downloaded from: https://grand-challenge.org/site/vessel12/
and should be stored at ./Data/VESSEL12/
```
├── Data
│    └── VESSEL12
│        ├── VESSEL12_01-05
│        ├── VESSEL12_01-20_Lungmasks
│        ├── VESSEL12_06-10
│        ├── VESSEL12_11-15
│        ├── VESSEL12_16-20
│        └── VESSEL12_ExampleScans
│            ├── Annotations
│            ├── Lungmasks
│            └── Scans
├── LICENSE
├── README.md
└── scripts
    ├── config.py
    ├── config.pyc
    ├── LearnClassifier
    ├── LearnDictionary
    ├── UseClassifier
    └── utils
```
## Structure
The entire processing pipeline for the VESSEL12 data is set up in the config.py file.
* Dictionary learning (Unsupervised step). First the dictionary has to be learned on a number of given volumes. The volumes don't have to be annotated. 
* Classifier learning (Supervised step). Based on the learned features, train the classifier of choice.
* Testing module. Apply filters from the dictionary and use a classifier.
* Some additional functionality: 3d patch extraction, 3d Gaussian pyramids, loading/saving data.
The dictionaries and classifier weights are serialized in the ./Data/Serialized directory.

### LearnDictionary
Execute the scripts in following order: <br />
1. ExtractPatches.py <br />
2. LearnDictionary.py <br />
### LearnClassifier
Execute the scripts in following order: <br />
1. ExtractXy_multithread.py <br />
2. ConcatenateXy.py <br />
3. TrainClassifier.py or MakeMeasurements.py <br />
### Usage
Once the dictionary and classifier are learned, they can by uses on a given volume. <br />
Execute the scripts in following order: <br />
1. UseClassifier.py <br />
2. ViewResults.py <br />

## Citation
Please cite these paper in your publications if it helps your research:

    @inproceedings{konopczynski2019automated,
        title={Automated multiscale 3D feature learning for vessels segmentation in Thorax CT images},
        author={Konopczy{\'n}ski, Tomasz and Kr{\"o}ger, Thorben and Zheng, Lei and Garbe, Christoph S and Hesser, J{\"u}rgen},
        booktitle={2016 IEEE Nuclear Science Symposium, Medical Imaging Conference and Room-Temperature Semiconductor Detector Workshop (NSS/MIC/RTSD)},
        pages={1--3},
        year={2019},
        organization={IEEE}
    }
    
Link to the paper:

- [Automated Multiscale 3D Feature Learning for Vessels Segmentation in Thorax CT Images](https://arxiv.org/abs/1901.01562)
