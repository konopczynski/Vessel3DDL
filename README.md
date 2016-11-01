Vessel3DDL

# README #

### Structure of this ###

* Dictionary learning (Unsupervised step). First the dictionary has to be learned on a number of given volumes. The volumes don't have to be annotated. 
* Classifier learning (Supervised step). Based on the learned features, train the classifier of choice.
* Testing module. Apply filters learned from the learned dictionary and use a classifier on that.

### Dictionary learning ###

The dictionaries and classifier weights are stored in some files.
So one can reuse them once they are learned, or fine-tune them later.


