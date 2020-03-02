# Semi Adversarial Network
Neural Network exam's project. 
Keras implementation of the work done in:
>_Semi-Adversarial Networks: Convolutional Autoencoders for Imparting Privacy to Face Images_ by V. Mirjalili, S. Raschka, A. M. Namboodiri, and A. Ross. https://arxiv.org/abs/1712.00321

Our work is based on the preprocessed **Celeb-A** dataset provided by 
[iPRoBe-lab's implementation](https://github.com/iPRoBe-lab/semi-adversarial-networks)

For more details about this project read our final [report](https://github.com/gianluca-pepe/semi-adversarial-network/blob/master/report.pdf).
## Usage

### Dataset
First you need to download and preprocess dataset.

#### CelebA 
Our work is based on the **Celeb-A** dataset provided by 
[iPRoBe-lab's implementation](https://github.com/iPRoBe-lab/semi-adversarial-networks)

- Download training set: [here](https://drive.google.com/file/d/1sd3TyefiPqvxIdoGl7Ysm3rxnjSDgP5h/)
- Download test set: [here](https://drive.google.com/open?id=12m2oQzkt3aXxOSPSRugqGwwertVA9pAa)

Place both `images-dpmcrop-train` and `images-dpmcrop-test` folders in repo's root and just run  
```
cd src/pre_processing
python prepare_dataset_celeba.py
```
#### Notes
At the end of pre-processing the `dataset` folder will be structured as follows:

``` 
dataset
|
+-- celebA
|   |   
|   +-- test
|   |   +-- female
|   |   +-- male
|   +-- train
|   |   +-- female
|   |   +-- male
|   +-- validation
|   |   +-- female
|   |   +-- male
+-- prototype
```
---
### Pre Training
In this phase models of the Semi-Adversarial-Network are trained independently.
AutoEncoder and Gender-Predictor have been pre-trained by us.

```
cd src/
```
For pretraining the Autoencoder
```
python pretrain_autencoder.py
```
For pretraining the Gender Classifier
```
python pretrain_genderclassifier.py
```

_Face-Matcher uses downloaded weights (in the same way of the reference paper) during Further-Training so we didn't need to pre-train it._

Obtained weights can be found here:
https://drive.google.com/file/d/1OgnvG74Qru9tbeXImUldcr60HuJoLrov/view?usp=sharing

---
### Further Training
Train the autoencoder using other modules' feedbacks.

The complete Semi-Adversarial model (`modules/san.py`) is formed by the following NNs:

- AutoEncoder
- Gender Classifier
- Face Matcher

Face Matcher and Gender Classifier are not trainable components during this phase.
<br/>_Be sure to have placed pretrained weights in_ `/weights` _and named them as follows:_
```
- autoencoder_pretrain_weights.h5
- genderclassifier_pretrain_weights.h5
- facematcher_pretrain_weights.h5
```
These are the only weights considered by `further_train.py`, so if you retrain one of the modules, please rename obtained
weights accordingly or manage weights loading as you prefer in each modules in `/src/modules`.
<br/>Then you can run further train with
```
cd src/
python further_train.py  
```

Obtained weights will be saved in  `/weights` with a timestamp attached to the file name. 
<br/>Obtained weights can be found here:
https://drive.google.com/file/d/1O4hjK8BBsD922SVnFEkPDitE4MEZ5ss8/view?usp=sharing



This project was developed with passion by: Bruno Marino & Gianluca Pepe.


