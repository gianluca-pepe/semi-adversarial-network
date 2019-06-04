# Dataset Preprocessing

Our work is based on the **Celeb-A** dataset provided by 
[iPRoBe-lab's implementation](https://github.com/iPRoBe-lab/semi-adversarial-networks)

- Download training set: [here](https://drive.google.com/file/d/1sd3TyefiPqvxIdoGl7Ysm3rxnjSDgP5h/)
- Download test set: [here](https://drive.google.com/open?id=12m2oQzkt3aXxOSPSRugqGwwertVA9pAa)

Place both `images-dpmcrop-train` and `images-dpmcrop-test` folders in repo's root and just run  
```
python prepare_dataset.py
```
*The script may require long time to complete the process*
<br/>Alternatively the final obtained custom dataset can also be found here:
https://github.com/CuriousDiscoverer/Custom-Celeb_A

## Notes
At the end of pre-processing we obtain the *dataset* folder structured as follows:

``` 
    dataset
    |   
    +-- prototype
    +-- test
    |   +-- female
    |   +-- male
    +-- train
    |   +-- female
    |   +-- male
    +-- validation
    |   +-- female
    |   +-- male
```

