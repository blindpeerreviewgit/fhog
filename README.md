# aaai18
Code for paper blind review

-----------
Information
-----------

This code uses a modified version of Dlib 19.4 (http://dlib.net/) which is included in this repository.

The training and testing datasets are included. Images are obtained from multiple sources, including the SPQR dataset (http://www.dis.uniroma1.it/~labrococo/?q=node/459).

Level up to which scheme is applied is hard-coded to 7 in dlib-19.4-modified/dlib/image_processing/scan_fhog_pyramid.h line:928 ([link to line](https://github.com/blindpeerreviewgit/fhog/blob/master/dlib-19.4-modified/dlib/image_processing/scan_fhog_pyramid.h#L928)) (sorry, research code...). After changes, the library must be recompiled.

Other parameters are found in the same file as above, at lines 685-693.

Configurations:

Original baseline:
* set USE_HACK to 0

Power law (Dollar et al. 2014):
* set USE_HACK to 1
* set hybridLevel to 10

Paper results:
* set USE_HACK to 1
* set hybridLevel to values between 2 and 9

Note: Average precision on the testing dataset will be lower than those reported in the paper. Results from the paper were obtained by doing a hyperparameter search at each level. This implementation has hardcoded hyperparameters and is only optimized for the original baseline.

--------------------
Instructions
--------------------
```
git clone https://github.com/blindpeerreviewgit/fhog.git
cd fhog
cd dlib-19.4-modified
mkdir build
cd build
cmake ..
make -j
cd ../..
mkdir build
cd build
cmake ..
make -j
./learnRobot ../dataset ../dataset
```
-------------
Sample output
-------------
```
[...]

objective:     1398.83
objective gap: 1398.82
risk:          46.8166
risk gap:      46.8166
num planes:    3
iter:          1

objective:     506.993
objective gap: 506.687
risk:          16.9581
risk gap:      16.9581
num planes:    4
iter:          2

objective:     772.134
objective gap: 771.459
risk:          25.8197
risk gap:      25.8197
num planes:    5
iter:          3

[...]

hack time: 0.00600257
avg hack time: 0.00665809
correct_hits=161 total_hits=168 total_true_targets=185
testing results (precision, recall, average precision):  0.958333  0.87027 0.865549

[...]
```
Robot detection:

![alt text](https://github.com/blindpeerreviewgit/fhog/raw/master/sample/sample-output.png)

Learned vignette:

![alt text](https://github.com/blindpeerreviewgit/fhog/raw/master/sample/sample-output2.png)
