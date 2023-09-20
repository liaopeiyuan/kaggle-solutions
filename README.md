# kaggle-solutions

(Updated Sep 2023) This is a repository containing some of my solutions to Kaggle competitions from 5 years ago. It is no longer being maintained. Feel free to use them as you wish.

## Dependencies
- tensorflow-gpu
- torch
- torchvision
- tensorboard
- Pillow
- opencv-python
- pandas
- numpy

## Past Collaborators

TGS: Renan @scherzocrk, Ding Han @DhDiego 

Doodle: Venn @wangvenn, SeuTao @SeuTao, earhian @earhian, Tom57 @ThomasTilli, CHAN, wilan @wilanw, Andi Leslie Yan @RedLiN5

## Current projects:

1. [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge/leaderboard)
- Placement: **5/3291** (Gold Medal)
- [Solution Journal](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69051)
- **Model Performances:**

|model|#fold|#models|public LB|private LB|placement|
|-------------------------|-|-|-----|-----|-------|
|ResNet34-512|1|1|0.859|0.862|N/A|
|ResNet34-256|1|1|0.868|0.870|N/A|
|ResNet34-128|1|1|0.837|0.855|N/A|
|ResNet34-128|1|6|0.854|0.873|N/A|
|ResNet34-128|10|60|0.869|0.884|N/A|
|SE-ResNeXt-50-BN-128|3|3|0.861|0.880|N/A|
|ResNet34-128+256-post|1|2|0.875|0.888|22|
|SE-ResNeXt-50-OC-128|5|15|0.869|0.889|20|
|SE-ResNeXt-50-OC-256-post|10|95|0.881|0.892|9|
|ResNet34-256|10|60|0.875|0.894|8|
|2models-2scale-stable-post|10|100|0.885|0.894|5|

2. [Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition/leaderboard)
- Placement: **12/1316** (Gold Medal)
- [Solution Journal](https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73710)
