results:

| model | image_size | num_preclass | Local | PB |
| :------ | ------: | :------: | :------: | :------: |
| resnet34(2TTA) | 112 | 30k | 0.875 | 0.923 |
| xception(2TTA) | 112 | 50k | 0.886 | 0.932 |
| xception(2TTA) | 96  | 110k| 0.891 | 0.940 |
| xception(10TTA)| 96  | 110k| 0.891 | 0.941 |
| xception(10TTA)| 96  | all | 0.894 | 0.941 |
| xception(10TTA)| 96  | all+110k | 0.894 | 0.942 |
| inceptionv4(10TTA)| 96  | all | 0.894 | 0.941 |
| inceptionv4+xcepttion(10TTA)| 96  | all | 0.894 | 0.943 |
| seresnext101(10TTA)| 96  | all | 0.895 | 0.944 |
| seresnext50+101+xception+inceptionv4_2drawingways| 96  | all | 0.902 | 0.947 |




