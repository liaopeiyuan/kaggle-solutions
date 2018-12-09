### Update 11.26
- 5 different drawing methods, with comments(both English and Chinese), with the corresponding example images
- All methods are tested just now and if you don't wanna use my augmentation method, please remove the "seq" part of the function
- By the way, for the encode by time related functions, please input the image from train_simplified_time (which has x,y,time)

#### Reproduce my processes: see the image "training processes"
let me upload my checkpoint in Google Drive & 百度云, may be you can finetune the model by adding more data for seresnext50 model

In split_data, if you didn't change the random seed, our validation set should be same. Just in case, I will also upload my validation set. You can confirm if they are same. Or, you can input my checkpoint and continue to train (just for test), the first iter should show ~0.887

Update: I have sent you the data via your email addresses, plz take a look

Update 11.21, I have updated my code in order to use 6 channels for training. Please check the readme file.
