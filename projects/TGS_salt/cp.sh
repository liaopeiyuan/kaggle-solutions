#!/bin/bash
t="train0_Unet_scSE_hyper.py"
for i in `seq 1 9`;
do
	cp train0_Unet_scSE_hyper.py "${t/0/$i}"
done
