#/bin/bash

DP0='none'
DP1='6,.5'
DP2='6,.5,7,.5'
DP3='7,.5'
DP='none'
# for DP in {$DP1,$DP2,$DP3}; do
for LR in {1.,}; do
for GAMMA in {.7,}; do
for EPOCHS in {15,}; do
for BS in {64,}; do
ROT=0
# for ROT in {15,30,45}; do
# for EB in {0,.2,.4,.6,.8,1}; do
# for EW in {0,.2,.4,.6,.8,1}; do
for EB in {.6,}; do
for EW in {1.,}; do
python main.py --epochs $EPOCHS  --batch-size $BS --lr $LR --gamma $GAMMA --drop_out $DP --erase_b $EB --erase_w $EW --rotation $ROT
#python main.py --epochs $EPOCHS  --batch-size $BS --lr $LR --gamma $GAMMA --drop_out $DP --erase_b $EB --erase_w $EW --ADAM
done
done
done
done
done
done
# done
# done