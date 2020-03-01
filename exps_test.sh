#/bin/bash

#python main.py --epochs  --batch-size --lr --gamma --drop_out --erase_b . --erase_w .

DP0='none'
DP1='6,.5'
DP2='6,.5,7,.5'
DP3='7,.5'
DP='none'
# for DP in {$DP1,$DP2,$DP3}; do
for LR in {1.,}; do
for GAMMA in {.7,}; do
for EPOCHS in {10,}; do
for BS in {64,}; do
ROT=0
# for ROT in {15,30,45}; do
# for EB in {0,.2,.4,.6,.8,1}; do
# for EW in {0,.2,.4,.6,.8,1}; do
for EB in {0,}; do
for EW in {0,}; do
MODEL_PATH=exps/run_BS_64_EPS_10_LR_1.0_GAMA_0.7_NONORM_1583071942/mnist.pth
# MODEL_PATH=exps/run_BS_64_EPS_10_LR_1.0_GAMA_0.7_1583071938/mnist.pth
MODEL_PATH=exps/run_BS_64_EPS_10_LR_1.0_GAMA_0.7_BN_NONORM_1583075232/mnist.pth
python main.py --epochs $EPOCHS  --batch-size $BS --lr $LR --gamma $GAMMA --drop_out $DP --erase_b $EB --erase_w $EW --rotation $ROT --no-norm --test --batch-norm --model-path $MODEL_PATH
python main.py --epochs $EPOCHS  --batch-size $BS --lr $LR --gamma $GAMMA --drop_out $DP --erase_b $EB --erase_w $EW --rotation $ROT --test --batch-norm --model-path $MODEL_PATH
done
done
done
done
done
done
# done
# done