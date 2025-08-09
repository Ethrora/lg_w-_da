--------------train--------------
nohup env CUDA_VISIBLE_DEVICES=0,1,2 python -m gluefactory.train [sp+lg_da_homography] --conf gluefactory/configs/superpoint+lightglue_da_homography.yaml > train.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0,1,2 python -m gluefactory.train [sp+lg_da_2heads_megadepth] --conf gluefactory/configs/superpoint+lightglue_da_megadepth.yaml train.load_experiment=[sp+lg_da_2heads_homography] > finetune.log 2>&1 &
注：bs=16
nohup env CUDA_VISIBLE_DEVICES=0,1,2 python -m gluefactory.train [sp+lg_da_megadepth] --conf gluefactory/configs/superpoint+lightglue_da_megadepth.yaml --restore > train_resume.log 2>&1 &

--------------test--------------
python -m gluefactory.eval.megadepth1500 --checkpoint [sp+lg_da_megadepth]
python -m gluefactory.eval.hpatches --checkpoint [sp+lg_da_megadepth]