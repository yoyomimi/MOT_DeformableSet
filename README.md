# MOT_DeformableSet

## Paper arxiv link: [TR-MOT: Multi-Object Tracking by Reference](https://arxiv.org/pdf/2203.16621.pdf  "paper").

## Method
![Image](https://github.com/yoyomimi/MOT_DeformableSet/blob/large/TR-MOT.png)

## Data
- The data should be prepared in the following structure (one example):
```
data/
   |———  mot_pkl
   |        └——————train
   |        |        └——————Crowdhuman_train.pkl
   |        |        └——————Crowdhuman_val.pkl
   |———  mot17_pkl
   |        └——————train
   |        |        └——————MOT17_train.pkl
   |        |

```
- Some XXX_pkl dirs:  [ MOT_data ](https://drive.google.com/drive/folders/10nWOoOa40ZvTI0t5tYLsEQhGkt7s1GBk?usp=sharing  " MOT_data ").


## Config File
- OUTPUT_ROOT: Save model name & saved work dir name under work_dirs/
- DATASET.FILE & DATASET.NAME: Dataset file name and corresponding Class module name
- DATASET.ROOT: root path for the anno files
- DATASET.PREFIX: Root of the image dir, i.e. prefix of the img path in the anno files
- MODEL.FILE & MODEL.NAME: Model file name and corresponding Class module name
- MODEL.RESUME_PATH & TRAIN.RESUME: Set TRAIN.RESUME to False to load the single model from resume path, or load the optimizer and train state too
- TRAINER.FILE & TRAINER.NAME: Trainer file name and corresponding Class module name


## Process the dataset id
- libs.dataset.match_ch.py (crowdhuman) line 33 - line 45 (create image pair using affine transform)

crowdhuman_idnum = 0 (do not invovle the id of crowdhuman  to self-supervise the id appearance learning or rs module training)

crowdhuman_idnum = 6000 (invovle 6000 human boxes to self-supervise the id appearance learning and rs module training)

commented (involve all the crowdhuman boxes to self-supervise the id appearance learning and rs module training)

```python
id_base = 0
cur_id = 0
for i, anno in enumerate(self.annotations):
    anno['ann']['extra_anns'] = np.array(anno['ann']['extra_anns']).reshape(-1, )
    if len(anno['ann']['extra_anns']) == 0:
        anno['ann']['extra_anns'] = -np.ones(len(anno['ann']['bboxes'])).reshape(-1, )
    for j, single_id in enumerate(anno['ann']['extra_anns']):
        if cur_id >= $crowdhuman_idnum$:
            anno['ann']['extra_anns'][j] = -1
        else:
            anno['ann']['extra_anns'][j] = cur_id
            cur_id += 1
 ```
 
 - libs.dataset.customtask.py (MOT17) line 33 - line 45 (create image pair using two frames adjacent in temporal line)
 ```python
id_base = $crowdhuman_idnum$ + 1
max_id = $crowdhuman_idnum$ + 1
 ```
 
 - configs.deformable_track.yaml RS.NIDS = total_idnum + 1, total_idnum = crowdhuman_idnum + mot17_idnum
 
 
## Pretrain on CrowdHuman and MOT17
```shell
srun --partition=$PARTITION --mpi=pmi2 -n 12 --gres=gpu:6 --ntasks-per-node=8 --job-name=TRAIN --kill-on-bad-exit=1  python3 tools/train.py --cfg configs/deformable_macthtrack_ch.yaml
```
- Involve all crowdhuman boxes or most of them for id appearance supervision and rs module training. The invovled box num is up to the GPU memory.
- DATASET.IMG_NUM_PER_GPU = 1
- 20 epoch training without lr drop
- TRAIN.LR: 0.0001
- My pretrained weight: [ full_chid_rs_pretrain.pth ](https://drive.google.com/file/d/1J6Jkw0Gx1pAviQlobLb9pV4BTpVfrI9J/view?usp=sharing  " full_chid_rs_pretrain.pth ").
- Deformable DETR pretrained weight: [ r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth ](https://drive.google.com/file/d/1MJkFL5xEWA7F5YLph0fQ2rwmhaFs_HeW/view?usp=sharing  " r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth ").

## Finetune on MOT
```shell
srun --partition=$PARTITION --mpi=pmi2 -n 12 --gres=gpu:6 --ntasks-per-node=8 --job-name=TRAIN --kill-on-bad-exit=1  python3 tools/train.py --cfg configs/deformable_track.yaml
```
- Involve 0 crowdhuman boxes (crowdhuman_idnum = 0) for id appearance training or rs module training
- DATASET.IMG_NUM_PER_GPU = 1
- 30 epoch training with lr drop by 0.1 in epoch 20
- TRAIN.LR: 0.0001
- MODEL.RESUME_PATH: [ full_chid_rs_pretrain.pth ](https://drive.google.com/file/d/1J6Jkw0Gx1pAviQlobLb9pV4BTpVfrI9J/view?usp=sharing  " full_chid_rs_pretrain.pth "). TRAIN.RESUM = False.

## Test on MOT
- tools.offset_track.py line 174
thr = 0.38
```python
def process_img(img_path, model, postprocessors, device, threshold=$thr$, references=None):
```
- tools.offset_track.py line 490-
```python
data_dir =  # your own data dir path (e.g. '/mnt/lustre/chenmingfei/data/MOT_data/')
# test mot17 (you can uncomment the val mot17 or any else, set seqs_str and data_root correspondingly)
seqs_str = '''MOT17-01-SDP
              MOT17-03-SDP
              MOT17-06-SDP
              MOT17-07-SDP
              MOT17-08-SDP
              MOT17-12-SDP
              MOT17-14-SDP'''
data_root = os.path.join(data_dir, 'MOT17/test')
```

- test 
```shell
PARTITION = pat_largescale
srun --partition=$PARTITION --mpi=pmi2 -n 1 --gres=gpu:1 --ntasks-per-node=8 --job-name=EVAL python3 tools/offset_track.py --cfg [config_path](e.g. configs/deformable_track.yaml) \
     MODEL.RESUME_PATH [resume_path](e.g. output/matchtrack_epoch025_checkpoint.pth)
```

- Results saved in work_dirs/$output_root$/results/

- Test results on MOT17, thr=0.38-0.42 (0.38 will be fine), our results is FRT (emphasized in red) in the following result picture.
![Image](https://github.com/yoyomimi/MOT_DeformableSet/blob/large/results.png)

## YOLOv5 + RS
```shell
srun --partition=$PARTITION --mpi=pmi2 -n 12 --gres=gpu:6 --ntasks-per-node=8 --job-name=TRAIN --kill-on-bad-exit=1  python3 tools/train.py --cfg configs/YOLOv5m_matchtrack.yaml
```
Pending check to this part...

## Notice
If you get any `address already in use` errror, please modify the port number of the returned ip address from the function `get_ip_address` in train.py or offset_track.py. Or you can just rewrite this function to generate ip address with a random port.
