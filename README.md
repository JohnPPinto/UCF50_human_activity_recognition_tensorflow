# Human Activity Recognition - Video Classification

A project on video classification using Tensorflow with UCF50 dataset.

In this project, I have used two baseline models approach: ConvLSTM and LRCN to tackle the video classification problem.

For more details and for the complete workflow on the project you can check my [notebook](https://github.com/JohnPPinto/UCF50_human_activity_recognition_tensorflow/blob/main/UCF50_human_activity_recognition_tensorflow.ipynb).

If you are interested in training the model and predicting any videos. You can use the python script files in the module directory.

Use train.py for training the dataset and predict.py for predicting the target video.

## Training Instruction

Most of the argument flags have default values, so you can run the train.py script directly.

**Flags values:**
* --model_name: LRCN [ConvLSTM or LRCN]
* --exp_name: 'experiment'
* --num_epochs: 100 
* --batch_size: 32 
* --seq_len: 20 
* --frame_size: 128 
* --lr: 0.001 
* --data_dir_path: data/UCF50 
* --class_list: None [A list containing class names, use None for all the classes] 
* --num_workers: os.cpu_count() 
* --callbacks: True [True or False]

You can check my [train_results.txt](https://github.com/JohnPPinto/UCF50_human_activity_recognition_tensorflow/blob/main/train_result.txt) file for better understanding.

**Example:**
```
python module/train.py --model_name LRCN --exp_name lrcn_exp --num_epochs 100 \
--batch_size 32 --seq_len 20 --frame_size 128 --lr 0.001 --data_dir_path data/UCF50 \
--class_list Biking Diving GolfSwing Punch Rowing --num_workers 8 --callbacks True
```

## Prediction Instruction

**Flags values:**
* --seq_len: 20 
* --frame_size: 128 
* --video_path: A Path for the video file. [Required]
* --model_path: saved_model/LRCN/lrcn_exp/2023-04-16-07:10:51 
* --class_list: Biking Diving GolfSwing Punch Rowing [Change this if you have trained on different classes.]

You can check my [predict_results.txt](https://github.com/JohnPPinto/UCF50_human_activity_recognition_tensorflow/blob/main/predict_result.txt) file for better understanding.

**Example:**
```
python module/predict.py --seq_len 20 --frame_size 128 \
--video_path data/UCF50/GolfSwing/v_GolfSwing_g01_c01.avi \
--model_path saved_model/LRCN/lrcn_exp/2023-04-16-07:10:51 \
--class_list Biking Diving GolfSwing Punch Rowing
```
