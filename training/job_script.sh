#!/usr/local/bin/zsh

# Name the job
#SBATCH --job-name=tf_faster_rcnn_train

# Declare the merged STDOUT/STDERR file
#SBATCH --output=output.%J.txt
 
# ask for 10 GB memory
#SBATCH --mem-per-cpu=10240M   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

#SBATCH --gres=gpu:volta:2

#SBATCH --time=60

### Change to the work directory
cd $HOME/master_thesis/yellow_sticky_network
 
### load modules and execute
module unload intelmpi
module switch intel gcc
module load python/3.6.8
module load cuda/101
module load cudnn/7.6.5

# load venv
source ~/.zshrc
source venv/bin/activate
 
# start non-interactive batch job
#python3 models/research/object_detection/builders/model_builder_tf2_test.py
python3 model_main_tf2.py --model_dir=training/faster_rcnn/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8 --pipeline_config_path=training/faster_rcnn/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8/pipeline.config

