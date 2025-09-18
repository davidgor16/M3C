#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=24000
#SBATCH --job-name="gopt"
#SBATCH --output=../exp/log_%j.txt

#set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source ../venv-gopt/bin/activate

# PARAMETROS GENERALES
model=gopt_vc
am=features_vc_clean_embbedings_norm

# PARAMETROS DE ENTRENAMIENTO
lr=1e-3
batch_size=2
epochs=50
embed_dim=24

# PARAMETROS DE EXTRACCION DE CARACTERISTICAS V/C
num_convs_vc=32
input_dim_vowels=15
input_dim_consonants=24
output_dim_vc=30
dropout_cnn_vc=0.4
dropout_mlp_vc=0.

# PARAMETROS DE EXTRACCION DE CARACTERISTICAS SSL
num_convs_ssl=2
input_dim_ssl=1024
output_dim_ssl=30
dropout_cnn_ssl=0.8
dropout_mlp_ssl=0.

# PARAMETROS DE FUSION DE CARACTERISTICAS
fusion_dim=30
dropout_mlp_fusion=0.3

# PARAMETROS DE EXTRACCION NIVEL FONEMA
num_convs_phn=32
output_dim_phn=30
dropout_cnn_phn=.0
dropout_mlp_phn=.0

# PARAMETROS DE EXTRACCION NIVEL PALABRA
num_convs_word=32
output_dim_word=30
dropout_cnn_word=0.2
dropout_mlp_word=.0

# PARAMETROS DE EXTRACCION NIVEL FRASE
num_convs_utt=32
output_dim_utt=30
dropout_cnn_utt=.0
dropout_mlp_utt=.0

# PARAMETROS MDD
alpha_mdd=0.03


exp_dir=../exp/gopt-${lr}-${depth}-${head}-${batch_size}-${embed_dim}-${model}-${am}-br

# repeat times
repeat_list=(0 1 2 3 4)
list2=(0.3)
list1=(0.0)

for dropout_mlp in "${list1[@]}"
do
  for dropout in "${list2[@]}"
  do
    for repeat in "${repeat_list[@]}"
    do
      mkdir -p $exp_dir-${repeat}
      python ./traintest.py --lr ${lr} --exp-dir ${exp_dir}-${repeat} \
      --batch_size ${batch_size} --embed_dim ${embed_dim} --model ${model} --am ${am} --n-epochs ${epochs} --alpha_mdd ${alpha_mdd} \
      --num_convs_vc ${num_convs_vc} --input_dim_vowels ${input_dim_vowels} --input_dim_consonants ${input_dim_consonants} --output_dim_vc ${output_dim_vc} --dropout_cnn_vc ${dropout_cnn_vc} --dropout_mlp_vc ${dropout_mlp_vc} \
      --num_convs_ssl ${num_convs_ssl} --input_dim_ssl ${input_dim_ssl} --output_dim_ssl ${output_dim_ssl} --dropout_cnn_ssl ${dropout_cnn_ssl} --dropout_mlp_ssl ${dropout_mlp_ssl} \
      --fusion_dim ${fusion_dim} --dropout_mlp_fusion ${dropout} \
      --num_convs_phn ${num_convs_phn} --output_dim_phn ${output_dim_phn} --dropout_cnn_phn ${dropout_cnn_phn} --dropout_mlp_phn ${dropout_mlp_phn} \
      --num_convs_word ${num_convs_word} --output_dim_word ${output_dim_word} --dropout_cnn_word ${dropout_cnn_word} --dropout_mlp_word ${dropout_mlp} \
      --num_convs_utt ${num_convs_utt} --output_dim_utt ${output_dim_utt} --dropout_cnn_utt ${dropout_cnn_utt} --dropout_mlp_utt ${dropout_mlp_utt} \

    done
    python ./collect_summary.py --exp-dir $exp_dir
  done
done