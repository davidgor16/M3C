# -*- coding: utf-8 -*-
# @Time    : 17/09/25 12:17 PM
# @Author  : David Fernandez Garcia
# @Affiliation  : Universidad de Valladolid (Research Group ECA-SIMM)
# @Email   : david.fernandez@uva.es
# @File    : run_M3C.sh

#!/usr/bin/env bash


# GENERAL PARAMETERS
model=m3c
am=features_vc_clean_embbedings_norm

# TRAINING PARAMETERS
lr=1e-3
batch_size=2
epochs=50
embed_dim=24

# V/C FEATURE EXTRACTION PARAMETERS
num_convs_vc=32
input_dim_vowels=15
input_dim_consonants=24
output_dim_vc=30
dropout_cnn_vc=0.4
dropout_mlp_vc=0.

# SSL FEATURE EXTRACTION PARAMETERS
num_convs_ssl=2
input_dim_ssl=1024
output_dim_ssl=30
dropout_cnn_ssl=0.8
dropout_mlp_ssl=0.

# FEATURE FUSION PARAMETERS
fusion_dim=30
dropout_mlp_fusion=0.3

# PHONE-LEVEL EXTRACTION PARAMETERS
num_convs_phn=32
output_dim_phn=30
dropout_cnn_phn=.0
dropout_mlp_phn=.0

# WORD-LEVEL EXTRACTION PARAMETERS
num_convs_word=32
output_dim_word=30
dropout_cnn_word=0.2
dropout_mlp_word=.0

# UTTERANCE-LEVEL EXTRACTION PARAMETERS
num_convs_utt=32
output_dim_utt=30
dropout_cnn_utt=.0
dropout_mlp_utt=.0

# MDD PARAMETERS
alpha_mdd=0.03

exp_dir=../exp/m3c-${lr}-${depth}-${head}-${batch_size}-${embed_dim}-${model}-${am}-br

# repeat times
repeat_list=(0)

for repeat in "${repeat_list[@]}"
      do
      mkdir -p $exp_dir-${repeat}
      python ./traintest.py --lr ${lr} --exp-dir ${exp_dir}-${repeat} \
      --batch_size ${batch_size} --embed_dim ${embed_dim} --model ${model} --am ${am} --n-epochs ${epochs} --alpha_mdd ${alpha_mdd} \
      --num_convs_vc ${num_convs_vc} --input_dim_vowels ${input_dim_vowels} --input_dim_consonants ${input_dim_consonants} --output_dim_vc ${output_dim_vc} --dropout_cnn_vc ${dropout_cnn_vc} --dropout_mlp_vc ${dropout_mlp_vc} \
      --num_convs_ssl ${num_convs_ssl} --input_dim_ssl ${input_dim_ssl} --output_dim_ssl ${output_dim_ssl} --dropout_cnn_ssl ${dropout_cnn_ssl} --dropout_mlp_ssl ${dropout_mlp_ssl} \
      --fusion_dim ${fusion_dim} --dropout_mlp_fusion ${dropout_mlp_fusion} \
      --num_convs_phn ${num_convs_phn} --output_dim_phn ${output_dim_phn} --dropout_cnn_phn ${dropout_cnn_phn} --dropout_mlp_phn ${dropout_mlp_phn} \
      --num_convs_word ${num_convs_word} --output_dim_word ${output_dim_word} --dropout_cnn_word ${dropout_cnn_word} --dropout_mlp_word ${dropout_mlp_word} \
      --num_convs_utt ${num_convs_utt} --output_dim_utt ${output_dim_utt} --dropout_cnn_utt ${dropout_cnn_utt} --dropout_mlp_utt ${dropout_mlp_utt} \

      done
python ./collect_summary.py --exp-dir $exp_dir
