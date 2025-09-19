# -*- coding: utf-8 -*-
# @Time    : 17/09/25 12:17 PM
# @Author  : David Fernandez Garcia
# @Affiliation  : Universidad de Valladolid (Research Group ECA-SIMM)
# @Email   : david.fernandez@uva.es
# @File    : M3C.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from custom_layers.attention import Aspect_Attention_op2
from custom_layers.scoreRestraintAttentionPooling import ScoreRestraintAttentionPooling
import torch.nn as nn
import torch.nn.functional as F


class CCC(nn.Module):
    def __init__(self, output_dim, num_convs=32, input_dim=24, dropout_cnn=.0, dropout_mlp=.0, input_channels = 1, kernel_size=(3,1)):
        super().__init__()
        
        self.embed_dim = input_dim
        self.num_kernels_layer1 = num_convs
                
        self.conv = nn.Conv2d(input_channels, num_convs, kernel_size=kernel_size)
                        
        self.phn_proj = nn.Linear(num_convs*(input_dim-(kernel_size[1]-1)),output_dim)
        
        self.layerNorm1 = nn.LayerNorm(num_convs*(input_dim-(kernel_size[1]-1)))
        
        self.dropout_cnn = nn.Dropout(dropout_cnn)
        
        self.dropout_mlp = nn.Dropout(dropout_mlp)
        
        self.layerNorm2 = nn.LayerNorm(output_dim)
                                   
    def forward(self, x):
                  
        x = self.conv(x)
               
        x = x.flatten(1)
                
        x = F.relu(self.layerNorm1(x))
        
        x = self.dropout_cnn(x)
        
        x = F.relu(self.layerNorm2(self.phn_proj(x)))
        
        x = self.dropout_mlp(x)
                
        return x
#----------------------------------------------------------------------------------------------------

class M3C(nn.Module):
    def __init__(self,
                 # Parameters for V/C feature extraction
                 num_convs_vc, input_dim_vowels, input_dim_consonants, output_dim_vc, dropout_cnn_vc, dropout_mlp_vc,
                 # Parameters for SSL feature extraction
                 num_convs_ssl, input_dim_ssl, output_dim_ssl, dropout_cnn_ssl, dropout_mlp_ssl,
                 # Parameters for SSL and V/C fusion
                 fusion_dim, dropout_mlp_fusion,
                 # Phoneme-level parameters
                 num_convs_phn, output_dim_phn, dropout_cnn_phn, dropout_mlp_phn,
                 # Word-level parameters
                 num_convs_word, output_dim_word, dropout_cnn_word, dropout_mlp_word,
                 # Utterance-level parameters
                 num_convs_utt, output_dim_utt, dropout_cnn_utt, dropout_mlp_utt,
                 ):
        super().__init__()
        
        self.output_dim_duration_energy = 8
        self.output_dim_vc = output_dim_vc
        self.output_dim_ssl = output_dim_ssl
        self.fusion_dim = fusion_dim
        self.output_dim_phn = output_dim_phn
        self.output_dim_word = output_dim_word
        self.output_dim_utt = output_dim_utt
        
        # FEATURE EXTRACTION LEVEL
        
        # GOP
        self.ccc_vowels= CCC(input_dim=input_dim_vowels, num_convs=num_convs_vc, output_dim=output_dim_vc,dropout_cnn=dropout_cnn_vc, dropout_mlp=dropout_mlp_vc)
        self.ccc_consonants= CCC(input_dim=input_dim_consonants, num_convs=num_convs_vc, output_dim=output_dim_vc,dropout_cnn=dropout_cnn_vc, dropout_mlp=dropout_mlp_vc)
    
        # SSL
        self.ccc_ssl= CCC(input_dim=input_dim_ssl, num_convs=num_convs_ssl, output_dim=output_dim_ssl ,dropout_cnn=dropout_cnn_ssl, dropout_mlp=dropout_mlp_ssl)
        
        # Feature fusion
        self.fusion_mlp = nn.Sequential(nn.Linear(output_dim_vc+output_dim_ssl+self.output_dim_duration_energy,fusion_dim),nn.LayerNorm(fusion_dim),nn.Dropout(dropout_mlp_fusion))
        
        #--------------------------------------------------------------------------------------------------------------------
        
        # PHONEME LEVEL
        self.ccc_phn= CCC(input_dim=fusion_dim+1, num_convs=num_convs_phn, output_dim=output_dim_phn, dropout_cnn=dropout_cnn_phn, dropout_mlp=dropout_mlp_phn, kernel_size=(3,1))
        
        self.proj_phn = nn.Linear(output_dim_phn, output_dim_phn)
        self.proj_mdd = nn.Linear(output_dim_phn, output_dim_phn)
        self.phn_attn_tmp = Aspect_Attention_op2(output_dim_phn)
     
        self.mlp_phn = nn.Sequential(nn.Linear(output_dim_phn,1))
        self.mlp_phn_mdd = nn.Sequential(nn.Linear(output_dim_phn, 48))
        
        #--------------------------------------------------------------------------------------------------------------------
        
        # WORD LEVEL
        self.ccc_word= CCC(input_dim=output_dim_phn, num_convs=num_convs_word, output_dim=output_dim_word, dropout_cnn=0., dropout_mlp=dropout_mlp_word, kernel_size=(12,1))
                
        self.proj_w1 = nn.Sequential(nn.Linear(output_dim_word, output_dim_word), nn.Dropout(dropout_cnn_word))
        self.proj_w2 = nn.Sequential(nn.Linear(output_dim_word, output_dim_word), nn.Dropout(dropout_cnn_word))
        self.proj_w3 = nn.Sequential(nn.Linear(output_dim_word, output_dim_word), nn.Dropout(dropout_cnn_word))
        self.word_attn_tmp = Aspect_Attention_op2(output_dim_word)
    
        self.mlp_word_w1 = nn.Sequential(nn.Linear(output_dim_word,1))
        self.mlp_word_w2 = nn.Sequential(nn.Linear(output_dim_word,1))
        self.mlp_word_w3 = nn.Sequential(nn.Linear(output_dim_word,1))
        
        #--------------------------------------------------------------------------------------------------------------------
        
        # UTTERANCE LEVEL
        self.ccc_utt= CCC(input_dim=output_dim_utt, num_convs=num_convs_utt, output_dim=output_dim_utt, dropout_cnn=dropout_cnn_utt, dropout_mlp=dropout_mlp_utt, kernel_size=(50,1))
        
        self.proj_u1 = nn.Linear(output_dim_utt, output_dim_utt)
        self.proj_u2 = nn.Linear(output_dim_utt, output_dim_utt)
        self.proj_u3 = nn.Linear(output_dim_utt, output_dim_utt)
        self.proj_u4 = nn.Linear(output_dim_utt, output_dim_utt)
        self.proj_u5 = nn.Linear(output_dim_utt, output_dim_utt)
        self.utt_attn_tmp = Aspect_Attention_op2(output_dim_utt)
        
        self.utt_score_attn = ScoreRestraintAttentionPooling(num_scores=52, hidden_dim=output_dim_utt)

        self.mlp_utt_u1 = nn.Sequential(nn.Linear(output_dim_utt,1))
        self.mlp_utt_u2 = nn.Sequential(nn.Linear(output_dim_utt,1))
        self.mlp_utt_u3 = nn.Sequential(nn.Linear(output_dim_utt,1))
        self.mlp_utt_u4 = nn.Sequential(nn.Linear(output_dim_utt,1))
        self.mlp_utt_u5 = nn.Sequential(nn.Linear(output_dim_utt,1))
        
        #--------------------------------------------------------------------------------------------------------------------

    def forward(self, x, phn, words, dur_feat, ener_feat,w2v_feat,hubert_feat,wavlm_feat):
        
        
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
        with open('dicts/arpabet_to_vc.json', 'r') as file:
            arpa_vc = json.load(file)
              
        with open('dicts/pureLabel_to_(0-41).json', 'r') as file:
            arpa_long = json.load(file)
             
        with open('dicts/(3-41)_to_(0,numPhonesUsed).json', 'r') as file:
            long_short = json.load(file)
            
        long_vc = {int(value): arpa_vc.get(key, key) for key, value in arpa_long.items()}
        del long_vc[0]
        del long_vc[1]
        del long_vc[2]
        short_vc ={long_short[str(key)]: value for key, value in long_vc.items()}
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
        
        # DISTINCTION BETWEEN VOWELS AND CONSONANTS IN phn
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
        
        max_key = max(short_vc.keys())
        lookup_table_svc = torch.zeros(max_key+2, dtype=torch.float32)
        for key, value in short_vc.items():
            if value == "V":
                lookup_table_svc[key] = 0
                
            if value == "C":
                lookup_table_svc[key] = 1
                
        lookup_table_svc[-1] = -1
        lookup_table_svc = lookup_table_svc.to("cuda")

        phn_vc = lookup_table_svc[phn.int()]
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
               
        # Remove SIM_EMB (Not used in this model)
        x = x[:,:,[0,1,2],:]
            
        # Create a mask to remove positions that correspond to -1s
        mask = (phn != -1)
        
        # Get indices where phn_vc == 1
        index_consonants = torch.where(phn_vc == 1, torch.arange(50, device="cuda").expand(x.shape[0], 50), -1)
        
        # Get indices where phn_vc == 0
        index_vowels = torch.where(phn_vc == 0, torch.arange(50, device="cuda").expand(x.shape[0], 50), -1)
                 
        # Declare the new processed representation
        x_conv = -torch.ones(x.shape[0],x.shape[1],(self.output_dim_vc + self.output_dim_ssl),device="cuda")
                     
        # Process with the CNN for both vowels and consonants
        for i in range(x.shape[0]):
            
            # GOP
            x_vowels = x[i,index_vowels[i] != -1,:,:15]
            x_consonants = x[i,index_consonants[i] != -1]
        
            x_vowels = x_vowels.unsqueeze(1)
            x_consonants = x_consonants.unsqueeze(1)
            
            x_vowels = self.ccc_vowels(x_vowels)
            x_consonants = self.ccc_consonants(x_consonants)
            
            # SSL
            # HuBERT
            hubert_vowels = hubert_feat[i,index_vowels[i] != -1]
            hubert_consonants = hubert_feat[i,index_consonants[i] != -1]
            hubert_vowels = hubert_vowels.unsqueeze(1).unsqueeze(1)
            hubert_consonants = hubert_consonants.unsqueeze(1).unsqueeze(1)
            
            # W2V
            w2v_vowels = w2v_feat[i,index_vowels[i] != -1]
            w2v_consonants = w2v_feat[i,index_consonants[i] != -1]
            w2v_vowels = w2v_vowels.unsqueeze(1).unsqueeze(1)
            w2v_consonants = w2v_consonants.unsqueeze(1).unsqueeze(1)
            
            # WavLM
            wavlm_vowels = wavlm_feat[i,index_vowels[i] != -1]
            wavlm_consonants = wavlm_feat[i,index_consonants[i] != -1]
            wavlm_vowels = wavlm_vowels.unsqueeze(1).unsqueeze(1)
            wavlm_consonants = wavlm_consonants.unsqueeze(1).unsqueeze(1)
            
            ssl_features_vowels = torch.cat((hubert_vowels,w2v_vowels,wavlm_vowels),dim=2)
            ssl_features_consonants = torch.cat((hubert_consonants,w2v_consonants,wavlm_consonants),dim=2)
 
            ssl_features_vowels = self.ccc_ssl(ssl_features_vowels)
            ssl_features_consonants = self.ccc_ssl(ssl_features_consonants)
             
            # Concatenate the outputs of the CCCs
            x_vowels = torch.cat((x_vowels,ssl_features_vowels), dim=1)
            x_consonants = torch.cat((x_consonants,ssl_features_consonants), dim=1)
           
            # Reconstruct the original structure
            x_conv[i,index_vowels[i] != -1] = x_vowels
            x_conv[i,index_consonants[i] != -1] = x_consonants
         
        # Add phoneme-level features
        dur_feat = dur_feat.unsqueeze(2)
        x_conv = torch.cat((x_conv,dur_feat,ener_feat), dim=2)
        
        # Project the features
        x_conv_shape = x_conv.shape
        x_conv = x_conv.view(-1,(self.output_dim_vc + self.output_dim_ssl + self.output_dim_duration_energy))
        x_conv = self.fusion_mlp(x_conv)
        
        # Restore the original shape
        x_conv=x_conv.view(x_conv_shape[0],x_conv_shape[1], self.fusion_dim)
        
        # Add a vowel/consonant position (bit)
        vc_emb = torch.zeros(x.shape[0], x.shape[1], 1, device="cuda")
        vc_emb[phn_vc == 0] = torch.ones(1, device="cuda")
        x_conv = torch.cat((x_conv,vc_emb),dim=2)
        
        
        # PHONEMES -------------------------------------------------------------------------------------------------
        
        # Initial declarations
        x_phn = -torch.ones(x.shape[0],x.shape[1],self.output_dim_phn,device="cuda")
        p = -torch.ones(x.shape[0],x.shape[1], 1,device="cuda")
        mdd = -torch.ones(x.shape[0],x.shape[1], 48,device="cuda")
        filler = -torch.ones(self.fusion_dim+1 ,device="cuda")
        phn_reps = []
        mdd_reps = []

        # Select TRIPHONES
        for i in range(x_conv.shape[0]):
            conv_phn=[]
            for j in range(x_conv.shape[1]):
                
                phoneme = x_conv[i,j]
                
                if phoneme.sum() == 0:
                    continue
                
                if j in index_vowels[i] or j in index_consonants[i]:
                      
                    if j == 0:
                        triphoneme = torch.stack([filler, phoneme, x_conv[i,j+1]])
                        conv_phn.append(triphoneme)
                        continue
                    
                    elif j+1 not in index_vowels[i] and j+1 not in index_consonants[i]:
                        triphoneme = torch.stack([x_conv[i,j-1], phoneme, filler])
                        conv_phn.append(triphoneme)
                        continue
                
                    else:
                        triphoneme = torch.stack([x_conv[i,j-1], phoneme, x_conv[i,j+1]])
                        conv_phn.append(triphoneme)
                        continue
                
            conv_phn = torch.stack(conv_phn)
            
            # Prepare the input
            conv_phn = conv_phn.unsqueeze(1)
            
            # Process with the phoneme CCC
            conv_phn = self.ccc_phn(conv_phn)
                     
            phn_rep = self.proj_phn(conv_phn)
            mdd_rep = self.proj_mdd(conv_phn)
            phn_reps.append(phn_rep)
            mdd_reps.append(mdd_rep)            

            p_list = (phn_rep.unsqueeze(0), mdd_rep.unsqueeze(0)) 
            p_attns = []
            for k in range(len(p_list)):
                target_p = p_list[k]
                non_target_p = torch.cat((p_list[:k] + p_list[k+1:]), dim=1)
                p_attn = self.phn_attn_tmp(target_p, non_target_p)
                p_sum = target_p + p_attn
                p_attns.append(p_sum)
           
            # Compute phoneme-level regressions
            p_predict = self.mlp_phn(p_attns[0].squeeze(0))
            mdd_predict = self.mlp_phn_mdd(p_attns[1].squeeze(0))
                        
            # Add those vectors to the phoneme tensor
            x_phn[i,:conv_phn.shape[0]] = conv_phn
                
            # Add the regressions to the regression tensor
            p[i,:p_predict.shape[0]] = p_predict
            mdd[i,:mdd_predict.shape[0]] = mdd_predict
        #----------------------------------------------------------------------------------------------------
        
        # WORDS ------------------------------------------------------------------------------------------

        # Preliminaries
        unique_indices= [torch.unique(tensor) for tensor in words]
        x_word = -torch.ones(x.shape[0],x.shape[1],self.output_dim_word,device="cuda")
        w1 = -torch.ones(x.shape[0],x.shape[1], 1,device="cuda")
        w2 = -torch.ones(x.shape[0],x.shape[1], 1,device="cuda")
        w3 =  -torch.ones(x.shape[0],x.shape[1], 1,device="cuda")
        w1_reps = []
        w2_reps = []
        w3_reps = []
        
        for i in range(x_conv.shape[0]):
            conv_words = []
            for j in unique_indices[i]:
                if j == -1:
                    continue
                
                word_phn = x_phn[i, words[i] == j]
                padded_word_phn = F.pad(word_phn, (0,0,0,12 - word_phn.shape[0]), "constant", -1)
                                
                for _ in range(len(word_phn)):
                    conv_words.append(padded_word_phn)
                   
            # Group the concatenations
            conv_words = torch.stack(conv_words)
                 
            # Prepare the input
            conv_words = conv_words.unsqueeze(1)
        
            # Process with the word CCC
            conv_words = self.ccc_word(conv_words)
            
            ### Second output word score with aspects attention
            conv_words = conv_words.unsqueeze(0)
            w1_rep = self.proj_w1(conv_words).squeeze()
            w2_rep = self.proj_w2(conv_words).squeeze()
            w3_rep = self.proj_w3(conv_words).squeeze()
            w1_reps.append(w1_rep)
            w2_reps.append(w2_rep)
            w3_reps.append(w3_rep)
              
            w_list = (w1_rep.unsqueeze(0), w2_rep.unsqueeze(0), w3_rep.unsqueeze(0))
            w_attns = []
            for k in range(3):
                 target_w = w_list[k]
                 non_target_w = torch.cat((w_list[:k] + w_list[k+1:]), dim=1)
                 w_attn = self.word_attn_tmp(target_w, non_target_w)
                 w = target_w + w_attn
                 w_attns.append(w)

            # Compute word-level regression
            w1_predict = self.mlp_word_w1(w_attns[0].squeeze())
            w2_predict  = self.mlp_word_w2(w_attns[1].squeeze())
            w3_predict  = self.mlp_word_w3(w_attns[2].squeeze())
            
            # Add those vectors to the word tensor
            conv_words = conv_words.squeeze(0)
            x_word[i,:conv_words.shape[0]] = conv_words
            
            # Extract the metric values
            w1[i,:w1_predict.shape[0]]  = w1_predict
            w2[i,:w2_predict.shape[0]]  = w2_predict
            w3[i,:w3_predict.shape[0]]  = w3_predict   
        #----------------------------------------------------------------------------------------------------
        
        # UTTERANCE ------------------------------------------------------------------------------------------
        u1 = -torch.ones(x.shape[0], 1,device="cuda")
        u2 = -torch.ones(x.shape[0], 1,device="cuda")
        u3 = -torch.ones(x.shape[0], 1,device="cuda")        
        u4 = -torch.ones(x.shape[0], 1,device="cuda")
        u5 = -torch.ones(x.shape[0], 1,device="cuda")
        
        # Prepare the input
        x_word = x_word.unsqueeze(1)

        # Process at utterance level with the CCC
        conv_utt = self.ccc_utt(x_word)
        
        # Concatenate word-level scores
        word_scores = torch.stack((w1,w2,w3),dim=2).squeeze()
        
        if word_scores.dim() == 2: 
            word_scores = word_scores.unsqueeze(0)
        
        # Expand conv_utt
        conv_utt = conv_utt.unsqueeze(1).expand(conv_utt.shape[0], p[i].size(0), -1)
                        
        for i in range(x_conv.shape[0]):
            
            # Apply the ScoreRestraint Attention Pooling
            utt_score = self.utt_score_attn(conv_utt[i], p[i], mdd[i], word_scores[i], mask=mask[i])
            
            utt_score = utt_score.unsqueeze(0)
            u1_rep = self.proj_u1(utt_score)
            u2_rep = self.proj_u2(utt_score)
            u3_rep = self.proj_u3(utt_score)
            u4_rep = self.proj_u4(utt_score)
            u5_rep = self.proj_u5(utt_score)
            
            u_list = (u1_rep, u2_rep, u3_rep, u4_rep, u5_rep)
            
            u_attns = []
            for k in range(5):
                target_u = u_list[k]
                non_target_u = torch.cat((u_list[:k] + u_list[k+1:]), dim=1)
                u_attn = self.utt_attn_tmp(target_u, non_target_u)
                u = target_u + u_attn
                u_attns.append(u)
            
        
            u1[i] = self.mlp_utt_u1(u_attns[0].squeeze(0))
            u2[i] = self.mlp_utt_u2(u_attns[1].squeeze(0))
            u3[i] = self.mlp_utt_u3(u_attns[2].squeeze(0))
            u4[i] = self.mlp_utt_u4(u_attns[3].squeeze(0))
            u5[i] = self.mlp_utt_u5(u_attns[4].squeeze(0))

        #----------------------------------------------------------------------------------------------------

                
        return u1, u2, u3, u4, u5, p, w1, w2, w3, x_phn, mdd
