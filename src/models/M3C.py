import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import json
import pandas as pd
import pickle
import torchvision as tv
import sys
from custom_layers.attention import Aspect_Attention_op2
from custom_layers.scoreRestraintAttentionPooling import ScoreRestraintAttentionPooling

# MI MODIFICACION
#----------------------------------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

    
class Net(nn.Module):
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

 
class GOPT_VC(nn.Module):
    def __init__(self,
                 num_convs_vc, input_dim_vowels, input_dim_consonants, output_dim_vc, dropout_cnn_vc, dropout_mlp_vc, # Parametros extracción de características V/C
                 num_convs_ssl, input_dim_ssl, output_dim_ssl, dropout_cnn_ssl, dropout_mlp_ssl,  # Parametros extracción de características SSL
                 fusion_dim, dropout_mlp_fusion, # Parametros de fusion SSL y V/C
                 num_convs_phn, output_dim_phn, dropout_cnn_phn, dropout_mlp_phn, # Parametros a nivel de fonema
                 num_convs_word, output_dim_word, dropout_cnn_word, dropout_mlp_word, # Parametros a nivel de palabra
                 num_convs_utt, output_dim_utt, dropout_cnn_utt, dropout_mlp_utt, # Parametros a nivel de frase
                 ):
        super().__init__()
        
        self.output_dim_duration_energy = 8
        self.output_dim_vc = output_dim_vc
        self.output_dim_ssl = output_dim_ssl
        self.fusion_dim = fusion_dim
        self.output_dim_phn = output_dim_phn
        self.output_dim_word = output_dim_word
        self.output_dim_utt = output_dim_utt
        
               
        # NIVEL DE EXTRACCIÓN DE CARACTERISTICAS
        
        # GOP
        self.conv_vowels= Net(input_dim=input_dim_vowels, num_convs=num_convs_vc, output_dim=output_dim_vc,dropout_cnn=dropout_cnn_vc, dropout_mlp=dropout_mlp_vc)
        self.conv_consonants= Net(input_dim=input_dim_consonants, num_convs=num_convs_vc, output_dim=output_dim_vc,dropout_cnn=dropout_cnn_vc, dropout_mlp=dropout_mlp_vc)
    
        # SSL
        self.conv_ssl= Net(input_dim=input_dim_ssl, num_convs=num_convs_ssl, output_dim=output_dim_ssl ,dropout_cnn=dropout_cnn_ssl, dropout_mlp=dropout_mlp_ssl)
        
        # Fusion caraterísticas
        self.fusion_mlp = nn.Sequential(nn.Linear(output_dim_vc+output_dim_ssl+self.output_dim_duration_energy,fusion_dim),nn.LayerNorm(fusion_dim),nn.Dropout(dropout_mlp_fusion))
        
        #--------------------------------------------------------------------------------------------------------------------
                
        self.proj_phn = nn.Linear(output_dim_phn+1, output_dim_phn)
        self.proj_mdd = nn.Linear(output_dim_phn+1, output_dim_phn)
        self.phn_attn_tmp = Aspect_Attention_op2(output_dim_phn)
     
        self.mlp_phn = nn.Sequential(nn.Linear(output_dim_phn,1))
        self.mlp_phn_mdd = nn.Sequential(nn.Linear(output_dim_phn, 48))
        
        #--------------------------------------------------------------------------------------------------------------------
        
        # Nivel de PALABRA
        self.conv_word= Net(input_dim=output_dim_phn+1, num_convs=num_convs_word, output_dim=output_dim_word, dropout_cnn=0., dropout_mlp=dropout_mlp_word, kernel_size=(12,1))
                
        self.proj_w1 = nn.Sequential(nn.Linear(output_dim_word, output_dim_word), nn.Dropout(dropout_cnn_word))
        self.proj_w2 = nn.Sequential(nn.Linear(output_dim_word, output_dim_word), nn.Dropout(dropout_cnn_word))
        self.proj_w3 = nn.Sequential(nn.Linear(output_dim_word, output_dim_word), nn.Dropout(dropout_cnn_word))
        self.word_attn_tmp = Aspect_Attention_op2(output_dim_word)
    
        self.mlp_word_w1 = nn.Sequential(nn.Linear(output_dim_word,1))
        self.mlp_word_w2 = nn.Sequential(nn.Linear(output_dim_word,1))
        self.mlp_word_w3 = nn.Sequential(nn.Linear(output_dim_word,1))
        
        #--------------------------------------------------------------------------------------------------------------------
        
        # Nivel de FRASE
        self.conv_utt= Net(input_dim=output_dim_utt, num_convs=num_convs_utt, output_dim=output_dim_utt, dropout_cnn=dropout_cnn_utt, dropout_mlp=dropout_mlp_utt, kernel_size=(50,1))
        
        self.proj_u1 = nn.Linear(output_dim_utt, output_dim_utt)
        self.proj_u2 = nn.Linear(output_dim_utt, output_dim_utt)
        self.proj_u3 = nn.Linear(output_dim_utt, output_dim_utt)
        self.proj_u4 = nn.Linear(output_dim_utt, output_dim_utt)
        self.proj_u5 = nn.Linear(output_dim_utt, output_dim_utt)
        self.utt_attn_tmp = Aspect_Attention_op2(output_dim_utt)
        
        self.utt_score_attn = ScoreRestraintAttentionPooling(num_scores=52, hidden_dim=output_dim_utt)

        self.mlp_utt_u1 = nn.Sequential(nn.Linear(output_dim_utt,1))
        self.mlp_utt_u2 = nn.Sequential(nn.Linear(output_dim_utt,1), nn.Sigmoid())
        self.mlp_utt_u3 = nn.Sequential(nn.Linear(output_dim_utt,1))
        self.mlp_utt_u4 = nn.Sequential(nn.Linear(output_dim_utt,1))
        self.mlp_utt_u5 = nn.Sequential(nn.Linear(output_dim_utt,1))
        
        #--------------------------------------------------------------------------------------------------------------------

    def forward(self, x, phn, words, dur_feat, ener_feat,w2v_feat,hubert_feat,wavlm_feat):
        
        
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
        with open('arpabet_to_vc.json', 'r') as file:
            arpa_vc = json.load(file)
              
        with open('pureLabel_to_(0-41).json', 'r') as file:
            arpa_long = json.load(file)
             
        with open('(3-41)_to_(0,numPhonesUsed).json', 'r') as file:
            long_short = json.load(file)
            
        long_vc = {int(value): arpa_vc.get(key, key) for key, value in arpa_long.items()}
        del long_vc[0]
        del long_vc[1]
        del long_vc[2]
        short_vc ={long_short[str(key)]: value for key, value in long_vc.items()}
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
        
        # DISTINCION ENTRE VOCALES Y CONSONANTES DEL phn
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
               
        # Quitamos el SIM_EMB
        x = x[:,:,[0,1,2],:]
            
        # Creamos una mascara para quitar las posiciones que corresponden a -1s
        mask = (phn != -1)
        
        # Obtener los índices donde phn_vc == 1
        index_consonants = torch.where(phn_vc == 1, torch.arange(50, device="cuda").expand(x.shape[0], 50), -1)
        
        # Obtener los índices donde phn_vc == 0
        index_vowels = torch.where(phn_vc == 0, torch.arange(50, device="cuda").expand(x.shape[0], 50), -1)
        
        # Obtener los índices donde phn_vc == -1
        index_no_phn = torch.where(phn_vc == -1, torch.arange(50, device="cuda").expand(x.shape[0], 50), -1)
         
        # Declaramos la nueva representación procesada
        x_conv = -torch.ones(x.shape[0],x.shape[1],(self.output_dim_vc + self.output_dim_ssl),device="cuda")
                     
        # Procesamos por la CNN, tanto vocales como consonantes
        for i in range(x.shape[0]):
            
            # GOP
            x_vowels = x[i,index_vowels[i] != -1,:,:15]
            x_consonants = x[i,index_consonants[i] != -1]
        
            x_vowels = x_vowels.unsqueeze(1)
            x_consonants = x_consonants.unsqueeze(1)
            
            x_vowels = self.conv_vowels(x_vowels)
            x_consonants = self.conv_consonants(x_consonants)
            
            # SSL
            # Hubert
            hubert_vowels = hubert_feat[i,index_vowels[i] != -1]
            hubert_consonants = hubert_feat[i,index_consonants[i] != -1]
            hubert_vowels = hubert_vowels.unsqueeze(1).unsqueeze(1)
            hubert_consonants = hubert_consonants.unsqueeze(1).unsqueeze(1)
            
            # W2V
            w2v_vowels = w2v_feat[i,index_vowels[i] != -1]
            w2v_consonants = w2v_feat[i,index_consonants[i] != -1]
            w2v_vowels = w2v_vowels.unsqueeze(1).unsqueeze(1)
            w2v_consonants = w2v_consonants.unsqueeze(1).unsqueeze(1)
            
            # WAVLM
            wavlm_vowels = wavlm_feat[i,index_vowels[i] != -1]
            wavlm_consonants = wavlm_feat[i,index_consonants[i] != -1]
            wavlm_vowels = wavlm_vowels.unsqueeze(1).unsqueeze(1)
            wavlm_consonants = wavlm_consonants.unsqueeze(1).unsqueeze(1)
            
            ssl_features_vowels = torch.cat((hubert_vowels,w2v_vowels,wavlm_vowels),dim=2)
            ssl_features_consonants = torch.cat((hubert_consonants,w2v_consonants,wavlm_consonants),dim=2)
 
            ssl_features_vowels = self.conv_ssl(ssl_features_vowels)
            ssl_features_consonants = self.conv_ssl(ssl_features_consonants)
             
            # Concatenamos las salidad de las CNNs
            x_vowels = torch.cat((x_vowels,ssl_features_vowels), dim=1)
            x_consonants = torch.cat((x_consonants,ssl_features_consonants), dim=1)
           
            # Reconstruimos la estructura original
            x_conv[i,index_vowels[i] != -1] = x_vowels
            x_conv[i,index_consonants[i] != -1] = x_consonants
         
        # Añadimos características a nivel de fonema   
        dur_feat = dur_feat.unsqueeze(2)
        x_conv = torch.cat((x_conv,dur_feat,ener_feat), dim=2)
        
        # Proyectamos las características
        x_conv_shape = x_conv.shape
        x_conv = x_conv.view(-1,(self.output_dim_vc + self.output_dim_ssl + self.output_dim_duration_energy))
        x_conv = self.fusion_mlp(x_conv)
        
        # Reconstruimos la forma original          
        x_conv=x_conv.view(x_conv_shape[0],x_conv_shape[1], self.fusion_dim)
        
        # Añadimos un embeding vocal / consonante
        vc_emb = torch.zeros(x.shape[0], x.shape[1], 1, device="cuda")
        vc_emb[phn_vc == 0] = torch.ones(1, device="cuda")
        x_conv = torch.cat((x_conv,vc_emb),dim=2)
        
    
        # FONEMAS -------------------------------------------------------------------------------------------------
        
# FONEMAS -------------------------------------------------------------------------------------------------
        
        # Declaraciones iniciales
        x_phn = -torch.ones(x.shape[0],x.shape[1],self.output_dim_phn+1,device="cuda")
        p = -torch.ones(x.shape[0],x.shape[1], 1,device="cuda")
        mdd = -torch.ones(x.shape[0],x.shape[1], 48,device="cuda")
        phn_reps = []
        mdd_reps = []

        # Realizamos la selección de TRIFONEMAS
        for i in range(x_conv.shape[0]):
                     
            phn_rep = self.proj_phn(x_conv[i])
            mdd_rep = self.proj_mdd(x_conv[i])
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
           
            # Calculamos las regresiones a nivel de fonema
            p_predict = self.mlp_phn(p_attns[0].squeeze(0))
            mdd_predict = self.mlp_phn_mdd(p_attns[1].squeeze(0))
            
             
            # Añadimos esos vectores al tensor de fonemas
            x_phn[i,:x_conv.shape[1]] = x_conv[i]
                
            # Añadimos las regresiones al tensor de regresiones
            p[i,:p_predict.shape[0]] = p_predict
            mdd[i,:mdd_predict.shape[0]] = mdd_predict
        #----------------------------------------------------------------------------------------------------
        
        # PALABRAS ------------------------------------------------------------------------------------------

        # Declaraciones previas
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
                   
            # Agrupamos las concatenaciones
            conv_words = torch.stack(conv_words)
                 
            # Preparamos la entrada
            conv_words = conv_words.unsqueeze(1)
        
            # Procesamos con la CNN de palabras
            conv_words = self.conv_word(conv_words)
            
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

            # Calculamos la regresion a nivel de palabra
            w1_predict = self.mlp_word_w1(w_attns[0].squeeze())
            w2_predict  = self.mlp_word_w2(w_attns[1].squeeze())
            w3_predict  = self.mlp_word_w3(w_attns[2].squeeze())
            
            # Añadimos esos vectores al tensor de palabras
            conv_words = conv_words.squeeze(0)
            x_word[i,:conv_words.shape[0]] = conv_words
            
            # Extraemos los valores de las métricas
            w1[i,:w1_predict.shape[0]]  = w1_predict
            w2[i,:w2_predict.shape[0]]  = w2_predict
            w3[i,:w3_predict.shape[0]]  = w3_predict   
        #----------------------------------------------------------------------------------------------------
        
        # FRASE ------------------------------------------------------------------------------------------
        u1 = -torch.ones(x.shape[0], 1,device="cuda")
        u2 = -torch.ones(x.shape[0], 1,device="cuda")
        u3 = -torch.ones(x.shape[0], 1,device="cuda")        
        u4 = -torch.ones(x.shape[0], 1,device="cuda")
        u5 = -torch.ones(x.shape[0], 1,device="cuda")
        
        # Preparamos la entrada
        x_word = x_word.unsqueeze(1)

        # Procesamos a nivel de frase con la CNN
        conv_utt = self.conv_utt(x_word)
        
        # Concatenamos las valoraciones a nivel de palabra
        word_scores = torch.stack((w1,w2,w3),dim=2).squeeze()
        
        if word_scores.dim() == 2: 
            word_scores = word_scores.unsqueeze(0)
        
        # Ampliamos conv_utt
        conv_utt = conv_utt.unsqueeze(1).expand(conv_utt.shape[0], p[i].size(0), -1)
                        
        for i in range(x_conv.shape[0]):
            
           
            # Aplicamos el ScoreRestraint...
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

