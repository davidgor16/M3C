# -*- coding: utf-8 -*-
# @Time    : 9/20/21 12:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# train and test the models
import sys
import os, shutil
import time
from torch.utils.data import Dataset, DataLoader
from models.M3C import M3C

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from models import *
import argparse

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# GENERAL PARAMETERS
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n-epochs", type=int, default=3, help="number of maximum training epochs")
parser.add_argument("--batch_size", type=int, default=25, help="training batch size")
parser.add_argument("--embed_dim", type=int, default=12, help="M3C transformer embedding dimension")
parser.add_argument("--loss_w_phn", type=float, default=1, help="weight for phoneme-level loss")
parser.add_argument("--loss_w_word", type=float, default=1, help="weight for word-level loss")
parser.add_argument("--loss_w_utt", type=float, default=1, help="weight for utterance-level loss")
parser.add_argument("--model", type=str, default='gopt', help="name of the model")
parser.add_argument("--am", type=str, default='librispeech', help="name of the acoustic models")
parser.add_argument("--noise", type=float, default=0., help="the scale of random noise added on the input GoP feature")
parser.add_argument("--alpha_mdd", type=float, default=0.3, help="weight for MDD loss")

# V/C FEATURE EXTRACTION PARAMETERS
parser.add_argument("--num_convs_vc", type=int, default=32, help="Number of convolutional kernels of the vowel and consonant feature extractor CNN")
parser.add_argument("--input_dim_vowels", type=int, default=6, help="Dimension of the input for the Vowel CCC")
parser.add_argument("--input_dim_consonants", type=int, default=6, help="Dimension of the input for the Consonant CCC")
parser.add_argument("--output_dim_vc", type=int, default=6, help="Dimension of the output for the Consonant/Vowel CCC")
parser.add_argument("--dropout_cnn_vc", type=float, default=0., help="Probability of dropout for Vowel/Consonant CCC output")
parser.add_argument("--dropout_mlp_vc", type=float, default=0., help="Probability of dropout for Vowel/Consonant  MLP output")

# SSL FEATURE EXTRACTION PARAMETERS
parser.add_argument("--num_convs_ssl", type=int, default=32, help="Number of convolutional kernels of the SSL feature extractor CCC")
parser.add_argument("--input_dim_ssl", type=int, default=6, help="Dimension of the input for the SSL CCC")
parser.add_argument("--output_dim_ssl", type=int, default=6, help="Dimension of the output for the SSL CCC")
parser.add_argument("--dropout_cnn_ssl", type=float, default=0., help="Probability of dropout for SSL CCC output")
parser.add_argument("--dropout_mlp_ssl", type=float, default=0., help="Probability of dropout for SSL MLP output")

# FEATURE FUSION PARAMETERS
parser.add_argument("--fusion_dim", type=int, default=10, help="Dimension of the output of the MLP fusion of SSL and V/C features")
parser.add_argument("--dropout_mlp_fusion", type=float, default=0., help="Probability of dropout for Fusion MLP output")

# PHONEME-LEVEL PARAMETERS
parser.add_argument("--num_convs_phn", type=int, default=32, help="Number of convolutional kernels of the Phoneme feature extractor CCC")
parser.add_argument("--output_dim_phn", type=int, default=6, help="Dimension of the output for the Phoneme CCC")
parser.add_argument("--dropout_cnn_phn", type=float, default=0., help="Probability of dropout for Phoneme CCC output")
parser.add_argument("--dropout_mlp_phn", type=float, default=0., help="Probability of dropout for Phoneme MLP output")

# WORD-LEVEL PARAMETERS
parser.add_argument("--num_convs_word", type=int, default=32, help="Number of convolutional kernels of the Word feature extractor CCC")
parser.add_argument("--output_dim_word", type=int, default=6, help="Dimension of the output for the Word CCC")
parser.add_argument("--dropout_cnn_word", type=float, default=0., help="Probability of dropout for Word CCC output")
parser.add_argument("--dropout_mlp_word", type=float, default=0., help="Probability of dropout for Word MLP output")

# UTTERANCE-LEVEL PARAMETERS
parser.add_argument("--num_convs_utt", type=int, default=32, help="Number of convolutional kernels of the Utterance feature extractor CCC")
parser.add_argument("--output_dim_utt", type=int, default=6, help="Dimension of the output for the Utterance CCC")
parser.add_argument("--dropout_cnn_utt", type=float, default=0., help="Probability of dropout for Utterance CCC output")
parser.add_argument("--dropout_mlp_utt", type=float, default=0., help="Probability of dropout for Utterance MLP output")

# just to generate the header for the result.csv
def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    distance_header_score = ['mean_dis', 'sum_dis']
    mdd_header_score = ['f1_mdd', 'precision_mdd', 'recall_mdd']
    
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    header = phn_header + utt_header + word_header + distance_header_score + mdd_header_score
    return header

def train(audio_model, train_loader_apa, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    print(f"The value of alpha_mdd is: {args.alpha_mdd}")

    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in audio_model.parameters()) / 1e3))
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)

    loss_fn = nn.MSELoss()
    loss_fn_mdd = nn.CrossEntropyLoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([1, 37])

    while epoch < args.n_epochs:
                
        train_loader = train_loader_apa
            
        print('Epoch: {}'.format(epoch))
        print('Number of batches: {}'.format(len(train_loader)))
            
        audio_model.train()
        
        for i, (audio_input, phn_label, dur_feat, ener_feat, w2v_feat, hubert_feat, wavlm_feat, cano_phns, real_phns, utt_label, word_label) in enumerate(train_loader):
                    
            wavlm_feat = wavlm_feat.to(device, non_blocking=True)
            hubert_feat = hubert_feat.to(device, non_blocking=True)
            w2v_feat = w2v_feat.to(device, non_blocking=True)
            ener_feat = ener_feat.to(device, non_blocking=True)
            dur_feat = dur_feat.to(device, non_blocking=True)
            audio_input = audio_input.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)
                        
            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))
    
            words = word_label[:,:,3]
            u1, u2, u3, u4, u5, p, w1, w2, w3, x_phn, mdd = audio_model(audio_input, cano_phns, words, dur_feat, ener_feat, w2v_feat, hubert_feat, wavlm_feat)
             
            # APA LOSS--------------------------------------------------------------------------------------------------------------------------
            
            # filter out the padded tokens, only calculate the loss based on the valid tokens
            # < 0 is a flag of padded tokens
            # phone level loss, mse
            mask_phn = (phn_label>=0)
            p = p.squeeze(2)
            p = p * mask_phn 
            phn_label = phn_label * mask_phn
            loss_phn = loss_fn(p, phn_label)

            # avoid the 0 losses of the padded tokens impacting the performance
            loss_phn = loss_phn * (mask_phn.shape[0] * mask_phn.shape[1]) / torch.sum(mask_phn)

            # utterance level loss, also mse
            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            loss_utt = loss_fn(utt_preds ,utt_label)
                
            # word level loss
            word_label = word_label[:, :, 0:3]
            mask = (word_label>=0)
            word_pred = torch.cat((w1,w2,w3), dim=2)
            word_pred = word_pred * mask
            word_label = word_label * mask
            loss_word = loss_fn(word_pred, word_label)
            loss_word = loss_word * (mask.shape[0] * mask.shape[1] * mask.shape[2]) / torch.sum(mask)

            loss_apa =  args.loss_w_phn * loss_phn + args.loss_w_utt * loss_utt + args.loss_w_word * loss_word
    
            #------------------------------------------------------------------------------------------------------------------------------------
               
            # MDD Loss--------------------------------------------------------------------------------------------------------------------------
            mask = real_phns != -1
            real_phns = real_phns[mask].to(device)
            mdd = mdd[mask]
                     
            # We compute the MDD loss
            loss_mdd = loss_fn_mdd(mdd, real_phns.long())
                
            #------------------------------------------------------------------------------------------------------------------------------------
               
            # Final loss
            loss = loss_apa + args.alpha_mdd * loss_mdd 

            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            global_step += 1
                

        print('start validation')
        
        # ensemble results
        # don't save prediction for the training set
        tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr, tr_mean_dis, tr_sum_dis, tr_precision, tr_recall, tr_f1_score = validate(audio_model, train_loader, args, -1)
        te_mse, te_corr, te_utt_mse, te_utt_corr, te_word_mse, te_word_corr, te_mean_dis, te_sum_dis, te_precision, te_recall, te_f1_score = validate(audio_model, test_loader, args, best_mse)

        print('Phone: Test MSE: {:.3f}, CORR: {:.3f}'.format(te_mse.item(), te_corr))
        print('Utterance:, ACC: {:.3f}, COM: {:.3f}, FLU: {:.3f}, PROC: {:.3f}, Total: {:.3f}'.format(te_utt_corr[0], te_utt_corr[1], te_utt_corr[2], te_utt_corr[3], te_utt_corr[4]))
        print('Word:, ACC: {:.3f}, Stress: {:.3f}, Total: {:.3f}'.format(te_word_corr[0], te_word_corr[1], te_word_corr[2]))
        print('Mean Distance: {:.3f}, Sum Distance: {:.3f}'.format(te_mean_dis, te_sum_dis))
        print('Precision: {:.3f}, Recall: {:.3f}, F1 Score: {:.3f}'.format(te_precision, te_recall, te_f1_score))
        sys.stdout.flush()

        print('-------------------validation finished-------------------')

        if te_mse < best_mse:
            best_mse = te_mse
            best_epoch = epoch
             
            # Save results
            result[0, :6] = [epoch, tr_mse, tr_corr, te_mse, te_corr, optimizer.param_groups[0]['lr']]

            result[0, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, te_utt_mse, te_utt_corr])

            result[0, 26:32] = np.concatenate([tr_word_corr, te_word_corr])
                
            result[0,32:37] = [te_mean_dis.item(), te_sum_dis.item(), te_f1_score, te_precision, te_recall]

            header = ','.join(gen_result_header())
            np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')

        if best_epoch == epoch:
            if os.path.exists("%s/models/" % (exp_dir)) == False:
                os.mkdir("%s/models" % (exp_dir))
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

        if global_step > warm_up_step:
            scheduler.step()
        
        epoch += 1
          
def validate(audio_model, val_loader, args, best_mse):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    centroids = {}
    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []
    with torch.no_grad():
        for i, (audio_input, phn_label, dur_feat, ener_feat, w2v_feat, hubert_feat, wavlm_feat, cano_phns, real_phns, utt_label, word_label) in enumerate(val_loader):
            
            wavlm_feat = wavlm_feat.to(device)
            hubert_feat = hubert_feat.to(device)
            w2v_feat = w2v_feat.to(device)
            audio_input = audio_input.to(device)
            dur_feat = dur_feat.to(device)
            ener_feat = ener_feat.to(device)
            
            # compute output
            words = word_label[:,:,3]
            u1, u2, u3, u4, u5, p, w1, w2, w3, x_phn, mdd = audio_model(audio_input, cano_phns, words, dur_feat, ener_feat,w2v_feat, hubert_feat, wavlm_feat)
            
            if best_mse != -1:
                
                mask = cano_phns != -1
                cano_phns = cano_phns[mask]
                real_phns = real_phns[mask]
                x_phn = x_phn[mask]
                mdd = mdd[mask]
                
                # DISTANCES -------------------------------------------------------------------------------------------------
    
                unique_classes = torch.unique(cano_phns)

                for cls in unique_classes:
                    mask = (cano_phns == cls)  
                    vectors_cls = x_phn[mask]  
                    
                    if cls.item() not in centroids:
                        centroids[cls.item()] = []  
        
                    # Append the new vectors (as lists) to the existing ones
                    centroids[cls.item()].extend(vectors_cls.tolist())
                
                # Compute the centroid for each class
                for cls in centroids.keys():
                    if len(centroids[cls]) > 0:
                        centroids[cls] = torch.tensor(centroids[cls]).mean(dim=0)
                
                # Compute distances between each pair of centroids
                num_classes = len(centroids.keys())
                
                centroids_list = [centroids[cls] for cls in sorted(centroids.keys())]  # Sort by key

                centroids_tensor = torch.stack(centroids_list)
                distances = torch.cdist(centroids_tensor, centroids_tensor, p=2)  # L2 norm (Euclidean)
                
                # Mask the diagonal (distance of a centroid to itself = 0)
                mask = ~torch.eye(num_classes, dtype=torch.bool, device=centroids_tensor.device)
                valid_distances = distances[mask]  # Only distances between different centroids
                        
                # Mean distance between centroids
                mean_dis = valid_distances.mean()
                
                # Total (sum) distance between centroids
                sum_dis = valid_distances.sum()
                
                # MDD ------------------------------------------------------------------------------------------------
                # Compute precision, recall, and F1-score
                from sklearn.metrics import precision_recall_fscore_support
                mdd = mdd.cpu().numpy()
                mdd = np.argmax(mdd, axis=1)  # Convert to class labels
                real_phns = real_phns.cpu().numpy()
                precision, recall, f1_score, _ = precision_recall_fscore_support(real_phns, mdd, average='macro')
                # ------------------------------------------------------------------------------------------------------------
                
            else:
                mean_dis = sum_dis = 0
                precision = recall = f1_score = 0
            # ------------------------------------------------------------------------------------------------------------
            
            
            p = p.to('cpu').detach()
            u1, u2, u3, u4, u5 = u1.to('cpu').detach(), u2.to('cpu').detach(), u3.to('cpu').detach(), u4.to('cpu').detach(), u5.to('cpu').detach()
            w1, w2, w3 = w1.to('cpu').detach(), w2.to('cpu').detach(), w3.to('cpu').detach()

            A_phn.append(p)
            A_phn_target.append(phn_label)

            A_u1.append(u1)
            A_u2.append(u2)
            A_u3.append(u3)
            A_u4.append(u4)
            A_u5.append(u5)
            A_utt_target.append(utt_label)

            A_w1.append(w1)
            A_w2.append(w2)
            A_w3.append(w3)
            A_word_target.append(word_label)

        # phone level
        A_phn, A_phn_target  = torch.cat(A_phn), torch.cat(A_phn_target)

        # utterance level
        A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = torch.cat(A_u1), torch.cat(A_u2), torch.cat(A_u3), torch.cat(A_u4), torch.cat(A_u5), torch.cat(A_utt_target)

        # word level
        A_w1, A_w2, A_w3, A_word_target = torch.cat(A_w1), torch.cat(A_w2), torch.cat(A_w3), torch.cat(A_word_target)

        # get the scores
        phn_mse, phn_corr = valid_phn(A_phn, A_phn_target)

        A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
        utt_mse, utt_corr = valid_utt(A_utt, A_utt_target)

        A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
        word_mse, word_corr, valid_word_pred, valid_word_target = valid_word(A_word, A_word_target)

        if phn_mse < best_mse:
            print('new best phn mse {:.3f}, now saving predictions.'.format(phn_mse))

            # create the directory
            if os.path.exists(args.exp_dir + '/preds') == False:
                os.mkdir(args.exp_dir + '/preds')

            # saving the phn target, only do once
            if os.path.exists(args.exp_dir + '/preds/phn_target.npy') == False:
                np.save(args.exp_dir + '/preds/phn_target.npy', A_phn_target)
                np.save(args.exp_dir + '/preds/word_target.npy', valid_word_target)
                np.save(args.exp_dir + '/preds/utt_target.npy', A_utt_target)

            np.save(args.exp_dir + '/preds/phn_pred.npy', A_phn)
            np.save(args.exp_dir + '/preds/word_pred.npy', valid_word_pred)
            np.save(args.exp_dir + '/preds/utt_pred.npy', A_utt)

    return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr, mean_dis, sum_dis, precision, recall, f1_score

def valid_phn(audio_output, target):
    valid_token_pred = []
    valid_token_target = []
    audio_output = audio_output.squeeze(2)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            # only count valid tokens, not padded tokens (represented by negative values)
            if target[i, j] >= 0:
                valid_token_pred.append(audio_output[i, j])
                valid_token_target.append(target[i, j])
    valid_token_target = np.array(valid_token_target)
    valid_token_pred = np.array(valid_token_pred)

    valid_token_mse = np.mean((valid_token_target - valid_token_pred) ** 2)
    corr = np.corrcoef(valid_token_pred, valid_token_target)[0, 1]
    return valid_token_mse, corr

def valid_utt(audio_output, target):
    mse = []
    corr = []
    for i in range(5):
        cur_mse = np.mean(((audio_output[:, i] - target[:, i]) ** 2).numpy())
        cur_corr = np.corrcoef(audio_output[:, i], target[:, i])[0, 1]
        mse.append(cur_mse)
        corr.append(cur_corr)
    return mse, corr

def valid_word(audio_output, target):
    word_id = target[:, :, -1]
    target = target[:, :, 0:3]

    valid_token_pred = []
    valid_token_target = []

    # unique, counts = np.unique(np.array(target), return_counts=True)
    # print(dict(zip(unique, counts)))

    # for each utterance
    for i in range(target.shape[0]):
        prev_w_id = 0
        start_id = 0
        # for each token
        for j in range(target.shape[1]):
            cur_w_id = word_id[i, j].int()
            # if a new word
            if cur_w_id != prev_w_id:
                # average each phone belongs to the word
                valid_token_pred.append(np.mean(audio_output[i, start_id: j, :].numpy(), axis=0))
                valid_token_target.append(np.mean(target[i, start_id: j, :].numpy(), axis=0))
                # sanity check, if the range indeed contains a single word
                if len(torch.unique(target[i, start_id: j, 1])) != 1:
                    print(target[i, start_id: j, 0])
                # if end of the utterance
                if cur_w_id == -1:
                    break
                else:
                    prev_w_id = cur_w_id
                    start_id = j

    valid_token_pred = np.array(valid_token_pred)
    # this rounding is to solve the precision issue in the label
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list, corr_list = [], []
    # for each (accuracy, stress, total) word score
    for i in range(3):
        valid_token_mse = np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2)
        corr = np.corrcoef(valid_token_pred[:, i], valid_token_target[:, i])[0, 1]
        mse_list.append(valid_token_mse)
        corr_list.append(corr)
    return mse_list, corr_list, valid_token_pred, valid_token_target

class GoPDataset(Dataset):
    def __init__(self, set, am='librispeech'):
        
        # normalize the input to 0 mean and unit std.
        print(am)
         
        if am=="features_vc_clean_embbedings_norm":
            dir='GoP_VC_Features_Embbeding_norm'
                    
        else:
            raise ValueError('Acoustic Model Unrecognized.')

        if set == 'train':
            if "clean" in am:
                print("CLEAN DATA")
                self.feat = torch.tensor(np.load('../data/'+dir+'/tr_clean_feat.npy'), dtype=torch.float)
            else:
                self.feat = torch.tensor(np.load('../data/'+dir+'/tr_feat.npy'), dtype=torch.float)
            
            self.wavlm_feat = torch.tensor(np.load('../data/'+dir+'/tr_wavlm_features.npy'), dtype=torch.float)
            self.hubert_feat = torch.tensor(np.load('../data/'+dir+'/tr_hub_features.npy'), dtype=torch.float)
            self.w2v_feat = torch.tensor(np.load('../data/'+dir+'/tr_w2v_features.npy'), dtype=torch.float)
            self.ener_feat = torch.tensor(np.load('../data/'+dir+'/tr_energy_features.npy'), dtype=torch.float)
            self.dur_feat = torch.tensor(np.load('../data/'+dir+'/tr_duration_features.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/librispeech_cesar/tr_label_phn.npy'), dtype=torch.float)
            self.real_phn_label = torch.tensor(np.load('../data/librispeech_cesar/tr_real_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/librispeech_cesar/tr_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/librispeech_cesar/tr_label_word.npy'), dtype=torch.float)
            
        elif set == 'test':
            if "clean" in am:
                print("CLEAN DATA")
                self.feat = torch.tensor(np.load('../data/'+dir+'/te_clean_feat.npy'), dtype=torch.float)
            else:
                self.feat = torch.tensor(np.load('../data/'+dir+'/te_feat.npy'), dtype=torch.float)
                
            self.wavlm_feat = torch.tensor(np.load('../data/'+dir+'/te_wavlm_features.npy'), dtype=torch.float)
            self.hubert_feat = torch.tensor(np.load('../data/'+dir+'/te_hub_features.npy'), dtype=torch.float)
            self.w2v_feat = torch.tensor(np.load('../data/'+dir+'/te_w2v_features.npy'), dtype=torch.float)
            self.ener_feat = torch.tensor(np.load('../data/'+dir+'/te_energy_features.npy'), dtype=torch.float)
            self.dur_feat = torch.tensor(np.load('../data/'+dir+'/te_duration_features.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/librispeech_cesar/te_label_phn.npy'), dtype=torch.float)
            self.real_phn_label = torch.tensor(np.load('../data/librispeech_cesar/te_real_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/librispeech_cesar/te_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/librispeech_cesar/te_label_word.npy'), dtype=torch.float)

        # No normalization is needed because our inputs are already normalized.
        #self.feat = self.norm_valid(self.feat, norm_mean, norm_std)
       
        # normalize the utt_label to 0-2 (same with phn score range)
        self.utt_label = self.utt_label / 5
        # the last dim is word_id, so not normalizing
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5
        self.phn_label[:, :, 1] = self.phn_label[:, :, 1]
        
    
    # only normalize valid tokens, not padded token
    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        # feat, phn_label, phn_id, utt_label, word_label
        return self.feat[idx, :], self.phn_label[idx, :, 1], self.dur_feat[idx,:], self.ener_feat[idx,:], self.w2v_feat[idx,:], self.hubert_feat[idx,:], self.wavlm_feat[idx,:], self.phn_label[idx, :, 0],  self.real_phn_label[idx, :], self.utt_label[idx, :], self.word_label[idx, :]

args = parser.parse_args()

am = args.am
print('now train with {:s} acoustic models'.format(am))
feat_dim = {"features_vc_clean_embbedings_norm":24}

input_dim=feat_dim[am]


if args.model == 'm3c':
    print('now train a M3C model')
    audio_mdl = M3C(num_convs_vc = args.num_convs_vc, input_dim_vowels = args.input_dim_vowels, input_dim_consonants = args.input_dim_consonants, output_dim_vc = args.output_dim_vc, dropout_cnn_vc= args.dropout_cnn_vc, dropout_mlp_vc = args.dropout_mlp_vc,
                        num_convs_ssl = args.num_convs_ssl, input_dim_ssl = args.input_dim_ssl, output_dim_ssl = args.output_dim_ssl, dropout_cnn_ssl= args.dropout_cnn_ssl, dropout_mlp_ssl = args.dropout_mlp_ssl,
                        fusion_dim=args.fusion_dim, dropout_mlp_fusion=args.dropout_mlp_fusion,
                        num_convs_phn=args.num_convs_phn, output_dim_phn=args.output_dim_phn, dropout_cnn_phn=args.dropout_cnn_phn, dropout_mlp_phn=args.dropout_cnn_phn, # Phoneme-level parameters
                        num_convs_word=args.num_convs_word, output_dim_word=args.output_dim_word, dropout_cnn_word=args.dropout_cnn_word, dropout_mlp_word=args.dropout_cnn_word, # Word-level parameters
                        num_convs_utt=args.num_convs_utt, output_dim_utt=args.output_dim_utt, dropout_cnn_utt=args.dropout_cnn_utt, dropout_mlp_utt=args.dropout_cnn_utt, # Utterance-level parameters
) 
    print()
    print("V/C FEATURE EXTRACTION PARAMETERS")
    print(f"Number of convolutions: {args.num_convs_vc}")
    print(f"Input dim vowels: {args.input_dim_vowels}")
    print(f"Input dim consonants: {args.input_dim_consonants}")
    print(f"Output dim: {args.output_dim_vc}")
    print(f"Dropout CNN: {args.dropout_cnn_vc}")
    print(f"Dropout MLP: {args.dropout_mlp_vc}")
    print()
    print()
    
    print("SSL FEATURE EXTRACTION PARAMETERS")
    print(f"Number of convolutions: {args.num_convs_ssl}")
    print(f"Input dim: {args.input_dim_ssl}")
    print(f"Output dim: {args.output_dim_ssl}")
    print(f"Dropout CNN: {args.dropout_cnn_ssl}")
    print(f"Dropout MLP: {args.dropout_mlp_ssl}")
    print()
    print()
    
    print("FEATURE FUSION PARAMETERS")
    print(f"Fusion dim: {args.fusion_dim}")
    print(f"Dropout MLP: {args.dropout_mlp_fusion}")
    print()
    print()
    
    print("PHONEME-LEVEL PARAMETERS")
    print(f"Number of convolutions: {args.num_convs_phn}")
    print(f"Output dim: {args.output_dim_phn}")
    print(f"Dropout CNN: {args.dropout_cnn_phn}")
    print(f"Dropout MLP: {args.dropout_mlp_phn}")
    print()
    print()
    
    print("WORD-LEVEL PARAMETERS")
    print(f"Number of convolutions: {args.num_convs_word}")
    print(f"Output dim: {args.output_dim_word}")
    print(f"Dropout CNN: {args.dropout_cnn_word}")
    print(f"Dropout MLP: {args.dropout_mlp_word}")
    print()
    print()
    
    print("UTTERANCE-LEVEL PARAMETERS")
    print(f"Number of convolutions: {args.num_convs_utt}")
    print(f"Output dim: {args.output_dim_utt}")
    print(f"Dropout CNN: {args.dropout_cnn_utt}")
    print(f"Dropout MLP: {args.dropout_mlp_utt}")
    print()
    print()
    sys.stdout.flush()
          
print(am)
tr_dataset = GoPDataset('train', am=am)
tr_dataloader_apa = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
te_dataset = GoPDataset('test', am=am)
te_dataloader = DataLoader(te_dataset, batch_size=2500, shuffle=False)

train(audio_mdl, tr_dataloader_apa, te_dataloader, args)
