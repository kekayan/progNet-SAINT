import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from augmentations import embed_data_mask


def SAINT_pretrain(model,train_ds, valid_ds,opt,device):
    workers = 2
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=workers)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=True,num_workers=workers) 

    optimizer = optim.AdamW(model.parameters(),lr=0.0001)
    pt_aug_dict = {
        'noise_type' : opt.pt_aug,
        'lambda' : opt.pt_aug_lam
    }
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    best_val_loss = 100000
    patience = 10
    early_stop_counter = 0

    print("Pretraining begins!")
    for epoch in range(opt.pretrain_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_categ, x_cont, _ ,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            
            # embed_data_mask function is used to embed both categorical and continuous data.
            if 'cutmix' in opt.pt_aug:
                from augmentations import add_noise
                x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model)
            else:
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            
            if 'mixup' in opt.pt_aug:
                from augmentations import mixup_data
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=opt.mixup_lam, use_cuda=True if device == 'cuda' else False) 
            
            loss = 0
            if 'contrastive' in opt.pt_tasks:
                aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                if opt.pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif opt.pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')
                logits_per_aug1 = aug_features_1 @ aug_features_2.t()/opt.nce_temp
                logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/opt.nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                loss   = opt.lam0*(loss_1 + loss_2)/2

            elif 'contrastive_sim' in opt.pt_tasks:
                aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_1 = model.pt_mlp(aug_features_1)
                aug_features_2 = model.pt_mlp2(aug_features_2)
                c1 = aug_features_1 @ aug_features_2.t()
                loss+= opt.lam1*torch.diagonal(-1*c1).add_(1).pow_(2).sum()

            if 'denoising' in opt.pt_tasks:
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                # if con_outs.shape(-1) != 0:
                # import ipdb; ipdb.set_trace()
                if len(con_outs) > 0:
                    con_outs =  torch.cat(con_outs,dim=1)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                l1 = 0
                # import ipdb; ipdb.set_trace()
                n_cat = x_categ.shape[-1]
                for j in range(1,n_cat):
                    l1+= criterion1(cat_outs[j],x_categ[:,j])
                loss += opt.lam2*l1 + opt.lam3*l2 


            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validloader, 0):
                x_categ, x_cont, _ ,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
                
                # embed_data_mask function is used to embed both categorical and continuous data.
                if 'cutmix' in opt.pt_aug:
                    from augmentations import add_noise
                    x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                    _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model)
                else:
                    _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
                
                if 'mixup' in opt.pt_aug:
                    from augmentations import mixup_data
                    x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=opt.mixup_lam, use_cuda=True if device == 'cuda' else False) 
                val_loss = 0
                if 'contrastive' in opt.pt_tasks:
                    aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                    aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                    aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                    aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                    if opt.pt_projhead_style == 'diff':
                        aug_features_1 = model.pt_mlp(aug_features_1)
                        aug_features_2 = model.pt_mlp2(aug_features_2)
                    elif opt.pt_projhead_style == 'same':
                        aug_features_1 = model.pt_mlp(aug_features_1)
                        aug_features_2 = model.pt_mlp(aug_features_2)
                    else:
                        print('Not using projection head')
                    logits_per_aug1 = aug_features_1 @ aug_features_2.t()/opt.nce_temp
                    logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/opt.nce_temp
                    targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                    loss_1 = criterion1(logits_per_aug1, targets)
                    loss_2 = criterion1(logits_per_aug2, targets)
                    val_loss   = opt.lam0*(loss_1 + loss_2)/2
                elif 'contrastive_sim' in opt.pt_tasks:
                    aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                    aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                    aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                    aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                    c1 = aug_features_1 @ aug_features_2.t()
                    val_loss+= opt.lam1*torch.diagonal(-1*c1).add_(1).pow_(2).sum()
                if 'denoising' in opt.pt_tasks:
                    cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                    # if con_outs.shape(-1) != 0:
                    # import ipdb; ipdb.set_trace()
                    if len(con_outs) > 0:
                        con_outs =  torch.cat(con_outs,dim=1)
                        l2 = criterion2(con_outs, x_cont)
                    else:
                        l2 = 0
                    l1 = 0
                    # import ipdb; ipdb.set_trace()
                    n_cat = x_categ.shape[-1]
                    for j in range(1,n_cat):
                        l1+= criterion1(cat_outs[j],x_categ[:,j])
                    val_loss += opt.lam2*l1 + opt.lam3*l2
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.eval()
            checkpoint = {'model': model, 'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, opt.save_path)
            print('Model Checkpoint Saved!')
            early_stop_counter = 0
        else:
            early_stop_counter+=1
            print('Early Stopping Counter: ',early_stop_counter)
            if early_stop_counter > patience:
                print('Early Stopping!')
                break    
        
        print(f'Epoch: {epoch}, Running Loss: {running_loss} , Val Loss: {val_loss}')

    print('END OF PRETRAINING!')
    return model