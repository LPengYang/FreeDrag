import copy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import numpy as np


def get_features_plus(feature, position):
    # feature: (1,C,H,W)
    # position: (N,2)
    # return: (N,C)
    device = feature.device

    y = position[:,0]
    x = position[:,1]

    x0 = x.long()
    x1 = x0+1
    y0 = y.long()
    y1 = y0+1
    
    wa = ((x1.float() - x) * (y1.float() - y)).to(device).unsqueeze(1).detach()
    wb = ((x1.float() - x) * (y - y0.float())).to(device).unsqueeze(1).detach()
    wc = ((x - x0.float()) * (y1.float() - y)).to(device).unsqueeze(1).detach()
    wd = ((x - x0.float()) * (y - y0.float())).to(device).unsqueeze(1).detach()

    Ia = feature[:, :, y0, x0].squeeze(0).transpose(1,0)
    Ib = feature[:, :, y1, x0].squeeze(0).transpose(1,0)
    Ic = feature[:, :, y0, x1].squeeze(0).transpose(1,0)
    Id = feature[:, :, y1, x1].squeeze(0).transpose(1,0)

    output = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return output

def update_signs(sign_point_pairs, current_point, target_point,loss_supervised,threshold_d,threshold_l):
    
    distance = (current_point-target_point).pow(2).sum(dim=1).pow(0.5)
    sign_point_pairs[distance<threshold_d]  = 1
    sign_point_pairs[distance>=threshold_d] = 0
    sign_point_pairs[loss_supervised>threshold_l] = 0

    return sign_point_pairs


def get_each_point(current,target_final,L, feature_map,max_distance,template_feature,
                    loss_initial,loss_end,position_local,threshold_l):
    d_max = max_distance 
    d_remain = (current-target_final).pow(2).sum().pow(0.5)
    interval_number  = 10 # for point localization 
    intervals = torch.arange(0,1+1/interval_number,1/interval_number,device = current.device)[1:].unsqueeze(1)

    if loss_end < threshold_l:
    # if True:
        target_max = current + min(d_max/(d_remain+1e-8),1)*(target_final-current) 
        candidate_points = (1-intervals)*current.unsqueeze(0) + intervals*target_max.unsqueeze(0)
        candidate_points_repeat = candidate_points.repeat_interleave(position_local.shape[0],dim=0)
        position_local_repeat = position_local.repeat(intervals.shape[0],1)

        candidate_points_local = candidate_points_repeat +position_local_repeat
        features_all = get_features_plus(feature_map, candidate_points_local)

        features_all = features_all.reshape((intervals.shape[0],-1))
        dif_location = abs(features_all-template_feature.flatten(0).unsqueeze(0)).mean(1)
        min_idx = torch.argmin(abs(dif_location-L))
        current_best = candidate_points[min_idx,:]
        return current_best
    
    elif loss_end<loss_initial:
         return current

    else:
        current = current- min(d_max/(d_remain+1e-8),1)*(target_final-current) # rollback 
        d_remain = (current-target_final).pow(2).sum().pow(0.5)
        target_max = current + min(2*d_max/(d_remain+1e-8),1)*(target_final-current) # double the localization range

        candidate_points = (1-intervals)*current.unsqueeze(0) + intervals*target_max.unsqueeze(0)
        candidate_points_repeat = candidate_points.repeat_interleave(position_local.shape[0],dim=0)
        position_local_repeat = position_local.repeat(intervals.shape[0],1)
        candidate_points_local = candidate_points_repeat +position_local_repeat
        features_all = get_features_plus(feature_map, candidate_points_local)
        features_all = features_all.reshape((intervals.shape[0],-1))
        dif_location = abs(features_all-template_feature.flatten(0).unsqueeze(0)).mean(1)
        min_idx = torch.argmin(dif_location)   # l=0 in this case
        current_best = candidate_points[min_idx,:]
        return current_best

    
def get_current_target(sign_points, current_target,target_point,L,feature_map,max_distance,template_feature,
                       loss_initial,loss_end,position_local,threshold_l):
     for k in range(target_point.shape[0]):
         if sign_points[k] ==0: # sign_points ==0 means the remains distance to target point is larger than the preset threshold
            current_target[k,:] = get_each_point(current_target[k,:],target_point[k,:],\
                                L,feature_map,max_distance,template_feature[k],loss_initial[k], loss_end[k],position_local,threshold_l)
     return current_target

def get_xishu(loss_k,a,b): 
    xishu = xishu = 1/(1+(a*(loss_k-b)).exp())
    return xishu


def get_position_for_feature(win_r):
    k = torch.linspace(-win_r,win_r,steps= 2*win_r+1)
    k1= k.repeat(2*win_r+1,1).transpose(1,0).flatten(0).unsqueeze(0)
    k2= k.repeat(1,2*win_r+1)
    return torch.cat((k1,k2),dim=0).transpose(1,0)


def check_handle_reach_target(handle_points,
                              target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()

# obtain the bilinear interpolated feature patch centered around (x, y) with radius r



def freedrag_update(model,
                          init_code,
                          t,
                          handle_points,
                          target_points,
                          mask,
                          args,
                          win_r = 4,
                          ):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    l_expected = args.l_expected
    d_max = args.d_max

    text_emb = model.get_text_embeddings(args.prompt).detach()
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(init_code, t, encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        x_prev_0,_ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    
    device = init_code.device
    point_pairs_number = len(handle_points)
    sign_points = torch.zeros(point_pairs_number,device = device)
    Loss_l1 = torch.nn.L1Loss()
    
    threshold_d = 2
    threshold_l = 0.5*l_expected
    aa = torch.log(torch.tensor(9.0,device=device))/(0.6*l_expected)
    bb = 0.2*l_expected

    print(l_expected,d_max)
    position_local = get_position_for_feature(int(win_r)).to(device)
    loss_ini = torch.zeros(point_pairs_number).to(device)
    loss_end = torch.zeros(point_pairs_number).to(device)
    current_targets = torch.stack(handle_points,dim=0).to(device)
    target_points = torch.stack(target_points,dim=0).to(device)
    current_F = copy.deepcopy(F0).to(device)
    
    F_template = []
    for k in range(point_pairs_number):
        F_template.append(get_features_plus(F0,current_targets[k,:]+position_local))
    
    np.set_printoptions(formatter={'float': '{:0.2f}'.format})
    
    step_idx = 0
    while True:  
        # break if all handle points have reached the targets
        if torch.all(sign_points==1): 
            break
        if step_idx > args.max_step:
            break

        # do linear search according distance and feature discrepancy
        current_targets = get_current_target(sign_points, current_targets,target_points,l_expected,current_F, 
            d_max, F_template, loss_ini, loss_end, position_local, threshold_l)
        print("current: ", current_targets.cpu().numpy())

        d_remain = (current_targets-target_points).pow(2).sum(dim=1).pow(0.5)
        print('d_remain: ', d_remain.cpu().numpy())

        for step in range(5):          
            step_idx += 1      
            # if step_idx > args.n_pix_step or step_idx>step_threshold+10:
            #     break

            if step_idx > args.max_step:
                break
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                unet_output, F1 = model.forward_unet_features(init_code, t, encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
                x_prev_updated,_ = model.step(unet_output, t, init_code)
                
                loss_supervised = torch.zeros(point_pairs_number).to(device)
                current_feature = []
                for k in range(point_pairs_number):
                    current_feature.append(get_features_plus(F1,current_targets[k,:]+position_local))
                    loss_supervised[k] = Loss_l1(current_feature[k],F_template[k].detach())
                
                loss_feature = loss_supervised.sum()
                if torch.any(interp_mask!=0):
                    loss_mask = ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().mean()
                    loss = loss_feature + args.lam*loss_mask
                    # print("lam", args.lam)
                else:
                    
                    loss = loss_feature

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if step == 0:
                loss_ini = loss_supervised
            if loss_supervised.max()<threshold_l:
                break

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, F1 = model.forward_unet_features(init_code, t, encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
                # x_prev_updated,_ = model.step(unet_output, t, init_code)
        
            current_feature = []
            for k in range(point_pairs_number):
                current_feature.append (get_features_plus(F1,current_targets[k,:]+position_local))
                loss_end[k] = Loss_l1(current_feature[k],F_template[k].detach())    
        print("loss_ini:", loss_ini.detach().cpu().numpy(), "loss_end:", loss_end.detach().cpu().numpy())
        
        sign_points = update_signs(sign_points,current_targets,target_points,loss_end,threshold_d,threshold_l)
        for k in range(point_pairs_number):
            if  sign_points[k]==1: 
                xishu = 1
            else:
                xishu = get_xishu(loss_end[k].detach(), aa, bb)

            F_template[k] = xishu*current_feature[k].detach() + (1-xishu)*F_template[k] 
        current_F = F1.detach()

    print(step_idx-1,d_remain.cpu().numpy())  
    return init_code

def freedrag_update_gen(model,
                          init_code,
                          t,
                          handle_points,
                          target_points,
                          mask,
                          args,
                          win_r = 4,
                          ):
    
    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"

    text_emb = model.get_text_embeddings(args.prompt).detach()

    text_emb = model.get_text_embeddings(args.prompt).detach()
    if args.guidance_scale > 1.0:
        unconditional_input = model.tokenizer(
            [args.neg_prompt],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_emb = model.text_encoder(unconditional_input.input_ids.to(text_emb.device))[0].detach()
        text_emb = torch.cat([unconditional_emb, text_emb], dim=0)

    # the init output feature of unet
    with torch.no_grad():
        if args.guidance_scale > 1.:
            model_inputs_0 = copy.deepcopy(torch.cat([init_code] * 2))
        else:
            model_inputs_0 = copy.deepcopy(init_code)
        unet_output, F0 = model.forward_unet_features(model_inputs_0, t, encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        if args.guidance_scale > 1.:
            coef = args.guidance_scale / (2*args.guidance_scale - 1.0)
            F0 = torch.cat([(1-coef)*F0[0], coef*F0[1]], dim=0).unsqueeze(dim=0)

            unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
            unet_output = unet_output_uncon + args.guidance_scale * (unet_output_con - unet_output_uncon)
        x_prev_0,_ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    # handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    
    device = init_code.device
    point_pairs_number = len(handle_points)
    sign_points = torch.zeros(point_pairs_number,device = device)
    Loss_l1 = torch.nn.L1Loss()
    l_expected = args.l_expected
    d_max = args.d_max
    threshold_d = 2
    threshold_l = 0.5*l_expected
    aa = torch.log(torch.tensor(9.0,device=device))/(0.6*l_expected)
    bb = 0.2*l_expected
    
    print(l_expected,d_max)
    position_local = get_position_for_feature(win_r).to(device)
    loss_ini = torch.zeros(point_pairs_number).to(device)
    loss_end = torch.zeros(point_pairs_number).to(device)
    current_targets = torch.stack(handle_points,dim=0).to(device)
    target_points = torch.stack(target_points,dim=0).to(device)

    current_F = copy.deepcopy(F0).to(device)
    
    F_template = []
    for k in range(point_pairs_number):
        F_template.append(get_features_plus(F0,current_targets[k,:]+position_local))
    
    np.set_printoptions(formatter={'float': '{:0.2f}'.format})
    step_idx = 0
    while True:  
        # break if all handle points have reached the targets
        if torch.all(sign_points==1): 
            break
        if step_idx > args.max_step:
            break

        # do linear search according distance and feature discrepancy
        current_targets = get_current_target(sign_points, current_targets,target_points,l_expected,current_F, 
            d_max, F_template, loss_ini, loss_end, position_local, threshold_l)
        print("current: ", current_targets.cpu().numpy())

        d_remain = (current_targets-target_points).pow(2).sum(dim=1).pow(0.5)
        print('d_remain: ', d_remain.cpu().numpy())

        for step in range(5):          
            step_idx += 1      
            if step_idx > args.max_step:
                break
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):    
                if args.guidance_scale > 1.:
                    model_inputs = init_code.repeat(2,1,1,1)
                else:
                    model_inputs = init_code

                unet_output, F1 = model.forward_unet_features(model_inputs, t, encoder_hidden_states=text_emb,
                      layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)

                if args.guidance_scale > 1.:
                    coef = args.guidance_scale / (2*args.guidance_scale - 1.0)
                    F1 = torch.cat([(1-coef)*F1[0], coef*F1[1]], dim=0).unsqueeze(dim=0)

                    unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
                    unet_output = unet_output_uncon + args.guidance_scale * (unet_output_con - unet_output_uncon)

                x_prev_updated,_ = model.step(unet_output, t, init_code)
         
                loss_supervised = torch.zeros(point_pairs_number).to(device)
                current_feature = []
                for k in range(point_pairs_number):
                    current_feature.append(get_features_plus(F1,current_targets[k,:]+position_local))
                    loss_supervised[k] = Loss_l1(current_feature[k], F_template[k].detach())       
                loss_feature = loss_supervised.sum()
            
                if torch.any(interp_mask!=0):
                    loss_mask = ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().mean()
                    loss = loss_feature + args.lam*loss_mask
                else:
                    loss = loss_feature

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()        
            if step == 0:
                loss_ini = loss_supervised
            if loss_supervised.max()<threshold_l:
                break

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if args.guidance_scale > 1.:
                    model_inputs = init_code.repeat(2,1,1,1)
                else:
                    model_inputs = init_code

                _, F1 = model.forward_unet_features(model_inputs, t, encoder_hidden_states=text_emb,
                      layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)

                if args.guidance_scale > 1.:
                    coef = args.guidance_scale / (2*args.guidance_scale - 1.0)
                    F1 = torch.cat([(1-coef)*F1[0], coef*F1[1]], dim=0).unsqueeze(dim=0)

            current_feature = []
            for k in range(point_pairs_number):
                current_feature.append (get_features_plus(F1,current_targets[k,:]+position_local))
                loss_end[k] = Loss_l1(current_feature[k],F_template[k].detach())    
        print("loss_ini:", loss_ini.detach().cpu().numpy(), "loss_end:", loss_end.detach().cpu().numpy())
        
        sign_points = update_signs(sign_points,current_targets,target_points,loss_end,threshold_d,threshold_l)
        for k in range(point_pairs_number):
            if  sign_points[k]==1: 
                xishu = 1
            else:
                xishu = get_xishu(loss_end[k].detach(), aa, bb)
            F_template[k] = xishu*current_feature[k].detach() + (1-xishu)*F_template[k] 
        current_F = F1.detach()

    print(step_idx-1,d_remain.cpu().numpy())
    return init_code