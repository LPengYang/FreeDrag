import PIL
import PIL.Image
import PIL.ImageDraw
import math
import numpy as np
from typing import List, Optional, Tuple
import copy
import torch
import torch.nn.functional as F

def get_position_for_feature(win_r,handle_size,full_size):
    k = torch.linspace(-(win_r*(handle_size/full_size)),win_r*(handle_size/full_size),steps= win_r)
    # k = torch.linspace(-(win_r//2),win_r//2,steps= win_r)
    k1= k.repeat(win_r,1).transpose(1,0).flatten(0).unsqueeze(0)
    k2= k.repeat(1,win_r)
    return torch.cat((k1,k2),dim=0).transpose(1,0)

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
    sign_point_pairs[loss_supervised>threshold_l] =0


def get_each_point(current,target_final,L, feature_map,max_distance,template_feature,
                    loss_initial,loss_end,position_local,threshold_l):
    d_max = max_distance 
    d_remain = (current-target_final).pow(2).sum().pow(0.5)
    interval_number  = 10 # for point localization 
    intervals = torch.arange(0,1+1/interval_number,1/interval_number,device = current.device)[1:].unsqueeze(1)

    if loss_end < threshold_l:
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
    

def free_drag(model, points, mask, handle_size, train_layer_index, ws_original,max_steps,l_expected,d_max,sample_interval,device):
        # max_steps: max optimization for the total motion
        # l_expected: expected loss at the beginning of each sub-motion
        # d_max: max distance for each sub-motion (in the feature map)
        # handle_size: the size of handled feature map

        win_r = 3
        threshold_l = 0.5*l_expected
        aa = torch.log(torch.tensor(9.0).cuda())/(0.6*l_expected)
        bb = 0.2*l_expected
        feature_original, img_mid_original = model.get_features(ws_original,x=None, img=None, mid_size= handle_size)
        
        use_mask = False
        if np.any(mask==1):
            mask = torch.tensor(mask,dtype=torch.float32,device=device).unsqueeze(0).unsqueeze(0)
            mask_resized = F.interpolate(mask,size = (handle_size,handle_size),mode ='bilinear')
            mask_resized = mask_resized.repeat(1,feature_original.shape[1],1,1) >0
            use_mask = True
            
        _,img_show_original = model.get_features(ws=ws_original,x=feature_original,img=img_mid_original,mid_size=handle_size)
        full_size = img_show_original.shape[2]
        threshold_d = handle_size/full_size
        position_local = get_position_for_feature(win_r,handle_size,full_size).to(device)
        
    
        ws_trainable = ws_original[:, :train_layer_index, :].detach().clone().requires_grad_(True)
        ws_untrainable = ws_original[:, train_layer_index:, :].detach().clone().requires_grad_(False)
        
        optimizer_mlp = torch.optim.Adam([
                    {'params':ws_trainable}
                    ], lr=0.002,  eps=1e-08, weight_decay=0, amsgrad=False)
        Loss_l1 = torch.nn.L1Loss()
        
        handle_point = [torch.tensor(p, device=device).float() for p in points['handle']]
        target_point = [torch.tensor(p, device=device).float() for p in points['target']]
        handle_point = torch.stack(handle_point)
        target_point = torch.stack(target_point)
        handle_point = handle_point *(handle_size/full_size)
        target_point = target_point *(handle_size/full_size)

        point_pairs_number = target_point.shape[0]
        template_feature = []
        for k in range(point_pairs_number):
            template_feature.append( get_features_plus(feature_original,handle_point[k,:]+position_local))

        step_number = 0
        current_target = handle_point.clone().cuda()
        current_feature_map = feature_original.detach()

        sign_points= torch.zeros(point_pairs_number).to(device) # determiner if the localization point is closest to target point
        loss_ini = torch.zeros(point_pairs_number).to(device)
        loss_end = torch.zeros(point_pairs_number).to(device)
        
        
        while step_number<max_steps:
             if torch.all(sign_points==1):         
                _,img_show = model.get_features(ws_input,x=feature_mid, img=img_mid, mid_size=handle_size,noise_mode='const')
                yield img_show, current_target*(full_size/handle_size), step_number, full_size, torch.cat([ws_trainable, ws_untrainable], dim=1).detach()
                break
             
             current_target = get_current_target(sign_points, current_target,target_point,l_expected,current_feature_map, 
                d_max, template_feature, loss_ini, loss_end, position_local, threshold_l)

             for step in range(5):          
                step_number +=1      

                ws_input = torch.cat((ws_trainable,ws_untrainable),dim=1)
                feature_mid, img_mid = model.get_features(ws_input,x=None, img=None, mid_size=handle_size,noise_mode='const')
                
                loss_supervised = torch.zeros(point_pairs_number).to(device)
                current_feature = []
                for k in range(point_pairs_number):
                    current_feature.append(get_features_plus(feature_mid,current_target[k,:]+position_local))
                    loss_supervised[k] = Loss_l1(current_feature[k],template_feature[k].detach())
                
                loss_feature = loss_supervised.sum()

                if use_mask:
                    loss_mask = Loss_l1(feature_mid[~mask_resized],feature_original[~mask_resized].detach())
                    loss = loss_feature + 10*loss_mask
                else:
                    loss = loss_feature
                loss.backward()
                optimizer_mlp.step()
                optimizer_mlp.zero_grad()

                if step_number%sample_interval==0:
                    _,img_show = model.get_features(ws_input,x=feature_mid, img=img_mid, mid_size=handle_size,noise_mode='const')
                    yield img_show, current_target*(full_size/handle_size), step_number, full_size, torch.cat([ws_trainable, ws_untrainable], dim=1).detach()

                if step ==0:
                    loss_ini = loss_supervised

                if loss_supervised.max()<0.5*threshold_l:
                    break

                if step_number == max_steps:
                    _,img_show = model.get_features(ws_input,x=feature_mid, img=img_mid, mid_size=handle_size,noise_mode='const')
                    yield img_show, current_target*(full_size/handle_size), step_number, full_size, torch.cat([ws_trainable, ws_untrainable], dim=1).detach()
                    break


             with torch.no_grad():
                 ws_input = torch.cat((ws_trainable,ws_untrainable),dim=1)
                 feature_mid, img_mid = model.get_features(ws_input,x=None, img=None, mid_size=handle_size,noise_mode='const')
                
                 current_feature = []
                 for k in range(point_pairs_number):
                    current_feature.append (get_features_plus(feature_mid,current_target[k,:]+position_local))
                    loss_end[k] = Loss_l1(current_feature[k],template_feature[k].detach())

             update_signs(sign_points,current_target,target_point,loss_end,threshold_d,0.5*threshold_l)
             for k in range(point_pairs_number):
                if  sign_points[k]==1: 
                    xishu = 1
                else:
                    xishu = get_xishu(loss_end[k].detach(), aa, bb)
                template_feature[k] = xishu*current_feature[k].detach() + (1-xishu)*template_feature[k] 
                      
             current_feature_map = feature_mid.detach()
             
def get_ellipse_coords(
    point: Tuple[int, int], radius: int = 5
) -> Tuple[int, int, int, int]:
    """
    Returns the coordinates of an ellipse centered at the given point.

    Args:
        point (Tuple[int, int]): The center point of the ellipse.
        radius (int): The radius of the ellipse.

    Returns:
        A tuple containing the coordinates of the ellipse in the format (x_min, y_min, x_max, y_max).
    """
    center = point
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )

def draw_handle_target_points(
        img: PIL.Image.Image,
        handle_points: List[Tuple[int, int]],
        target_points: List[Tuple[int, int]],
        radius: int = 5,
        color = "red"):
    """
    Draws handle and target points with arrow pointing towards the target point.

    Args:
        img (PIL.Image.Image): The image to draw on.
        handle_points (List[Tuple[int, int]]): A list of handle [x,y] points.
        target_points (List[Tuple[int, int]]): A list of target [x,y] points.
        radius (int): The radius of the handle and target points.
    """
    if not isinstance(img, PIL.Image.Image):
        img = PIL.Image.fromarray(img)

    if len(handle_points) == len(target_points) + 1:
        target_points = copy.deepcopy(target_points) + [None]

    draw = PIL.ImageDraw.Draw(img)
    for handle_point, target_point in zip(handle_points, target_points):
        handle_point = [handle_point[1], handle_point[0]]
        # Draw the handle point
        handle_coords = get_ellipse_coords(handle_point, radius)
        draw.ellipse(handle_coords, fill=color)

        if target_point is not None:
            target_point = [target_point[1], target_point[0]]
            # Draw the target point
            target_coords = get_ellipse_coords(target_point, radius)
            draw.ellipse(target_coords, fill="blue")

            # Draw arrow head
            arrow_head_length = radius*1.5

            # Compute the direction vector of the line
            dx = target_point[0] - handle_point[0]
            dy = target_point[1] - handle_point[1]
            angle = math.atan2(dy, dx)

            # Shorten the target point by the length of the arrowhead
            shortened_target_point = (
                target_point[0] - arrow_head_length * math.cos(angle),
                target_point[1] - arrow_head_length * math.sin(angle),
            )

            # Draw the arrow (main line)
            draw.line([tuple(handle_point), shortened_target_point], fill='white', width=int(0.8*radius))

            # Compute the points for the arrowhead
            arrow_point1 = (
                target_point[0] - arrow_head_length * math.cos(angle - math.pi / 6),
                target_point[1] - arrow_head_length * math.sin(angle - math.pi / 6),
            )

            arrow_point2 = (
                target_point[0] - arrow_head_length * math.cos(angle + math.pi / 6),
                target_point[1] - arrow_head_length * math.sin(angle + math.pi / 6),
            )

            # Draw the arrowhead
            draw.polygon([tuple(target_point), arrow_point1, arrow_point2], fill='white')
    return np.array(img)
             