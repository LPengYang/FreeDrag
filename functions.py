import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import math
import numpy as np
from typing import List, Tuple
import copy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import lpips

def add_watermark_np(input_image_array, watermark_text="AI Generated"):
    image = PIL.Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = PIL.Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = PIL.ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = PIL.ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = PIL.Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return watermarked_array

def to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')

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
        aa = torch.log(torch.tensor(9.0,device=device))/(0.6*l_expected)
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
        current_target = handle_point.clone().to(device)
        current_feature_map = feature_original.detach()

        sign_points= torch.zeros(point_pairs_number).to(device) # determiner if the localization point is closest to target point
        loss_ini = torch.zeros(point_pairs_number).to(device)
        loss_end = torch.zeros(point_pairs_number).to(device)
        step_threshold = max_steps
        
        while step_number<max_steps:
             if torch.all(sign_points==1):         
                _,img_show = model.get_features(ws_input,x=feature_mid, img=img_mid, mid_size=handle_size,noise_mode='const')
                yield img_show, current_target*(full_size/handle_size), step_number, full_size, torch.cat([ws_trainable, ws_untrainable], dim=1).detach()
                break
             
             current_target = get_current_target(sign_points, current_target,target_point,l_expected,current_feature_map, 
                d_max, template_feature, loss_ini, loss_end, position_local, threshold_l)
             d_remain = (current_target-target_point).pow(2).sum(dim=1).pow(0.5)

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

                if step_number == max_steps or step_number>step_threshold+10:
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
             if d_remain.max()<threshold_d:
                 step_threshold = step_number
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

def noise_regularize(noises):
    loss = 0

    for noise_res in noises:
        size = noise_res.shape[2]
        for i in range(noise_res.shape[0]):
            noise = noise_res[i,:].unsqueeze(0).unsqueeze(0)
            while True:
                loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
                )

                if size <= 8:
                    break

                noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
                noise = noise.mean([3, 5])
                size //= 2
    return loss


def noise_normalize_(noises):
    for noise_res in noises:
        for i in range(noise_res.shape[0]):
            noise = noise_res[i,:]
            mean = noise.mean()
            std = noise.std()
            noise.data.add_(-mean).div_(std)

def lambda_rule(step):
    if step<50:
        lr_l = (step+1)/50
    elif step <750:
        lr_l = 1
    else:
        lr_l = math.cos((step-750)/(250)*math.pi)/2 + 0.5
    return lr_l

def image_inversion(image_real,generated_res,G_load):
    model = G_load.g
    device = model.mapping.w_avg.device
    label = torch.zeros([1, model.c_dim], device=device)
    data_tranform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.Resize([generated_res,generated_res])])
    total_step = 1000
    label = torch.zeros([1, model.c_dim], device=device)
    with torch.no_grad():
        noise_sample = torch.randn(10000, model.z_dim, device=device)
        latent_out = model.get_ws(noise_sample,label)
        latent_out_single = latent_out[:,0,:]
        latent_mean = latent_out_single.mean(0)
        latent_std = ((latent_out_single - latent_mean).pow(2).sum()/10000) ** 0.5

    ws_mean = model.mapping.w_avg
    ws_trainable_single = ws_mean.unsqueeze(0).unsqueeze(0).clone().requires_grad_(True)
    image_real = data_tranform(image_real).unsqueeze(0).to(device)

    current_res = 4
    noises_train = []
    while current_res<=generated_res:
        if current_res == 4:
            noises_train.append(torch.randn([1,current_res, current_res],device=device).requires_grad_(True))
        else:
            noises_train.append(torch.randn([2,current_res, current_res],device=device).requires_grad_(True))
        current_res *= 2

    optimizer = torch.optim.Adam([ws_trainable_single]+noises_train, lr=0.1)
    scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    
    percept = lpips.LPIPS(net='vgg').to(device)
    for step in range(total_step):

            noise_strength = latent_std * 0.05 * max(0, 1 - step/(0.75*total_step) ) ** 2
            noise_latent = torch.randn_like(ws_trainable_single) * noise_strength
            ws_addnoise = ws_trainable_single + noise_latent.detach()

            ws_trainable = ws_addnoise.repeat(1,latent_out.shape[1],1)
            _, image_generated = model.get_features(ws=ws_trainable, noises=noises_train, noise_mode='trainable')
     
            loss_percep = percept(F.interpolate(image_generated, size=(256,256), mode='nearest'),
                                 F.interpolate(image_real.detach(), size=(256,256), mode='nearest'))
            loss_noise_regula  = noise_regularize(noises_train)
            loss = loss_percep + 1e5*loss_noise_regula 

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler_gen.step()

            noise_normalize_(noises_train)
    image_show = to_image(image_generated)
    
    if G_load.name == 'faces.pkl':
        add_watermark = torch.ones(1,device=device)
    else:
        add_watermark = torch.zeros(1,device=device)

    return ws_trainable, image_show, image_show, add_watermark




             