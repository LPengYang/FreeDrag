import gradio as gr
import torch
import numpy as np
from functions import to_image, draw_handle_target_points, free_drag, image_inversion, add_watermark_np
import dnnlib
from training import networks
import legacy
import cv2

# export CUDA_LAUNCH_BLOCKING=1
def load_model(model_name, device):

    path = './checkpoints/' + str(model_name)
    with dnnlib.util.open_url(path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) 
        G_copy = networks.Generator(z_dim=G.z_dim, c_dim= G.c_dim, w_dim =G.w_dim, 
                                     img_resolution = G.img_resolution,           
                                     img_channels   = G.img_channels,               
                                     mapping_kwargs      = G.init_kwargs['mapping_kwargs'])    

        G_copy.load_state_dict(G.state_dict())
        G_copy.to(device)
        del(G)
        for param in G_copy.parameters():
            param.requires_grad = False
    return G_copy, model_name

def draw_mask(image,mask):

    image_mask = image*(1-mask) +mask*(0.7*image+0.3*255.0)

    return image_mask


class ModelWrapper:
    def __init__(self, model,model_name):
        self.g = model
        self.name = model_name
        self.res = CKPT_SIZE[model_name][0]
        self.l = CKPT_SIZE[model_name][1]
        self.d = CKPT_SIZE[model_name][2]


# model, points, mask, feature_size, train_layer_index,max_step, device,seed=2023,max_distance=3, d=0.5
# img_show, current_target, step_number
def on_drag(model, points, mask, max_iters,latent,sample_interval,l_expected,d_max,save_video,add_watermark):

    if len(points['handle']) == 0:
        raise gr.Error('You must select at least one handle point and target point.')
    if len(points['handle']) != len(points['target']):
        raise gr.Error('You have uncompleted handle points, try to selct a target point or undo the handle point.')
    max_iters = int(max_iters)
    
    handle_size = 128
    train_layer_index=6
    l_expected = torch.tensor(l_expected,device=latent.device)
    d_max = torch.tensor(d_max,device=latent.device)
    mask[mask>0] = 1
    global stop_flag 
    stop_flag = False
    images_total = []
    for img_show, current_target, step_number,full_res, latent_optimized in free_drag(model.g,points,mask[:,:,0],handle_size, \
                train_layer_index,latent,max_iters,l_expected,d_max,sample_interval,device=latent.device):
        image = to_image(img_show)

        points['handle'] = [current_target[p,:].cpu().numpy().astype('int') for p in range(len(current_target[:,0]))]
        image_show = add_points_to_image(image, points, size=RES_TO_CLICK_SIZE[full_res],color="yellow")

        if np.any(mask[:,:,0]>0):
            image_show = draw_mask(image_show,mask)
            image_show = np.uint8(image_show)

        if add_watermark:
           image_show = add_watermark_np(np.array(image_show))[:,:,[0,1,2]]
           image_clear = add_watermark_np(np.array(image))[:,:,[0,1,2]]
        else:
           image_clear = image

        if save_video:
            images_total.append(image_show)
        yield (image_show, step_number, latent_optimized,image_clear,images_total,gr.Button.update(interactive=True))

        if stop_flag:
            break
   
def add_points_to_image(image, points, size=5,color="red"):
    image = draw_handle_target_points(image, points['handle'], points['target'], size, color)
    return image

def on_show_save():
    return gr.update(visible=True)

def on_click(image, target_point, points, res, evt: gr.SelectData):
    if target_point:
        points['target'].append([evt.index[1], evt.index[0]])
        image = add_points_to_image(image, points, size=RES_TO_CLICK_SIZE[res])
        return image, not target_point
    points['handle'].append([evt.index[1], evt.index[0]])
    image = add_points_to_image(image, points, size=RES_TO_CLICK_SIZE[res])
    return image, not target_point

def new_image(model,seed=-1):
    if seed == -1:
        seed = np.random.randint(1,1e6)
    z1 = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, model.g.z_dim)).to(device)
    label = torch.zeros([1, model.g.c_dim], device=device)
    ws_original= model.g.get_ws(z1,label,truncation_psi=0.7)
    _, img_show_original = model.g.synthesis(ws=ws_original,noise_mode='const')

    return to_image(img_show_original), to_image(img_show_original), ws_original, seed

def new_model(model_name):   
    model_load, _ = load_model(model_name, device)
    model = ModelWrapper(model_load,model_name)

    return model, model.res, model.l, model.d

def reset_all(image,mask,add_watermark=0):
    points = {'target': [], 'handle': []}
    target_point = False
    mask = np.zeros_like(mask,dtype=np.uint8)

    return points, target_point, image, None,mask, add_watermark

def add_mask(image_show,mask):
    image_show = draw_mask(image_show,mask)
    return image_show

def update_mask(image,mask_show):
    mask = np.zeros_like(image)
    if mask_show != None and np.any(mask_show['mask'][:,:,0]>1):
        mask[mask_show['mask'][:,:,:3]>0] =1
        image_mask = add_mask(image,mask)
        return np.uint8(image_mask), mask
    else:
        return image, mask

def on_select_mask_tab(image):
    return image

def change_stop_state():
    global stop_flag
    stop_flag = True

def save_video(imgs_show_list,frame):
    if len(imgs_show_list)>0:
        video_name = './process.mp4'
        fource = cv2.VideoWriter_fourcc(*'mp4v')
        full_res = imgs_show_list[0].shape[0]
        video_output = cv2.VideoWriter(video_name,fourcc=fource,fps=frame,frameSize = (full_res,full_res))
        for k in range(len(imgs_show_list)):
            video_output.write(imgs_show_list[k][:,:,::-1])
        video_output.release()
    return []
    
CKPT_SIZE = {
    'faces.pkl':[512, 0.3, 3],
    'horses.pkl': [256, 0.3, 3], 
    'elephants.pkl': [512, 0.4, 4],
    'lions.pkl':[512, 0.4, 4],
    'dogs.pkl':[1024, 0.4, 4],
    'bicycles.pkl':[256, 0.3, 3],
    'giraffes.pkl':[512, 0.4, 4],
    'cats.pkl':[512, 0.3, 3], 
    'cars.pkl':[512, 0.3, 3],
    'churches.pkl':[256, 0.3, 3],
    'metfaces.pkl':[1024, 0.3, 3],
}
RES_TO_CLICK_SIZE = {
    1024: 10,
    512: 5,
    256: 3,
}

if torch.cuda.is_available():    
   device =  'cuda'
else:
   device = 'cpu'
   
demo = gr.Blocks()

with demo:

    points = gr.State({'target': [], 'handle': []})
    target_point = gr.State(False)
    state = gr.State({})
    
    gr.Markdown(
            """
            # **FreeDrag**
            
            Official implementation of [FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing](https://github.com/LPengYang/FreeDrag)

            
            ## Parameter Description
             **max_step**: max number of optimization step 
             
             **sample_interval**: the interval between sampled optimization step. 
             This parameter only affects the visualization of intermediate results and does not have any impact on the final outcome. 
             For high-resolution images(such as model of dog), a larger sample_interval can significantly accelerate the dragging process.
             
             **Eepected initial loss and Max distance**: In the current version, both of these values are empirically set for each model. 
             Generally, for precise editing needs (e.g., merging eyes), smaller values are recommended, which may causes longer processing times. 
             Users can set these values according to practical editing requirements. We are currently seeking an automated solution.
             
             **frame_rate**: the frame rate for saved video.
             
             ## Hints
            - Handle points (Blue): the point you want to drag.
            - Target points (Red): the destination you want to drag towards to.
            - **Localized points (Yellow)**: the localized points in sub-motion         
            """,
        )
    
    with gr.Row():
        with gr.Column(scale=0.4):
            with gr.Accordion("Model"):
                with gr.Row():
                    with gr.Column(min_width=100):
                        seed = gr.Number(label='Seed',value=0)
                    with gr.Column(min_width=100):
                        button_new = gr.Button('Image Generate', variant='primary')
                        button_rand = gr.Button('Rand Generate')
                model_name = gr.Dropdown(label="Model name",choices=list(CKPT_SIZE.keys()),value = list(CKPT_SIZE.keys())[0])
            
            with gr.Accordion('Optional Parameters'):
                with gr.Row():
                    with gr.Column(min_width=100):
                         max_step = gr.Number(label='Max step',value=2000)
                    with gr.Column(min_width=100):
                         sample_interval = gr.Number(label='Interval',value=5,info="Sampling interval")

                model_load, _ = load_model(model_name.value, device)
                model = gr.State(ModelWrapper(model_load,model_name.value))
                l_expected = gr.Slider(0.1,0.5,label='Eepected initial loss for each sub-motion',value = model.value.l,step=0.05)
                d_max= gr.Slider(1.0,6.0,label='Max distance for each sub-motion (in the feature map)',value = model.value.d,step=0.5)

                res = gr.State(model.value.res)
                z1 = torch.from_numpy(np.random.RandomState(int(seed.value)).randn(1, model.value.g.z_dim)).to(device)
                label = torch.zeros([1, model.value.g.c_dim], device=device)
                ws_original= model.value.g.get_ws(z1,label,truncation_psi=0.7)
                latent = gr.State(ws_original)
                add_watermark = gr.State(torch.zeros(1,device=device))
                
                _, img_show_original = model.value.g.synthesis(ws=ws_original,noise_mode='const')
            
            with gr.Accordion('Video'):
                images_total = gr.State([])
                with gr.Row():
                    with gr.Column(min_width=100):
                      if_save_video = gr.Radio(["True","False"],value="False",label="if save video")
                    with gr.Column(min_width=100):
                      frame_rate = gr.Number(label="Frame rate",value=5)
                with gr.Row():
                    with gr.Column(min_width=100):
                      button_video = gr.Button('Save video', variant='primary')

            with gr.Accordion('Drag'):

                with gr.Row():
                    with gr.Column(min_width=200):
                        reset_btn = gr.Button('Reset points and mask')
                with gr.Row():
                    with gr.Column(min_width=100):
                        button_drag = gr.Button('Drag it', variant='primary')
                    with gr.Column(min_width=100):
                        button_stop = gr.Button('Stop')
                    
            progress = gr.Number(value=0, label='Steps', interactive=False)

        with gr.Column(scale=0.53):
            with gr.Tabs() as Tabs:
                    image_show = to_image(img_show_original)
                    image_clear = gr.State(image_show)
                    mask = gr.State(np.zeros_like(image_clear.value))
                    with gr.Tab('Setup Handle Points', id='input') as imagetab:
                        image = gr.Image(image_show).style(height=768, width=768)
                    with gr.Tab('Draw a Mask', id='mask') as masktab:
                        mask_show = gr.ImageMask(image_show).style(height=768, width=768)
    
    image.select(on_click, [image, target_point, points, res], [image, target_point]).then(on_show_save)

    image.upload(image_inversion,[image,res,model],[latent,image,image_clear,add_watermark]).then(reset_all,
                                        inputs=[image_clear,mask,add_watermark],outputs=[points,target_point,image,mask_show,mask,add_watermark])

    button_drag.click(on_drag, inputs=[model, points, mask, max_step,latent,sample_interval,l_expected,d_max,if_save_video,add_watermark], \
                               outputs=[image, progress, latent, image_clear, images_total, button_stop])
    button_stop.click(change_stop_state)
    
    button_video.click(save_video,inputs=[images_total,frame_rate],outputs=[images_total])
    reset_btn.click(reset_all,inputs=[image_clear,mask,add_watermark],outputs= [points,target_point,image,mask_show,mask,add_watermark]).then(on_show_save)

    button_new.click(new_image, inputs = [model,seed],outputs = [image, image_clear, latent,seed]).then(reset_all,
                                        inputs=[image_clear,mask],outputs=[points,target_point,image,mask_show,mask,add_watermark])

    button_rand.click(new_image, inputs = [model],outputs = [image, image_clear, latent,seed]).then(reset_all,
                                        inputs=[image_clear,mask],outputs=[points,target_point,image,mask_show,mask,add_watermark])

    model_name.change(new_model,inputs=[model_name],outputs=[model,res,l_expected,d_max]).then \
            (new_image, inputs = [model,seed],outputs = [image, image_clear, latent,seed]).then \
            (reset_all,inputs=[image_clear,mask],outputs=[points,target_point,image,mask_show,mask,add_watermark]) 

    imagetab.select(update_mask,[image,mask_show],[image,mask])
    masktab.select(on_select_mask_tab, inputs=[image], outputs=[mask_show])


if __name__ == "__main__":

     demo.queue(concurrency_count=3,max_size=20).launch(share=True)