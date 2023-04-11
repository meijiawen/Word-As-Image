import gradio as gr
import os
import argparse
from easydict import EasyDict as edict
import yaml
import os.path as osp
import random
import numpy.random as npr
import sys

path = '/home/xlab_app_center/app/code'
os.makedirs(path)
sys.path.append('/home/xlab_app_center/app/code')

# set up diffvg
os.system('git clone https://github.com/BachiLi/diffvg.git')
os.chdir('diffvg')
os.system('git submodule update --init --recursive')
os.system('python setup.py install --user')
sys.path.append("/home/xlab_app_center/.local/lib/python3.8/site-packages/diffvg-0.0.1-py3.8-linux-x86_64.egg")

os.chdir('/home/xlab_app_center/app')

import torch
from diffusers import StableDiffusionPipeline


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                               torch_dtype=torch.float16, use_auth_token=os.environ['HF_TOKEN']).to(device)

from typing import Mapping
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses import SDSLoss, ToneLoss, ConformalLoss
from utils import (
    edict_2_dict,
    update,
    check_and_create_dir,
    get_data_augs,
    save_image,
    preprocess,
    learning_rate_decay,
    combine_word)
import warnings

TITLE="""<h1 style="font-size: 42px;" align="center">Word-As-Image for Semantic Typography</h1>"""
DESCRIPTION="""A demo for [Word-As-Image for Semantic Typography](https://wordasimage.github.io/Word-As-Image-Page/). By using Word-as-Image, a visual representation of the meaning of the word is created while maintaining legibility of the text and font style. 
Please select a semantic concept word and a letter you wish to generate, it will take ~5 minutes to perform 500 iterations."""

DESCRIPTION += '\n<p>This demo is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"> Creative Commons Attribution-ShareAlike 4.0 International License</a>.</p>'

if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'


warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0


def set_config(semantic_concept, word, letter, font_name, num_steps):
    
    cfg_d = edict()
    cfg_d.config = "code/config/base.yaml"
    cfg_d.experiment = "demo"

    with open(cfg_d.config, 'r') as f:
        cfg_full = yaml.load(f, Loader=yaml.FullLoader)

    cfg_key = cfg_d.experiment
    cfgs = [cfg_d]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get('parent_config', 'baseline')
  
    cfg = edict()
    for options in reversed(cfgs):
        update(cfg, options)
    del cfgs

    cfg.semantic_concept = semantic_concept
    cfg.word = word
    cfg.optimized_letter = letter
    cfg.font = font_name
    cfg.seed = 0
    cfg.num_iter = num_steps
    
    if ' ' in cfg.word:
        raise gr.Error(f'should be only one word')
    cfg.caption = f"a {cfg.semantic_concept}. {cfg.prompt_suffix}"
    cfg.log_dir = f"output/{cfg.experiment}_{cfg.word}"
    if cfg.optimized_letter in cfg.word:
        cfg.optimized_letter = cfg.optimized_letter
    else:
        raise gr.Error(f'letter should be in word')

    cfg.letter = f"{cfg.font}_{cfg.optimized_letter}_scaled"
    cfg.target = f"code/data/init/{cfg.letter}"

    # set experiment dir
    signature = f"{cfg.letter}_concept_{cfg.semantic_concept}_seed_{cfg.seed}"
    cfg.experiment_dir = \
        osp.join(cfg.log_dir, cfg.font, signature)
    configfile = osp.join(cfg.experiment_dir, 'config.yaml')

    # create experiment dir and save config
    check_and_create_dir(configfile)
    with open(osp.join(configfile), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = False
    else:
        assert False
    return cfg


def init_shapes(svg_path, trainable: Mapping[str, bool]):
    svg = f'{svg_path}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)

    parameters = edict()

    # path points
    if trainable.point:
        parameters.point = []
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)

    return shapes_init, shape_groups_init, parameters


def run_main_ex(semantic_concept, word, letter, font_name, num_steps):
    return list(next(run_main_app(semantic_concept, word, letter, font_name, num_steps, 1)))
                        
def run_main_app(semantic_concept, word, letter, font_name, num_steps, example=0):
    
    cfg = set_config(semantic_concept, word, letter, font_name, num_steps)

    pydiffvg.set_use_gpu(torch.cuda.is_available())

    print("preprocessing")
    preprocess(cfg.font, cfg.word, cfg.optimized_letter, cfg.level_of_cc)
    filename_init = os.path.join("code/data/init/", f"{cfg.font}_{cfg.word}_scaled.svg").replace(" ", "_")
    if not example:
        yield gr.update(value=filename_init,visible=True),gr.update(visible=False),gr.update(visible=False)

    sds_loss = SDSLoss(cfg, device, model)

    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(svg_path=cfg.target, trainable=cfg.trainable)

    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
               torch.ones(img_init.shape[0], img_init.shape[1], 3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]

    tone_loss = ToneLoss(cfg)
    tone_loss.set_image_init(img_init)

    num_iter = cfg.num_iter
    pg = [{'params': parameters["point"], 'lr': cfg.lr_base["point"]}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    conformal_loss = ConformalLoss(parameters, device, cfg.optimized_letter, shape_groups)

    lr_lambda = lambda step: learning_rate_decay(step, cfg.lr.lr_init, cfg.lr.lr_final, num_iter,
                                                 lr_delay_steps=cfg.lr.lr_delay_steps,
                                                 lr_delay_mult=cfg.lr.lr_delay_mult) / cfg.lr.lr_init

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)  # lr.base * lrlambda_f

    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    for step in t_range:
        optim.zero_grad()

        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (
                    1 - img[:, :, 3:4])
        img = img[:, :, :3]

        filename = os.path.join(
            cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)
        if not example:
            yield gr.update(visible=True),gr.update(value=filename, label=f'iters: {step} / {num_iter}', visible=True),gr.update(visible=False)

        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        x_aug = data_augs.forward(x)

        # compute diffusion loss per pixel
        loss = sds_loss(x_aug)

        tone_loss_res = tone_loss(x, step)
        loss = loss + tone_loss_res

        loss_angles = conformal_loss()
        loss_angles = cfg.loss.conformal.angeles_w * loss_angles
        loss = loss + loss_angles

        loss.backward()
        optim.step()
        scheduler.step()

            
    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)

    combine_word(cfg.word, cfg.optimized_letter, cfg.font, cfg.experiment_dir)

    image = os.path.join(cfg.experiment_dir,f"{cfg.font}_{cfg.word}_{cfg.optimized_letter}.svg")
    yield gr.update(value=filename_init,visible=True),gr.update(visible=False),gr.update(value=image,visible=True)
 

with gr.Blocks() as demo:

    gr.HTML(TITLE)
    gr.Markdown(DESCRIPTION)
    
    with gr.Row():
        with gr.Column():

            semantic_concept = gr.Text(
                label='Semantic Concept',
                max_lines=1,
                placeholder=
                'Enter a semantic concept. For example: BUNNY'
            )

            word = gr.Text(
                label='Word',
                max_lines=1,
                placeholder=
                'Enter a word. For example: BUNNY'
            )
           
            letter = gr.Text(
                label='Letter',
                max_lines=1,
                placeholder=
                'Choose a letter in the word to optimize. For example: Y'
            )

            num_steps = gr.Slider(label='Optimization Iterations',
                      minimum=0,
                      maximum=500,
                      step=10,
                      value=500)
            
            font_name = gr.Text(value=None,visible=False,label="Font Name")
            gallery = gr.Gallery(value=[(os.path.join("images","KaushanScript-Regular.png"),"KaushanScript-Regular"), (os.path.join("images","IndieFlower-Regular.png"),"IndieFlower-Regular"),(os.path.join("images","Quicksand.png"),"Quicksand"),
                                       (os.path.join("images","Saira-Regular.png"),"Saira-Regular"), (os.path.join("images","LuckiestGuy-Regular.png"),"LuckiestGuy-Regular"),(os.path.join("images","DeliusUnicase-Regular.png"),"DeliusUnicase-Regular"),
                                       (os.path.join("images","Noteworthy-Bold.png"),"Noteworthy-Bold"), (os.path.join("images","HobeauxRococeaux-Sherman.png"),"HobeauxRococeaux-Sherman")],label="Font Name").style(grid=4)
            
            def on_select(evt: gr.SelectData):
                return evt.value
                
            gallery.select(fn=on_select, inputs=None, outputs=font_name)
          
            run = gr.Button('Generate')

        with gr.Column():
            result0 = gr.Image(type="filepath", label="Initial Word").style(height=333)
            result1 = gr.Image(type="filepath", label="Optimization Process").style(height=110)
            result2 = gr.Image(type="filepath", label="Final Result",visible=False).style(height=333)
            
        
    with gr.Row():
        # examples
        examples = [
            [
            "BUNNY",
            "BUNNY",
            "Y",
            "KaushanScript-Regular",
            500
            ],
            [
            "LION",
            "LION",
            "O",
            "Quicksand",
            500
            ],
            [
            "FROG",
            "FROG",
            "G",
            "IndieFlower-Regular",
            500
            ],
            [
            "CAT",
            "CAT",
            "C",
            "LuckiestGuy-Regular",
            500
            ],
        ]
        demo.queue(max_size=10, concurrency_count=2)
        gr.Examples(examples=examples,
                inputs=[
                    semantic_concept,
                    word,
                    letter, 
                    font_name,
                    num_steps
                ],
                outputs=[
                    result0,
                    result1,
                    result2
                ],
                fn=run_main_ex,
                cache_examples=True)
        
        
    # inputs
    inputs = [
        semantic_concept,
        word,
        letter, 
        font_name,
        num_steps
    ]

    outputs = [
        result0,
        result1,
        result2
    ]
    
    run.click(fn=run_main_app, inputs=inputs, outputs=outputs, queue=True)


demo.launch(share=False)
