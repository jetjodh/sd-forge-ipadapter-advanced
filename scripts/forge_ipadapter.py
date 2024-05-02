import numpy as np
from modules_forge.supported_preprocessor import PreprocessorClipVision, Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.forge_util import numpy_to_pytorch
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_ipadapter.IPAdapterPlus import IPAdapterApplyAdvanced
from pathlib import Path
from modules import scripts
from modules_forge.shared import controlnet_dir
import gradio as gr
from modules.api import api


opIPAdapterApplyAdvanced = IPAdapterApplyAdvanced().apply_ipadapter 

def to_base64_nparray(encoding: str):
    """
    Convert a base64 image into the image type the extension uses
    """

    return np.array(api.decode_base64_to_image(encoding)).astype('uint8')

class PreprocessorClipVisionForIPAdapter(PreprocessorClipVision):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.tags = ['IP-Adapter']
        self.model_filename_filters = ['IP-Adapter', 'IP_Adapter']
        self.sorting_priority = 20

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=self.load_clipvision(),
            image=numpy_to_pytorch(input_image),
            weight_type="original",
            noise=0.0,
            embeds=None,
            unfold_batch=False,
        )
        return cond


add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-H (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))

add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-bigG (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))

# class IPAdapterAdvancedPatcher(ControlModelPatcher):
#     @staticmethod
#     def try_build_from_state_dict(state_dict, ckpt_path):
#         model = state_dict

#         if ckpt_path.lower().endswith(".safetensors"):
#             st_model = {"image_proj": {}, "ip_adapter": {}}
#             for key in model.keys():
#                 if key.startswith("image_proj."):
#                     st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
#                 elif key.startswith("ip_adapter."):
#                     st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
#             model = st_model

#         if "ip_adapter" not in model.keys() or len(model["ip_adapter"]) == 0:
#             return None

#         o = IPAdapterAdvancedPatcher(model)

#         model_filename = Path(ckpt_path).name.lower()
#         if 'v2' in model_filename:
#             o.faceid_v2 = True
#             o.weight_v2 = True

#         return o

#     def __init__(self, state_dict):
#         super().__init__()
#         self.ip_adapter = state_dict
#         self.faceid_v2 = False
#         self.weight_v2 = False
#         return

#     def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
#         unet = process.sd_model.forge_objects.unet
#         print(kwargs, flush=True)

#         unet = opIPAdapterApplyAdvanced(
#             ipadapter=self.ip_adapter,
#             model=unet,
#             weight=self.strength,
#             start_at=self.start_percent,
#             end_at=self.end_percent,
#             faceid_v2=self.faceid_v2,
#             weight_v2=self.weight_v2,
#             attn_mask=mask.squeeze(1) if mask is not None else None,
#             weight_type=kwargs["weight_type"],
#             image_style=kwargs["image_style"],
#             image_composition=kwargs["image_composition"],  # Assuming no specific image composition is passed; adjust as necessary
#             image_negative=kwargs["image_negative"],
#             weight_composition=kwargs["weight_composition"],  # Default or adjust as necessary
#             combine_embeds=kwargs["combine_embeds"],
#             embeds_scaling=kwargs["embeds_scaling"],
#             layer_weights=kwargs["layer_weights"],
#             **cond,
#         )[0]

#         process.sd_model.forge_objects.unet = unet
#         return


class IPAdapterAdvancedForge(scripts.Script):
    def title(self):
        return "IP-Adapter Advanced"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            image_style = gr.Image(source='upload', type='numpy')
            image_composition = gr.Image(source='upload', type='numpy')
            image_negative = gr.Slider(label='Image Negative', minimum=0.0, maximum=1.0, value=0.0)
            weight_type = gr.Dropdown(label='Weight Type', choices=['original', 'v2'], value='original')
            weight_composition = gr.Slider(label='Weight Composition', minimum=0.0, maximum=1.0, value=0.5)
            combine_embeds = gr.Checkbox(label='Combine Embeds', value=False)
            embeds_scaling = gr.Slider(label='Embeds Scaling', minimum=0.0, maximum=1.0, value=0.5)
            layer_weights = gr.Textbox(label='Layer Weights', value='')

        return weight_type, image_style, image_composition, image_negative, weight_composition, combine_embeds, embeds_scaling, layer_weights

    # def process(self, p, *script_args, **kwargs):
    #     input_image, weight_type, image_style, image_composition, image_negative, weight_composition, combine_embeds, embeds_scaling, layer_weights = script_args

    #     if input_image is None:
    #         return

    #     ipadapter_path = load_file_from_url(
    #         url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    #         model_dir=controlnet_dir,
    #         file_name='CLIP-ViT-H-14.safetensors'
    #     )

    #     self.model = IPAdapterAdvancedPatcher(ipadapter_path)
    #     print('IP-Adapter loaded.')

    #     return
    
    def process_before_every_sampling(self, process, cond, mask, *script_args, **kwargs):
        weight_type, image_style, image_composition, image_negative, weight_composition, combine_embeds, embeds_scaling, layer_weights = script_args



        unet = process.sd_model.forge_objects.unet
        unet = opIPAdapterApplyAdvanced(
            ipadapter=self.ip_adapter,
            model=unet,
            weight=self.strength,
            start_at=self.start_percent,
            end_at=self.end_percent,
            faceid_v2=self.faceid_v2,
            weight_v2=self.weight_v2,
            attn_mask=mask.squeeze(1) if mask is not None else None,
            weight_type=weight_type,
            image_style=to_base64_nparray(image_style),
            image_composition=to_base64_nparray(image_composition),  # Assuming no specific image composition is passed; adjust as necessary
            image_negative=to_base64_nparray(image_negative),
            weight_composition=weight_composition,  # Default or adjust as necessary
            combine_embeds=combine_embeds,
            embeds_scaling=embeds_scaling,
            layer_weights=layer_weights,
            **cond,
        )[0]

        process.sd_model.forge_objects.unet = unet
        return

