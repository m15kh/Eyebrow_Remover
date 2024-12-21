import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import cv2

def remover(img, mask, prompt="removes all black pixels and replaces them with the content of the image", negative_prompt="adds any black pixels"):

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    pipeline.enable_xformers_memory_efficient_attention()

    init_image = load_image(img)
    mask_image = load_image(mask)

    result_image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
    return result_image
