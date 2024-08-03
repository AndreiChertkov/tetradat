import PIL
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
import torchvision


class StyleWrapper:
    def __init__(self, device):
        self.device = device

        model_id = 'timbrooks/instruct-pix2pix'
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix',
            torch_dtype=torch.float16, safety_checker=None)
        self.pipe.to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config)

    def run(self, image, prompt):
        result = self.pipe(prompt, image=image,
            num_inference_steps=10, image_guidance_scale=1)
        img = result.images[0]
        img = torchvision.transforms.ToTensor()(img)
        return img