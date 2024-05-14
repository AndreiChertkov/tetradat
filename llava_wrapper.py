from PIL import Image
from io import BytesIO
import re
import requests
import torch


from llava.constants import IMAGE_TOKEN_INDEX
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.constants import DEFAULT_IM_END_TOKEN
from llava.constants import DEFAULT_IM_START_TOKEN
from llava.constants import IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import process_images
from llava.mm_utils import tokenizer_image_token


class LlavaWrapper:
    def __init__(self, model_path='liuhaotian/llava-v1.5-7b'):
        self.args = type('Args', (), {
            'model_path': model_path,
            'model_base': None,
            'model_name': get_model_name_from_path(model_path),
            'query': prompt,
            'conv_mode': None,
            'image_file': '',
            'sep': ',',
            'temperature': 0,
            'top_p': None,
            'num_beams': 1,
            'max_new_tokens': 512
        })()

        disable_torch_init()

        self.model_name = get_model_name_from_path(self.args.model_path)
        out = load_pretrained_model(
            self.args.model_path, self.args.model_base, self.model_name)
        self.tokenizer, self.model, self.image_processor, context_len = out

    def run(self, prompt, image_file):
        self.args.query = prompt
        self.args.image_file = image_file

        qs = self.args.query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in self.model_name.lower():
            self.args.conv_mode = 'llava_llama_2'
        elif 'mistral' in self.model_name.lower():
            self.args.conv_mode = 'mistral_instruct'
        elif 'v1.6-34b' in self.model_name.lower():
            self.args.conv_mode = 'chatml_direct'
        elif 'v1' in self.model_name.lower():
            self.args.conv_mode = 'llava_v1'
        elif 'mpt' in self.model_name.lower():
            self.args.conv_mode = 'mpt'
        else:
            self.args.conv_mode = 'llava_v0'

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_files = image_parser(self.args)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        return outputs


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out