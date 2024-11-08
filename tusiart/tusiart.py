import json
import logging
import os
import random
import time
import traceback
from datetime import datetime, timedelta
from functools import partial
from uuid import uuid4

import requests
from dotenv import load_dotenv

from signature import generate_signature

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class TensorArtService(object):
    def __init__(self):
        self.app_id = os.environ['TUSIART_APP_ID']
        self.base_url = os.environ['TUSIART_BASE_URL']

    def format_headers(self, method, path, data):
        auth = generate_signature(method.upper(), path, json.dumps(data) if data else data, self.app_id)
        headers = {
            "Authorization": auth
        }
        return headers

    def txt2img(self, model_name, prompt, negative_prompt, width, height, steps, cfg_scale, sampler, *args, **kwargs):
        nonce_str = uuid4().hex
        json_data = {
            "request_id": nonce_str,
            "stages": [
                {
                    "type": "INPUT_INITIALIZE",
                    "inputInitialize": {
                        "seed": -1,
                        "count": 1
                    }
                },
                {
                    "type": "DIFFUSION",
                    "diffusion": {
                        "width": width,
                        "height": height,
                        "prompts": [
                            {
                                "text": prompt
                            }
                        ],
                        "negativePrompts": [
                            {
                                "text": negative_prompt
                            }
                        ],
                        "sampler": sampler,
                        "sdVae": "Automatic",
                        "steps": steps,
                        "sd_model": model_name,
                        "clip_skip": 2,
                        "cfg_scale": cfg_scale,
                        "lora": {
                            "items": [
                                {
                                    "loraModel": "724927343564527198",
                                    "weight": 0.8
                                }
                            ]
                        }
                    }
                }
            ]
        }
        path = '/v1/jobs'
        url = f'{self.base_url}{path}'
        status = 3
        img_url = ''
        task_id = ''
        try:
            headers = self.format_headers('POST', path, json_data)
            response = requests.post(url, json=json_data, headers=headers)
            logging.info(f"txt2img, params: {json_data}, resp: {response.text}")
            resp = response.json()
            task_id = resp['job']['id']
            status, img_url, failed_reason = self.progress(task_id)
        except Exception:
            failed_reason = 'stable diffusion server error'
            fmt = traceback.format_exc()
            logging.warning(f"txt2img error: {fmt}")
        return task_id, status, img_url, failed_reason

    def img2img(self, model_name, prompt_img_url, prompt, negative_prompt, width, height, steps, cfg_scale, sampler,
                *args, **kwargs):
        nonce_str = uuid4().hex
        image_resource_id = self.upload_image(prompt_img_url)
        json_data = {
            "request_id": nonce_str,
            "stages": [
                {
                    "type": "INPUT_INITIALIZE",
                    "inputInitialize": {
                        "image_resource_id": image_resource_id,
                        "seed": random.randint(1, 2147483647),
                        "count": 1
                    }
                },
                {
                    "type": "DIFFUSION",
                    "diffusion": {
                        "width": width,
                        "height": height,
                        "prompts": [
                            {
                                "text": prompt
                            }
                        ],
                        "negativePrompts": [
                            {
                                "text": negative_prompt
                            }
                        ],
                        "sampler": sampler,
                        "steps": steps,
                        "sd_model": model_name,
                        "clip_skip": 2,
                        "cfg_scale": cfg_scale
                    }
                },
                {
                    "type": "IMAGE_TO_UPSCALER",
                    "image_to_upscaler": {
                        "hr_upscaler": "4x-UltraSharp",
                        "hr_scale": 2,
                        "hr_second_pass_steps": 15,
                        "denoising_strength": 0.7
                    }
                }
            ]
        }
        path = '/v1/jobs'
        url = f'{self.base_url}{path}'
        status = 3
        img_url = ''
        task_id = ''
        try:
            headers = self.format_headers('POST', path, json_data)
            response = requests.post(url, json=json_data, headers=headers)
            logging.info(f"img2img, resp: {response.text}, params: {json_data}")
            resp = response.json()
            task_id = resp['job']['id']
            status, img_url, failed_reason = self.progress(task_id)
        except Exception:
            failed_reason = 'stable diffusion server error'
            fmt = traceback.format_exc()
            logging.warning(f"img2img error: {fmt}")
        return task_id, status, img_url, failed_reason

    def txt2gif(self, model_name, prompt, negative_prompt, width, height, steps, cfg_scale, sampler, fps, video_length,
                *args, **kwargs):
        nonce_str = uuid4().hex
        json_data = {
            "request_id": nonce_str,
            "stages": [
                {
                    "type": "INPUT_INITIALIZE",
                    "inputInitialize": {
                        "seed": -1,
                        "count": 1
                    }
                },
                {
                    "type": "DIFFUSION",
                    "diffusion": {
                        "width": width,
                        "height": height,
                        "prompts": [
                            {
                                "text": prompt
                            }
                        ],
                        "negativePrompts": [
                            {
                                "text": negative_prompt
                            }
                        ],
                        "sampler": sampler,
                        "steps": steps,
                        "sd_model": model_name,
                        "clip_skip": 2,
                        "cfg_scale": cfg_scale,
                        "animate_diff": {
                            "args": [{
                                "videoLength": video_length,
                                "fps": fps
                            }]
                        }
                    }
                }
            ]
        }
        path = '/v1/jobs'
        url = f'{self.base_url}{path}'
        status = 3
        img_url = ''
        task_id = ''
        try:
            headers = self.format_headers('POST', path, json_data)
            response = requests.post(url, json=json_data, headers=headers)
            logging.info(f"txt2gif, params: {json_data}, resp: {response.text}")
            resp = response.json()
            task_id = resp['job']['id']
            self.sd_url_to_s3 = partial(self.sd_url_to_s3, use_suffix='gif')
            status, img_url, failed_reason = self.progress(task_id)
        except Exception:
            failed_reason = 'stable diffusion server error'
            fmt = traceback.format_exc()
            logging.warning(f"txt2gif error: {fmt}")

        return task_id, status, img_url, failed_reason

    def upload_image(self, img_url):
        path = '/v1/resource/image'
        url = f'{self.base_url}{path}'
        params = {'term': 1}
        headers = self.format_headers('POST', path, params)
        response = requests.post(url, json=params, headers=headers)
        logging.info(f"upload_image 1, params: {params}, resp: {response.text}")
        data = response.json()
        resource_id = data['resourceId']
        url = data['putUrl']
        headers = data['headers']
        response = requests.get(img_url)
        response = requests.put(url, data=response.content, headers=headers)
        logging.info(f"upload_image 2, params: {params}, resp: {response.text}")
        return resource_id

    def progress(self, task_id):
        status = 3
        img_url = ''
        failed_reason = ''
        sd_status = None
        now = datetime.now()
        five_minutes_dt = now + timedelta(minutes=5)
        path = f'/v1/jobs/{task_id}'
        url = f'{self.base_url}{path}'
        headers = self.format_headers('GET', path, '')
        while now < five_minutes_dt:
            try:
                response = requests.get(url, headers=headers)
                logging.info(f"get_progress, resp: {response.text}")
                resp = response.json()
                sd_status = resp['job']['status']
                if sd_status == 'SUCCESS':
                    status = 2
                    success_info = resp['job']['successInfo']
                    img_url = success_info['videos'][0]['url'] \
                        if 'videos' in success_info \
                        else success_info['images'][0]['url']
                    break
                if sd_status == 'FAILED':
                    failed_reason = resp['job']['failedInfo']['reason']
                    break
                if sd_status == 'CANCELED':
                    failed_reason = 'job has canceled'
                    break
            except Exception:
                pass
            now = datetime.now()
            time.sleep(1)
        if sd_status not in ['SUCCESS', 'FAILED', 'CANCELED']:
            logging.info(f"get_progress timeout, task_id: {task_id}, sd_status: {sd_status}")
            status = 3
            failed_reason = 'request stable diffusion service timeout'
        return status, img_url, failed_reason


if __name__ == '__main__':
    ta = TensorArtService()
    model_name = '731168175858671052'
    prompt = 'Vectorial illustration, A cute fish in the style of lineart, full body, black lines, white background, Vector art, minimalist'
    negative_prompt = ''
    width, height = 512, 512
    steps = 15
    cfg_scale = 7
    sampler = 'DPM++ 2M Karras'
    print(ta.txt2img(model_name, prompt, negative_prompt, width, height, steps, cfg_scale, sampler))
