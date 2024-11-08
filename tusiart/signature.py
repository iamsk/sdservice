import base64
import hashlib
import json
import os
import time
from uuid import uuid4

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv


def generate_signature(method, url, body, app_id):
    method_str = method.upper()
    url_str = url
    timestamp = str(int(time.time()))
    nonce_str = hashlib.md5(timestamp.encode()).hexdigest()
    body_str = body
    to_sign = f"{method_str}\n{url_str}\n{timestamp}\n{nonce_str}\n{body_str}"
    current_path = os.path.dirname(__file__)
    private_key_path = current_path + '/tusiart_private_key.pem'
    with open(private_key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )
    signature = private_key.sign(
        to_sign.encode(),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    signature_base64 = base64.b64encode(signature).decode()
    auth_header = f"TAMS-SHA256-RSA app_id={app_id},nonce_str={nonce_str},timestamp={timestamp},signature={signature_base64}"
    return auth_header


if __name__ == '__main__':
    load_dotenv()
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
                    "width": 512,
                    "height": 768,
                    "prompts": [
                        {
                            "text": "1girl"
                        }
                    ],
                    "steps": 15,
                    "sd_model": "600423083519508503",
                    "clip_skip": 2,
                    "cfg_scale": 7
                }
            },
            {
                "type": "IMAGE_TO_UPSCALER",
                "image_to_upscaler": {
                    "hr_upscaler": "Latent",
                    "hr_scale": 2,
                    "hr_second_pass_steps": 10,
                    "denoising_strength": 0.3
                }
            }
        ]
    }
    auth = generate_signature('POST', '/v1/jobs', json.dumps(json_data), os.environ['TUSIART_APP_ID'])
    print(auth)
