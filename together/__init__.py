import base64

from together import Together

client = Together()


def run(prompt):
    response = client.images.generate(
        prompt=f"[{prompt}, line art, laser engraving]",
        model="black-forest-labs/FLUX.1-schnell-Free",
        width=768,
        height=768,
        steps=4,
        n=1,
        response_format="b64_json",
        update_at="2024-11-21T13:24:29.995Z"
    )
    b64_data = response.data[0].b64_json
    imgdata = base64.b64decode(b64_data)
    # filename = '/Users/zhangbin/Documents/ready2print/1.jpg'
    filename = '1.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)


if __name__ == '__main__':
    run('flying dog')
