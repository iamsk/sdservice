from tusiart import TensorArtService

animal = 'dragon'

ta = TensorArtService()
model_name = '619188908049212188'
prompt = f'Chinese_zodiac, {animal}, Chinese zodiac, simple drawing, One stroke of painting, a line art, black lines, white background, desert_sky, Vector art, minimalist'
negative_prompt = 'EasyNegative,FastNegativeV2,bad-artist-anime,bad-hands-5,lowres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,jpeg artifacts,signature,watermark,username,blurry,out of focus,censorship,Missing vagina,Blurry faces,Blank faces,bad face,Ugly,extra ear,amputee,missing hands,missing arms,missing legs,Extra fingers,6 fingers,Extra feet,Missing nipples,ghost,futanari,Extra legs,Extra hands,panties,pants,(painting by bad-artist-anime:0.9),(painting by bad-artist:0.9),text,error,blurry,jpeg artifacts,cropped,normal quality,artist name,(worst quality, low quality:1.4),twisted_hands,fused_fingers,Face Shadow,bad-image-v2-39000-neg,bhands-neg galaxybad_embedding,'
width, height = 512, 512
steps = 20
cfg_scale = 3.9
sampler = 'DPM++ 2M SDE Karras'
print(ta.txt2img(model_name, prompt, negative_prompt, width, height, steps, cfg_scale, sampler))
