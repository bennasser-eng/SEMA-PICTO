from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.unet.load_adapter("./lora_picto_model/final")

prompts = [
    "un chien qui court, style arasaac",
    "une voiture rouge, style arasaac",
    "un enfant qui lit, style arasaac"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=30).images[0]
    image.save(f"test_picto_{i}.png")
    print(f"Image sauvegardee: test_picto_{i}.png")
