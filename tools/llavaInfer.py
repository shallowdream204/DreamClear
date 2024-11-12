from PIL import Image
import os
from pathlib import Path
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse

def inferLLaVa_oneImage(img_path, prompt):
    image = Image.open(img_path)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    for k,v in inputs.items():
        print(k,v.shape)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    description = processor.batch_decode(outputs, skip_special_tokens=True)
    description = description[0].split("ASSISTANT:")[-1]
    print(description)


def generate_descriptions_for_directory(image_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in Path(image_dir).glob("*.png"): 
        image = Image.open(image_file)
        prompt = "USER: <image>\nDescribe this image and its style in a very detailed manner.\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        description = processor.batch_decode(outputs, skip_special_tokens=True)
        description = description[0].split("ASSISTANT:")[-1]

        output_file = Path(output_dir) / f"{image_file.stem}.txt"
        with open(output_file, "w") as f:
            f.write(description)
        print(f"Generated description for {image_file.name} saved to {output_file}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Generate image descriptions using Llava model.")
    parser.add_argument('--images_dir', type=str, required=True, help="Directory containing input images.")
    parser.add_argument('--caption_dir', type=str, required=True, help="Directory to save generated captions.")

    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
    
    generate_descriptions_for_directory(args.images_dir, args.caption_dir)
