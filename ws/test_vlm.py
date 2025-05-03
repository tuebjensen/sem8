import time
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

def main():
    model = AutoModel.from_pretrained(
        "nvidia/Eagle2-1B", trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(
        "nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True
    )
    processor.tokenizer.padding_side = "left"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://www.ilankelman.org/stopsigns/australia.jpg",
                },
                {"type": "text", "text": "Translate and write out what all the signs in the image say."},
            ],
        }
    ]

    text_list = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ]
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=text_list,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to("cuda")
    model = model.to("cuda")

    # time the next few lines
    start = time.perf_counter()
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    print("Time taken:", time.perf_counter() - start)

def modify_parser(parser):
    pass

if __name__ == "__main__":
    main()