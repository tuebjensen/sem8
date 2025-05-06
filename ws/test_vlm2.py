import time

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

select_layer = 12


def reduce_model(model):
    model.neftune_alpha = None

    # reduce vision model (Siglip)
    if hasattr(model.vision_model, "vision_model") and hasattr(
        model.vision_model.vision_model, "head"
    ):
        model.vision_model.vision_model.head = torch.nn.Identity()

    # remove LLM (Qwen)
    model.language_model.lm_head = torch.nn.Identity()
    while len(model.language_model.model.layers) > select_layer:
        model.language_model.model.layers.pop(-1)

    # Gr00t also removes model.vision_model.vision_model.vision_towers, but we
    # don't have that


def get_embeddings(
    model,
    reproject_vision: bool,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    visual_features=None,
    output_hidden_states=None,
    skip_llm=False,
    img_context_token_id=None,
) -> torch.LongTensor:
    assert pixel_values is not None
    assert img_context_token_id is not None

    vit_embeds = model.extract_feature(pixel_values)

    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = input_ids == img_context_token_id
    assert selected.sum() != 0

    embeds_to_scatter = vit_embeds.reshape(-1, C).to(
        input_embeds.device, input_embeds.dtype
    )
    input_embeds[selected] = embeds_to_scatter
    input_embeds = input_embeds.reshape(B, N, C)

    # return hidden_states
    embeddings = model.language_model.forward(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    embeddings = embeddings.hidden_states[-1]

    return embeddings


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
                {
                    "type": "text",
                    "text": "Translate and write out what all the signs in the image say.",
                },
            ],
        }
    ]

    text_list = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    ]
    image_inputs, video_inputs = processor.process_vision_info(messages)
    print(messages)
    inputs = processor(
        text=text_list,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to("cuda")
    model = model.to("cuda")
    img_context_token_id = processor.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    reduce_model(model)

    # time the next few lines
    start = time.perf_counter()
    with torch.no_grad():
        output = get_embeddings(
            model,
            reproject_vision=False,
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            img_context_token_id=img_context_token_id,
        )
    print("Time taken:", time.perf_counter() - start)
    print(output)

    # time twice to see if hot start helps
    start = time.perf_counter()
    with torch.no_grad():
        output = get_embeddings(
            model,
            reproject_vision=False,
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            img_context_token_id=img_context_token_id,
        )
    print("Time taken:", time.perf_counter() - start)
    print(output)


def modify_parser(parser):
    pass


if __name__ == "__main__":
    main()
