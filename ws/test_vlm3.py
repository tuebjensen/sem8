import time

import PIL
import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from eagle2_hg_model.inference_eagle_repo import EagleProcessor

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
    config = AutoConfig.from_pretrained("eagle2_hg_model", trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    reduce_model(model)
    model = model.to("cuda")
    # This is in accordance with https://huggingface.co/nvidia/GR00T-N1-2B/blob/main/config.json
    processor = EagleProcessor(
        model_path="eagle2_hg_model", max_input_tiles=1, model_spec=None
    )
    img_context_token_id = processor.get_img_context_token()

    model.language_model.requires_grad_(False)
    model.vision_model.requires_grad_(False)
    model.mlp1.requires_grad_(False)
    model.language_model.eval()
    model.vision_model.eval()
    model.mlp1.eval()

    if (
        hasattr(model, "vision_model")
        and hasattr(model.vision_model, "vision_model")
        and hasattr(model.vision_model.vision_model, "vision_towers")
        and len(model.vision_model.vision_model.vision_towers) > 1
    ):
        vision_towers = model.vision_model.vision_model.vision_towers

        if (
            hasattr(vision_towers[0], "vision_tower")
            and hasattr(vision_towers[0].vision_tower, "vision_model")
            and hasattr(vision_towers[0].vision_tower.vision_model, "encoder")
        ):
            vision_towers[
                0
            ].vision_tower.vision_model.encoder.gradient_checkpointing = False
            vision_towers[0].vision_tower.vision_model.head = torch.nn.Identity()

        if hasattr(vision_towers[1], "vision_tower"):
            vision_towers[1].vision_tower.head = torch.nn.Identity()

    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/640px-Red_Apple.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/be/2022-09-24_Motorsport%2C_IDM%2C_Finale_Hockenheimring_1DX_3890_by_Stepro.jpg",
        "https://www.ilankelman.org/stopsigns/australia.jpg",
        "https://images.unsplash.com/photo-1746105752085-21b6bf3ab0fc?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_photo-1713200811001-af93d0dcdfc2?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1746105752085-21b6bf3ab0fc?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1743275532243-ee04fdcd3899?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1742522314620-a41c790acbdb?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_photo-1666700698946-fbf7baa0134a?q=80&w=1936&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_photo-1673438132288-e8787f60ccb9?q=80&w=2069&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    ]

    for url in urls:
        message = [
            {
                "role": "system",
                "content": "You are a helpful assistant that explains the content of images.",
            },
            {
                "role": "user",
                "image": [{"url": url}],
                "content": "Explain the image.",
            },
        ]
        try:
            inputs = processor.prepare_input({"prompt": message})
        except PIL.UnidentifiedImageError as e:
            print(f"Error fetching image from {url}")
            continue

        inputs = BatchFeature(inputs).to("cuda")

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
        # print(output)


def modify_parser(parser):
    pass


if __name__ == "__main__":
    main()
