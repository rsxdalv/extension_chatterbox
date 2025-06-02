import os
import functools
import gradio as gr
from contextlib import contextmanager
import torch
from chatterbox.tts import ChatterboxTTS

from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.decorators import *
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.randomize_seed import randomize_seed_ui
from tts_webui.utils.OpenFolderButton import OpenFolderButton
from tts_webui.utils.get_path_from_root import get_path_from_root
from tts_webui.utils.get_path_from_root import get_path_from_root
from gradio_iconbutton import IconButton


def chatterbox_to(model: ChatterboxTTS, device, dtype):
    # model.ve.to(device=device, dtype=dtype)
    model.ve.to(device=device)
    model.t3.to(device=device, dtype=dtype)
    model.s3gen.to(device=device, dtype=dtype)
    model.conds.to(device=device)
    model.device = device
    torch.cuda.empty_cache()

    return model


@manage_model_state("chatterbox")
def get_model(
    model_name="just_a_placeholder", device=torch.device("cuda"), dtype=torch.float32
):
    model = ChatterboxTTS.from_pretrained(device=device)
    chatterbox_to(model, device, dtype)
    return model


@contextmanager
def chatterbox_model(model_name, device="cuda", dtype=torch.float32):
    model = get_model(
        # model_name="just_a_placeholder" + str(device) + str(dtype),
        # pretty name
        model_name=f"Chatterbox on {device} with {dtype}",
        device=torch.device(device),
    )
    # chatterbox_to(model, torch.device(device), dtype)

    use_autocast = dtype in [torch.float16, torch.bfloat16]

    with (
        torch.autocast(device_type=device, dtype=dtype)
        if use_autocast
        else torch.no_grad()
    ):
        yield model


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def tts(
    text,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    audio_prompt_path=None,
    # model
    model_name="just_a_placeholder",
    device="cuda",
    dtype="float32",
    # hyperparameters
    chunked=False,
    seed=-1,  # for signature compatibility
    **kwargs,
):
    device = get_best_device() if device == "auto" else device
    print(f"Using device: {device}")

    try:
        with chatterbox_model(
            model_name=model_name,
            device=device,
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[dtype],
        ) as model:

            def generate_chunk(text):
                print(f"Generating chunk: {text}")
                return model.generate(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )

            if chunked:
                from tts_webui.utils.split_text_functions import (
                    split_and_recombine_text,
                )

                texts = split_and_recombine_text(text)
                wavs = [generate_chunk(text) for text in texts]
                wav = torch.cat([w.squeeze() for w in wavs], dim=0)
                return {
                    "audio_out": (model.sr, wav.cpu().numpy()),
                }

            wav = generate_chunk(text)

        return {
            "audio_out": (model.sr, wav.cpu().numpy().squeeze()),
        }
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise gr.Error(f"Error: {e}")


@functools.wraps(tts)
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("chatterbox")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def tts_decorated(*args, _type=None, **kwargs):
    return tts(*args, **kwargs)


def get_voices():
    voices_dir = get_path_from_root("voices")
    results = ["select a voice to load it"] + [
        (x, os.path.join(voices_dir, x))
        for x in os.listdir(voices_dir)
        if x.endswith(".wav")
    ]
    return results


def ui():
    gr.HTML(
        """
  <h2 style="text-align: center;">Chatterbox TTS</h2>

  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div style="flex: 1; min-width: 300px;">
      <h2>Model sizes:</h2>
      <ul style="list-style-type: disc; padding-left: 20px; margin-top: 0;">
        <li><strong>t3_cfg.safetensors</strong> 2.13 GB</li>
        <li><strong>s3gen.safetensors</strong> 1.06 GB</li>
      </ul>
    </div>

    <div style="flex: 1; min-width: 300px;">
      <h2>Performance</h2>
      <p style="margin-top: 0;">
        During sampling on an <strong>NVIDIA RTX 3090</strong>, the generation process achieved the following results for a 7-second audio clip:
      </p>
      <ul style="list-style-type: disc; padding-left: 20px;">
        <li><strong>VRAM</strong>: ~5GB</li>
        <li><strong>Speed</strong>: ~32.04 iterations per second</li>
        <li><strong>Elapsed Time</strong>: ~5 seconds</li>
      </ul>
    </div>
  </div>
                """
    )

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Text")
            btn = gr.Button("Generate", variant="primary")
            chunked = gr.Checkbox(label="Split prompt into chunks", value=False)
            exaggeration = gr.Slider(
                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
                minimum=0,
                maximum=2,
                value=0.5,
            )
            cfg_weight = gr.Slider(
                label="CFG Weight/Pace", minimum=0.0, maximum=1, value=0.5
            )
            temperature = gr.Slider(
                label="Temperature", minimum=0.05, maximum=5, value=0.8
            )

            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    label="Audio Prompt",
                )
                with gr.Column():
                    IconButton("refresh").click(
                        fn=lambda: gr.Dropdown(choices=get_voices()),
                        outputs=[voice_dropdown],
                    )
                    voices_dir = get_path_from_root("voices")
                    OpenFolderButton(voices_dir, api_name="tortoise_open_voices")

            audio_prompt_path = gr.Audio(
                label="Reference Audio", type="filepath", value=None
            )

            voice_dropdown.change(
                lambda x: gr.Audio(value=x),
                inputs=[voice_dropdown],
                outputs=[audio_prompt_path],
            )
            seed, randomize_seed_callback = randomize_seed_ui()

            # model
            with gr.Accordion("Advanced", open=False):
                device = gr.Radio(
                    label="Device",
                    choices=["auto", "cuda", "mps", "cpu"],
                    value="auto",
                )
                dtype = gr.Radio(
                    label="Dtype",
                    # choices=["float32", "float16", "bfloat16"],
                    choices=["float32"],
                    value="float32",
                )
                model_name = gr.Dropdown(
                    label="Model",
                    choices=["just_a_placeholder"],
                    value="just_a_placeholder",
                    visible=False,
                )
                unload_model_button("chatterbox")

        with gr.Column():
            audio_out = gr.Audio(label="Audio Output")

    btn.click(
        **randomize_seed_callback,
    ).then(
        **dictionarize_wraps(
            tts_decorated,
            inputs={
                text: "text",
                exaggeration: "exaggeration",
                cfg_weight: "cfg_weight",
                temperature: "temperature",
                audio_prompt_path: "audio_prompt_path",
                seed: "seed",
                # model
                device: "device",
                dtype: "dtype",
                model_name: "model_name",
                # hyperparameters
                chunked: "chunked",
            },
            outputs={
                "audio_out": audio_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
            api_name="chatterbox_tts",
        )
    )


if __name__ == "__main__":
    if "demo" in locals():
        demo.close()
    with gr.Blocks() as demo:
        ui()

    demo.launch(
        server_port=7770,
    )
    # python -m workspace.extension_chatterbox.extension_chatterbox.gradio_app
