import functools
import gradio as gr

from chatterbox.tts import ChatterboxTTS
import numpy as np

from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.decorators import *
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.randomize_seed import randomize_seed_ui


@manage_model_state("chatterbox")
def get_model(model_name="just_a_placeholder", device="cuda"):
    return ChatterboxTTS.from_pretrained(device=device)


def tts(
    text,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    audio_prompt_path=None,
    **kwargs
):
    model = get_model(model_name="just_a_placeholder")
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    return {
        "audio_out": (model.sr, wav.cpu().numpy().squeeze()),
    }


@functools.wraps(tts)
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("kokoro")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def tts_decorated(*args, _type=None, **kwargs):
    return tts(*args, **kwargs)


def ui():
    gr.HTML(
        """
  <h2 style="text-align: center;">Chatterbox TTS</h2>

  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div style="flex: 1; min-width: 300px;">
      <h2>Model sizes:</h2>
      <ul style="list-style-type: disc; padding-left: 20px; margin-top: 0;">
        <li><strong>t3_cfg.pt</strong> 1.06 GB</li>
        <li><strong>s3gen.pt</strong> 1.06 GB</li>
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
            btn = gr.Button("Generate")
            exaggeration = gr.Slider(
                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", minimum=0, maximum=2, value=0.5
            )
            cfg_weight = gr.Slider(label="CFG Weight/Pace", minimum=0.2, maximum=1, value=0.5)
            temperature = gr.Slider(
                label="Temperature", minimum=0.05, maximum=5, value=0.8
            )
            audio_prompt_path = gr.Audio(label="Reference Audio", type="filepath")
            seed, randomize_seed_callback = randomize_seed_ui()

        with gr.Column():
            audio_out = gr.Audio(label="Audio Output")
            unload_model_button("chatterbox")

    btn.click(
        **randomize_seed_callback,
    ).then(
        **dictionarize(
            tts_decorated,
            inputs={
                text: "text",
                exaggeration: "exaggeration",
                cfg_weight: "cfg_weight",
                temperature: "temperature",
                audio_prompt_path: "audio_prompt_path",
            },
            outputs={
                "audio_out": audio_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
        )
    )


if __name__ == "__main__":
    if "demo" in locals():
        demo.close()
    with gr.Blocks() as demo:
        tts_ui()

    demo.launch()
