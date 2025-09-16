import functools
import gradio as gr
from gradio_iconbutton import IconButton

from tts_webui.decorators import *
from tts_webui.decorators.decorator_save_wav import (
    decorator_save_wav_generator_accumulated,
)
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
    decorator_extension_inner_generator,
    decorator_extension_outer_generator,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.randomize_seed import randomize_seed_ui
from tts_webui.utils.OpenFolderButton import OpenFolderButton
from tts_webui.utils.get_path_from_root import get_path_from_root

from .api import (
    move_model_to_device_and_dtype,
    tts,
    tts_stream,
    interrupt,
    get_voices,
    vc,
)
from .memory import get_chatterbox_memory_usage
from .SUPPORTED_LANGUAGES import SUPPORTED_LANGUAGES

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


@functools.wraps(tts_stream)
# @decorator_convert_audio_output_generator  # <-- This goes first/top
@decorator_extension_outer_generator
@decorator_apply_torch_seed_generator
@decorator_save_metadata_generator
@decorator_save_wav_generator_accumulated
@decorator_add_model_type_generator("chatterbox")
@decorator_add_base_filename_generator_accumulated
@decorator_add_date_generator
@decorator_log_generation_generator
@decorator_extension_inner_generator
@log_generator_time
def tts_generator_decorated(*args, **kwargs):
    yield from tts_stream(*args, **kwargs)


def ui():
    gr.HTML(
        """
  <h2 style="text-align: center;">Chatterbox TTS</h2>

  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div>
        <strong>Storage Size:</strong> 3.2 GB
    </div>
    <div>
        <strong>VRAM Usage</strong>: Float32: 7 GB, Bfloat16: 4 GB, CPU Offloading Passive VRAM: 0.7 GB
    </div>
  </div>
                """
    )
    with gr.Tabs():
        with gr.Tab("TTS"):
            with gr.Row():
                chatterbox_tts()
        with gr.Tab("Voice Conversion"):
            with gr.Row():
                chatterbox_vc()


def chatterbox_tts():
    with gr.Column():
        text = gr.Textbox(label="Text", lines=3)
        with gr.Row():
            btn_interrupt = gr.Button("Interrupt next chunk", interactive=False)
            btn_stream = gr.Button("Streaming generation", variant="secondary")
            btn = gr.Button("Generate", variant="primary")
        btn_interrupt.click(
            fn=lambda: gr.Button("Interrupting..."),
            outputs=[btn_interrupt],
        ).then(fn=interrupt, outputs=[btn_interrupt])

        with gr.Row():
            voice_dropdown = gr.Dropdown(
                label="Saved voices", choices=["refresh to load the voices"]
            )
            IconButton("refresh").click(
                fn=lambda: gr.Dropdown(choices=get_voices()),
                outputs=[voice_dropdown],
            )
            OpenFolderButton(
                get_path_from_root("voices", "chatterbox"),
                api_name="chatterbox_open_voices_dir",
            )

        audio_prompt_path = gr.Audio(
            label="Reference Audio", type="filepath", value=None
        )

        voice_dropdown.change(
            lambda x: gr.Audio(value=x),
            inputs=[voice_dropdown],
            outputs=[audio_prompt_path],
        )
        with gr.Row():
            model_name = gr.Radio(
                label="Model",
                choices=[
                    ("English", "just_a_placeholder"),
                    ("Multilingual", "multilingual"),
                ],
                value="just_a_placeholder",
                
            )
            language_id = gr.Dropdown(
                label="Language (Multilingual)",
                choices=[(name, id) for id, name in SUPPORTED_LANGUAGES.items()],
                value="en",
            )
        exaggeration = gr.Slider(
            label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
            minimum=0,
            maximum=2,
            value=0.5,
        )
        cfg_weight = gr.Slider(
            label="CFG Weight/Pace", minimum=0.0, maximum=1, value=0.5
        )
        temperature = gr.Slider(label="Temperature", minimum=0.05, maximum=5, value=0.8)

        seed, randomize_seed_callback = randomize_seed_ui()

    with gr.Column():
        audio_out = gr.Audio(label="Audio Output")
        streaming_audio_output = gr.Audio(
            label="Audio Output (streaming)", streaming=True, autoplay=True
        )

        gr.Markdown("## Settings")

        with gr.Accordion("Chunking", open=True), gr.Group():
            with gr.Row():
                chunked = gr.Checkbox(label="Split prompt into chunks", value=False)
                halve_first_chunk = gr.Checkbox(
                    label="Halve first chunk size",
                    value=False,
                )
            with gr.Row():
                desired_length = gr.Slider(
                    label="Desired length (characters)",
                    minimum=10,
                    maximum=1000,
                    value=200,
                    step=1,
                )
                max_length = gr.Slider(
                    label="Max length (characters)",
                    minimum=10,
                    maximum=1000,
                    value=300,
                    step=1,
                )
                cache_voice = gr.Checkbox(
                    label="Cache voice (not implemented)",
                    value=False,
                    visible=False,
                )
        # model
        with gr.Accordion("Model", open=False):
            with gr.Row():
                device = gr.Radio(
                    label="Device",
                    choices=["auto", "cuda", "mps", "cpu"],
                    value="auto",
                )
                dtype = gr.Radio(
                    label="Dtype",
                    choices=["float32", "float16", "bfloat16"],
                    value="bfloat16",
                )
            with gr.Row():
                cpu_offload = gr.Checkbox(label="CPU Offload", value=False)

            with gr.Row():
                btn_move_model = gr.Button("Move to device and dtype")
                btn_move_model.click(
                    fn=lambda: gr.Button("Moving..."),
                    outputs=[btn_move_model],
                ).then(
                    fn=move_model_to_device_and_dtype,
                    inputs=[device, dtype, cpu_offload],
                    outputs=[gr.Textbox(visible=False)],
                ).then(
                    fn=lambda: gr.Button("Move to device and dtype"),
                    outputs=[btn_move_model],
                )
                unload_model_button("chatterbox")

            gr.Markdown("## Advanced Settings")
            with gr.Group(), gr.Column():
                initial_forward_pass_backend = gr.Radio(
                    label="Initial forward pass backend",
                    choices=["eager", "cudagraphs"],
                    value="eager",
                )
                generate_token_backend = gr.Radio(
                    label="Generate token backend",
                    choices=[
                        "cudagraphs-manual",
                        "eager",
                        "cudagraphs",
                        "inductor",
                        "cudagraphs-strided",
                        "inductor-strided",
                    ],
                    value="cudagraphs-manual",
                )
                with gr.Row():
                    max_new_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=100,
                        maximum=1000,
                        value=1000,
                        step=10,
                    )
                    max_cache_len = gr.Slider(
                        label="Cache length",
                        minimum=200,
                        maximum=1500,
                        value=1500,
                        step=100,
                    )

            gr.Markdown("Memory usage:")
            gr.Button("Check memory usage").click(
                fn=get_chatterbox_memory_usage,
                outputs=[gr.Markdown()],
            )

    inputs = {
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
        language_id: "language_id",
        # hyperparameters
        chunked: "chunked",
        cpu_offload: "cpu_offload",
        cache_voice: "cache_voice",
        # chunks
        desired_length: "desired_length",
        max_length: "max_length",
        halve_first_chunk: "halve_first_chunk",
        # compile
        # use_compilation: "use_compilation",
        initial_forward_pass_backend: "initial_forward_pass_backend",
        generate_token_backend: "generate_token_backend",
        # optimization
        max_new_tokens: "max_new_tokens",
        max_cache_len: "max_cache_len",
    }

    generation_start = {
        "fn": lambda: [
            gr.Button("Generating...", interactive=False),
            gr.Button("Generating...", interactive=False),
            gr.Button("Interrupt next chunk", interactive=True, variant="stop"),
        ],
        "outputs": [btn, btn_stream, btn_interrupt],
    }
    generation_end = {
        "fn": lambda: [
            gr.Button("Generate", interactive=True, variant="primary"),
            gr.Button("Streaming generation", interactive=True, variant="secondary"),
            gr.Button("Interrupt next chunk", interactive=False, variant="stop"),
        ],
        "outputs": [btn, btn_stream, btn_interrupt],
    }

    btn.click(**randomize_seed_callback).then(**generation_start).then(
        **dictionarize_wraps(
            tts_decorated,
            inputs=inputs,
            outputs={
                "audio_out": audio_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
            api_name="chatterbox_tts",
        )
    ).then(**generation_end)

    btn_stream.click(**randomize_seed_callback).then(**generation_start).then(
        **dictionarize_wraps(
            tts_generator_decorated,
            inputs=inputs,
            outputs={
                "audio_out": streaming_audio_output,
                # "metadata": gr.JSON(visible=False),
                # "folder_root": gr.Textbox(visible=False),
            },
            api_name="chatterbox_tts_streaming",
        )
    ).then(**generation_end)


@functools.wraps(vc)
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("chatterbox-vc")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def vc_decorated(*args, **kwargs):
    return vc(*args, **kwargs)


def chatterbox_vc():
    with gr.Column():
        audio_in = gr.Audio(label="Input Audio", type="filepath", value=None)
        btn = gr.Button("Convert", variant="primary")
        audio_ref = gr.Audio(label="Audio Reference", type="filepath", value=None)
    with gr.Column():
        audio_out = gr.Audio(label="Output Audio")

    btn.click(fn=lambda: gr.Button("Converting..."), outputs=[btn]).then(
        **dictionarize_wraps(
            vc_decorated,
            inputs={audio_in: "audio_in", audio_ref: "audio_ref"},
            outputs={
                "audio_out": audio_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
            api_name="chatterbox_vc",
        )
    ).then(fn=lambda: gr.Button("Convert"), outputs=[btn])


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()

    demo.launch(
        server_port=7771,
    )
    # python -m workspace.extension_chatterbox.extension_chatterbox.gradio_app
