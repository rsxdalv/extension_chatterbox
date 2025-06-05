import os
import functools
import gradio as gr
from contextlib import contextmanager
import torch
from chatterbox.tts import ChatterboxTTS

from tts_webui.utils.manage_model_state import (
    manage_model_state,
    rename_model,
    get_current_model,
)
from tts_webui.decorators import *
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
from tts_webui.utils.get_path_from_root import get_path_from_root
from tts_webui.utils.split_text_functions import split_and_recombine_text
from gradio_iconbutton import IconButton


def chatterbox_to(model: ChatterboxTTS, device, dtype):
    print(f"Moving model to {str(device)}, {str(dtype)}")

    model.ve.to(device=device)
    model.t3.to(device=device, dtype=dtype)
    model.s3gen.to(device=device, dtype=dtype)
    # due to "Error: cuFFT doesn't support tensor of type: BFloat16" from torch.stft
    model.s3gen.tokenizer.to(dtype=torch.float32)
    model.conds.to(device=device)
    model.device = device
    torch.cuda.empty_cache()

    return model


def resolve_dtype(dtype):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def resolve_device(device):
    return get_best_device() if device == "auto" else device


def generate_model_name(device, dtype):
    return f"Chatterbox on {device} with {dtype}"


def move_model_to_device_and_dtype(device, dtype, cpu_offload):
    model = get_current_model("chatterbox")
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    if model is None:
        get_model("just_a_placeholder", device, dtype)
        return True
    rename_model("chatterbox", generate_model_name(device, dtype))
    device = torch.device("cpu" if cpu_offload else device)
    model = chatterbox_to(model, device, dtype)
    return True


@manage_model_state("chatterbox")
def get_model(
    model_name="just_a_placeholder", device=torch.device("cuda"), dtype=torch.float32
):
    model = ChatterboxTTS.from_pretrained(device=device)
    # having everything on float32 increases performance
    return chatterbox_to(model, device, dtype)


@contextmanager
def chatterbox_model(model_name, device="cuda", dtype=torch.float32):
    model = get_model(
        model_name=generate_model_name(device, dtype),
        device=torch.device(device),
        dtype=dtype,
    )

    use_autocast = dtype in [torch.float16, torch.bfloat16]

    with (
        torch.autocast(device_type=device, dtype=dtype)
        if use_autocast
        else torch.no_grad()
    ):
        yield model


@contextmanager
def cpu_offload_context(model, device, dtype, cpu_offload=False):
    if cpu_offload:
        chatterbox_to(model, torch.device(device), dtype)
    yield model
    if cpu_offload:
        chatterbox_to(model, torch.device("cpu"), dtype)


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


import asyncio


class InterruptionFlag:
    def __init__(self):
        self._interrupted = False
        self._ack_event = asyncio.Event()

    def interrupt(self):
        self._interrupted = True
        # Do NOT set ack_event here — only acknowledge() should

    def reset(self):
        self._interrupted = False
        self._ack_event.clear()

    def is_interrupted(self):
        return self._interrupted

    def acknowledge(self):
        self._ack_event.set()

    async def join(self, timeout=None):
        """Wait until acknowledge() is called after interrupt()."""
        try:
            await asyncio.wait_for(self._ack_event.wait(), timeout)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for interruption to be acknowledged.")


def interruptible(gen_func):
    @functools.wraps(gen_func)
    def wrapper(*args, interrupt_flag=None, **kwargs):
        interrupt_flag.reset()
        gen = gen_func(*args, **kwargs)
        try:
            for item in gen:
                yield item
                if interrupt_flag and interrupt_flag.is_interrupted():
                    print("Interrupted.")
                    break
        finally:
            interrupt_flag.acknowledge()
            if hasattr(gen, "close"):
                gen.close()

    return wrapper


@interruptible
def _tts_generator(
    text,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    audio_prompt_path=None,
    # model
    model_name="just_a_placeholder",
    device="cuda",
    dtype="float32",
    cpu_offload=False,
    # hyperparameters
    chunked=False,
    cache_voice=False,
    # streaming
    tokens_per_slice=1000,
    remove_milliseconds=100,
    remove_milliseconds_start=100,
    chunk_overlap_method="zero",
    # chunks
    desired_length=200,
    max_length=300,
    halve_first_chunk=False,
    seed=-1,  # for signature compatibility
    progress=gr.Progress(),
    streaming=False,
    # progress=gr.Progress(track_tqdm=True),
    **kwargs,
):
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)

    print(f"Using device: {device}")

    progress(0.0, desc="Retrieving model...")
    with chatterbox_model(
        model_name=model_name,
        device=device,
        dtype=dtype,
    ) as model, cpu_offload_context(model, device, dtype, cpu_offload):
        progress(0.1, desc="Generating audio...")

        def generate_chunk(text):
            print(f"Generating chunk: {text}")
            yield from model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                # stream
                tokens_per_slice=tokens_per_slice,
                remove_milliseconds=remove_milliseconds,
                remove_milliseconds_start=remove_milliseconds_start,
                chunk_overlap_method=chunk_overlap_method,
                # Not implemented
                # cache_voice=cache_voice,
            )

        texts = (
            split_and_recombine_text(text, desired_length, max_length)
            if chunked
            else [text]
        )
        if halve_first_chunk:
            texts = split_by_length_simple(texts[0], desired_length // 2, max_length // 2) + texts[1:]
        # for chunk in texts:
        for i, chunk in enumerate(texts):
            if not streaming:
                progress(i / len(texts), desc=f"Generating chunk: {chunk}")
            for wav in generate_chunk(chunk):
                yield {
                    "audio_out": (model.sr, wav.squeeze().cpu().numpy()),
                }


global_interrupt_flag = InterruptionFlag()


@functools.wraps(_tts_generator)
def tts_stream(*args, **kwargs):
    try:
        yield from _tts_generator(
            *args, interrupt_flag=global_interrupt_flag, streaming=True, **kwargs
        )
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise gr.Error(f"Error: {e}")


@functools.wraps(_tts_generator)
def tts(*args, **kwargs):
    try:
        wavs = list(
            _tts_generator(*args, interrupt_flag=global_interrupt_flag, **kwargs)
        )
        if not wavs:
            raise gr.Error("No audio generated")
        full_wav = np.concatenate([x["audio_out"][1] for x in wavs], axis=0)
        return {
            "audio_out": (wavs[0]["audio_out"][0], full_wav),
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


@functools.wraps(_tts_generator)
def tts_generator(*args, **kwargs):
    yield from tts_stream(*args, **kwargs)


import io
import wave
import torch
import numpy as np
import functools
import time


def decorator_convert_audio_output_generator(func):
    """Final decorator to convert audio_out from tuple to bytes before returning to caller"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for chunk in func(*args, **kwargs):
            if "audio_out" in chunk:
                # Convert the audio_out from (sample_rate, numpy_array) to bytes
                sample_rate, audio_data = chunk["audio_out"]
                audio_bytes = numpy_to_wav_bytes(audio_data, sample_rate)
                # chunk = {**chunk, "audio_out": audio_bytes}
                chunk["audio_out"] = audio_bytes
            yield chunk

    return wrapper


def numpy_to_wav_bytes(audio_data, sample_rate):
    """Convert numpy array to WAV format bytes"""
    # Ensure audio_data is in the right format
    if audio_data.dtype != np.int16:
        # Convert from float [-1, 1] to int16
        audio_data = (audio_data * 32767).astype(np.int16)

    # Create WAV file in memory
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    # Get the bytes
    buffer.seek(0)
    return buffer.getvalue()


@functools.wraps(tts_generator)
# @decorator_convert_audio_output_generator  # <-- This goes first/top
@decorator_extension_outer_generator
@decorator_apply_torch_seed_generator
# @decorator_save_metadata_generator
# @decorator_save_wav_generator
@decorator_add_model_type_generator("chatterbox")
# @decorator_add_base_filename_generator
@decorator_add_date_generator
@decorator_log_generation_generator
@decorator_extension_inner_generator
@log_generator_time
def tts_generator_decorated(*args, **kwargs):
    yield from tts_generator(*args, **kwargs)


async def interrupt():
    global_interrupt_flag.interrupt()
    await global_interrupt_flag.join()
    return "Interrupt next chunk"


def get_voices():
    voices_dir = get_path_from_root("voices", "chatterbox")
    os.makedirs(voices_dir, exist_ok=True)
    results = [
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
      <h2>Performance on NVIDIA RTX 3090</h2>
      <ul style="list-style-type: disc; padding-left: 20px;">
        <li><strong>VRAM</strong>: Float32: 5-7 GB, Bfloat16: 3-4 GB, CPU Offloading Passive: 0.7 GB of VRAM</li>
        <li><strong>Speed</strong>: ~32.04 iterations per second, 1:1 ratio</li>
      </ul>
    </div>
  </div>
                """
    )

    with gr.Row():
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

            seed, randomize_seed_callback = randomize_seed_ui()

        with gr.Column():
            audio_out = gr.Audio(label="Audio Output")
            streaming_audio_output = gr.Audio(
                label="Audio Output (streaming)", streaming=True, autoplay=True
            )

            gr.Markdown("## Settings")

            with gr.Accordion("Chunking", open=True), gr.Group():
                chunked = gr.Checkbox(label="Split prompt into chunks", value=False)
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
                    halve_first_chunk = gr.Checkbox(
                        label="Halve first chunk size",
                        value=False,
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
                        value="float32",
                    )
                    cpu_offload = gr.Checkbox(label="CPU Offload", value=False)
                    model_name = gr.Dropdown(
                        label="Model",
                        choices=["just_a_placeholder"],
                        value="just_a_placeholder",
                        visible=False,
                    )
                with gr.Row():
                    btn_move_model = gr.Button("Move to device and dtype")
                    btn_move_model.click(
                        fn=lambda: gr.Button("Moving..."),
                        outputs=[btn_move_model],
                    ).then(
                        fn=move_model_to_device_and_dtype,
                        inputs=[device, dtype, cpu_offload],
                    ).then(
                        fn=lambda: gr.Button("Move to device and dtype"),
                        outputs=[btn_move_model],
                    )
                    unload_model_button("chatterbox")

            with gr.Accordion("Streaming (Advanced Settings)", open=False):
                gr.Markdown(
                    """
Streaming has issues due to Chatterbox producing artifacts.
Tokens per slice: 
* 1000 is recommended, it is the default maximum value, equivalent to disabling streaming.
* One second is around 23.5 tokens.
                            
Remove milliseconds:
* 25 - 65 is recommended.
    * This removes the last 45 milliseconds of each slice to avoid artifacts.
* start: 15 - 35 is recommended.
    * This removes the first 25 milliseconds of each slice to avoid artifacts.
                            
Chunk overlap method:
* zero means that each chunk is seen sparately by the audio generator. 
* full means that each chunk is appended and decoded as one long audio file.
                            
Thus **the challenge is to fix the seams** - with no overlap, the artifacts are high. With a very long overlap, such as a 0.5s crossfade, the audio starts to produce echo.
"""
                )
                with gr.Row():
                    tokens_per_slice = gr.Slider(
                        label="Tokens per slice",
                        minimum=1,
                        maximum=1000,
                        value=1000,
                        step=1,
                    )
                    remove_milliseconds = gr.Slider(
                        label="Remove milliseconds",
                        minimum=0,
                        maximum=300,
                        value=45,
                        step=1,
                    )
                    remove_milliseconds_start = gr.Slider(
                        label="Remove milliseconds start",
                        minimum=0,
                        maximum=300,
                        value=25,
                        step=1,
                    )
                    chunk_overlap_method = gr.Radio(
                        label="Chunk overlap method",
                        choices=["zero", "full"],
                        value="zero",
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
        # hyperparameters
        chunked: "chunked",
        cpu_offload: "cpu_offload",
        cache_voice: "cache_voice",
        # streaming
        tokens_per_slice: "tokens_per_slice",
        remove_milliseconds: "remove_milliseconds",
        remove_milliseconds_start: "remove_milliseconds_start",
        chunk_overlap_method: "chunk_overlap_method",
        # chunks
        desired_length: "desired_length",
        max_length: "max_length",
        halve_first_chunk: "halve_first_chunk",
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


if __name__ == "__main__":
    if "demo" in locals():
        demo.close()
    with gr.Blocks() as demo:
        ui()

    demo.launch(
        server_port=7771,
    )
    # python -m workspace.extension_chatterbox.extension_chatterbox.gradio_app
