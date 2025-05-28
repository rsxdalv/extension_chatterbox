import gradio as gr
from tts_webui.utils.manage_model_state import manage_model_state, is_model_loaded
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.decorators import *
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
import functools

REPO_ID = "chatterbox-tts/chatterbox"


CHECKPOINT_DIR = "data/models/chatterbox/"
USE_HALF_PRECISION = True
USE_TORCH_COMPILE = False
USE_CPU_OFFLOAD = True
USE_OVERLAPPED_DECODE = True


@manage_model_state("chatterbox")
def get_model(
    model_name=REPO_ID,
    use_half_precision=None,
    torch_compile=None,
    checkpoint_dir=CHECKPOINT_DIR,
    cpu_offload=None,
    overlapped_decode=None,
):
    from chatterbox import ChatterboxPipeline

    if use_half_precision is None:
        use_half_precision = USE_HALF_PRECISION

    if torch_compile is None:
        torch_compile = USE_TORCH_COMPILE

    if cpu_offload is None:
        cpu_offload = USE_CPU_OFFLOAD

    if overlapped_decode is None:
        overlapped_decode = USE_OVERLAPPED_DECODE

    dtype = "bfloat16" if use_half_precision else "float32"

    model_demo = ChatterboxPipeline(
        checkpoint_dir=checkpoint_dir,
        dtype=dtype,
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )

    return model_demo


def store_global_settings(
    use_half_precision, use_torch_compile, use_cpu_offload, use_overlapped_decode
):
    global USE_HALF_PRECISION, USE_TORCH_COMPILE, USE_CPU_OFFLOAD, USE_OVERLAPPED_DECODE
    USE_HALF_PRECISION = use_half_precision
    USE_TORCH_COMPILE = use_torch_compile
    USE_CPU_OFFLOAD = use_cpu_offload
    USE_OVERLAPPED_DECODE = use_overlapped_decode
    if is_model_loaded("chatterbox"):
        return "Please unload the model to apply changes."
    return "Settings applied."


@manage_model_state("chatterbox_sampler")
def get_sampler(model_name=REPO_ID):
    from chatterbox import DataSampler

    data_sampler = DataSampler()
    return data_sampler


def decorator_chatterbox_adapter(fn):
    def chatterbox_infer(
        format: str = "wav",
        audio_duration: float = 10.0,
        voice_style: str = "natural, clear, conversational, neutral",
        text_input: str = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        manual_seeds: str = None,
        **kwargs,
    ):
        params = locals()
        del params["kwargs"]
        del params["fn"]
        return fn(**params, text=text_input[:50] if text_input else "")

    return chatterbox_infer


# This decorator will convert the return value from a list to a dict
def to_dict_decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, list) and len(result) > 0:
            import soundfile as sf

            audio, sample_rate = sf.read(result[0])

            return {"audio_out": (sample_rate, audio), "_original_result": result}
        return result

    return wrapper


# This decorator will convert the return value back from a dict to the original list
def from_dict_decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        # Check if the result is a dict with the _original_result key
        if isinstance(result, dict) and "_original_result" in result:
            return result["_original_result"]
        return result

    return wrapper


# @functools.wraps(chatterbox_infer)
@decorator_chatterbox_adapter
@from_dict_decorator  # This will run last, converting dict back to list
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("chatterbox")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@to_dict_decorator  # This will run first, converting list to dict
@log_function_time
def chatterbox_infer_decorated(*args, _type, text, **kwargs):
    from chatterbox import ChatterboxPipeline

    model_demo: ChatterboxPipeline = get_model(REPO_ID)

    return model_demo(*args, **kwargs)


def sample_data(*args, **kwargs):
    data_sampler = get_sampler(REPO_ID)

    return data_sampler.sample(*args, **kwargs)


def load_data(*args, **kwargs):
    data_sampler = get_sampler(REPO_ID)

    return data_sampler.load_data(*args, **kwargs)


def ui():
    # from chatterbox.ui.components import create_text2speech_ui

    from .components import create_text2speech_ui

    gr.Markdown(
        """<h2 style="text-align: center;">Chatterbox: Text-to-Speech Generation</h2>"""
    )

    with gr.Column(variant="panel"):
        gr.Markdown("### Model Settings")
        with gr.Row():
            use_half_precision = gr.Checkbox(
                label="Use half precision",
                value=USE_HALF_PRECISION,
            )
            use_torch_compile = gr.Checkbox(
                label="Use torch compile",
                value=USE_TORCH_COMPILE,
            )
            use_cpu_offload = gr.Checkbox(
                label="Use CPU offload",
                value=USE_CPU_OFFLOAD,
            )
            use_overlapped_decode = gr.Checkbox(
                label="Use overlapped decode",
                value=USE_OVERLAPPED_DECODE,
            )
            # save global changes when any of these change:
            unload_model_button("chatterbox")

        model_info = gr.Markdown("")

    for i in [
        use_half_precision,
        use_torch_compile,
        use_cpu_offload,
        use_overlapped_decode,
    ]:
        i.change(
            fn=store_global_settings,
            inputs=[
                use_half_precision,
                use_torch_compile,
                use_cpu_offload,
                use_overlapped_decode,
            ],
            outputs=[model_info],
            api_name="chatterbox_reload_model",
        )

    create_text2speech_ui(
        gr=gr,
        text2speech_process_func=chatterbox_infer_decorated,
        sample_data_func=sample_data,
        load_data_func=load_data,
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()
    demo.launch()
