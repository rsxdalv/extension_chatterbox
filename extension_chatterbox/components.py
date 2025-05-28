"""
Chatterbox TTS: Text-to-Speech Generation

https://github.com/chatterbox-tts/chatterbox

MIT License
"""

import gradio as gr
import os


TEXT_DEFAULT = "Hello, this is a sample text for text-to-speech generation. Chatterbox can convert this text into natural-sounding speech."

# Voice presets for different speaking styles
VOICE_PRESETS = {
    "Natural": "natural, clear, conversational, neutral",
    "Formal": "formal, professional, clear, authoritative", 
    "Casual": "casual, friendly, relaxed, warm",
    "Energetic": "energetic, enthusiastic, upbeat, dynamic",
    "Calm": "calm, soothing, gentle, peaceful",
    "Dramatic": "dramatic, expressive, theatrical, emotional",
    "News": "news, broadcast, clear, professional",
    "Storytelling": "storytelling, narrative, engaging, expressive",
}


# Add this function to handle preset selection
def update_voice_from_preset(preset_name):
    if preset_name == "Custom":
        return ""
    return VOICE_PRESETS.get(preset_name, "")


def create_output_ui(task_name="Text2Speech"):
    output_audio1 = gr.Audio(type="filepath", label=f"{task_name} Generated Audio")
    with gr.Accordion(f"{task_name} Parameters", open=False):
        input_params_json = gr.JSON(label=f"{task_name} Parameters")
    outputs = [output_audio1]
    return outputs, input_params_json


def dump_func(*args, **kwargs):
    print(*args)
    print(kwargs)
    return []


def create_text2speech_ui(
    gr,
    text2speech_process_func,
    sample_data_func=None,
    load_data_func=None,
):

    with gr.Row(equal_height=True):
        # Get base output directory from environment variable, defaulting to CWD-relative 'outputs'.
        output_file_dir = os.environ.get("CHATTERBOX_OUTPUT_DIR", "./outputs")
        if not os.path.isdir(output_file_dir):
            os.makedirs(output_file_dir, exist_ok=True)
        json_files = [f for f in os.listdir(output_file_dir) if f.endswith(".json")]
        json_files.sort(reverse=True, key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
        output_files = gr.Dropdown(
            choices=json_files,
            label="Select previous generated input params",
            scale=9,
            interactive=True,
        )
        load_bnt = gr.Button("Load", variant="primary", scale=1)

    with gr.Row():
        with gr.Column():
            with gr.Row(equal_height=True):
                audio_duration = gr.Slider(
                    1.0,
                    60.0,
                    step=0.1,
                    value=10.0,
                    label="Audio Duration (seconds)",
                    interactive=True,
                    scale=9,
                )
                format = gr.Dropdown(
                    choices=["mp3", "ogg", "flac", "wav"], value="wav", label="Format"
                )
                sample_bnt = gr.Button("Sample", variant="secondary", scale=1)

            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown(
                        """<center>Enter the text you want to convert to speech. Use voice presets for different speaking styles.</center>"""
                    )
                    with gr.Row():
                        voice_preset = gr.Dropdown(
                            choices=["Custom"] + list(VOICE_PRESETS.keys()),
                            value="Natural",
                            label="Voice Preset",
                            scale=1,
                        )
                        voice_style = gr.Textbox(
                            lines=1,
                            label="Voice Style",
                            max_lines=2,
                            value=VOICE_PRESETS["Natural"],
                            scale=9,
                        )

            # Add the change event for the preset dropdown
            voice_preset.change(
                fn=update_voice_from_preset, inputs=[voice_preset], outputs=[voice_style]
            )
            
            with gr.Group():
                gr.Markdown(
                    """<center>Enter the text content to be converted to speech.</center>"""
                )
                text_input = gr.Textbox(
                    lines=5,
                    label="Text Input",
                    max_lines=10,
                    value=TEXT_DEFAULT,
                )

            with gr.Accordion("Advanced Settings", open=False):
                speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Speech Speed",
                    interactive=True,
                )
                pitch = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Pitch",
                    interactive=True,
                )
                volume = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Volume",
                    interactive=True,
                )

                manual_seeds = gr.Textbox(
                    label="Manual seeds (default None)",
                    placeholder="1,2,3,4",
                    value=None,
                    info="Seed for the generation",
                )

            text2speech_bnt = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            outputs, input_params_json = create_output_ui()

    def json2output(json_data: dict):
        return (
            json_data.get("audio_duration", 10.0),
            json_data.get("voice_style", VOICE_PRESETS["Natural"]),
            json_data.get("text_input", TEXT_DEFAULT),
            json_data.get("speed", 1.0),
            json_data.get("pitch", 1.0),
            json_data.get("volume", 1.0),
            json_data.get("manual_seeds", None),
        )

    def sample_data():
        if sample_data_func:
            json_data = sample_data_func()
            return json2output(json_data)
        return json2output({})

    sample_bnt.click(
        sample_data,
        inputs=[],
        outputs=[
            audio_duration,
            voice_style,
            text_input,
            speed,
            pitch,
            volume,
            manual_seeds,
        ],
    )

    def load_data(json_file):
        if isinstance(output_file_dir, str) and json_file:
            json_file = os.path.join(output_file_dir, json_file)
            if load_data_func:
                json_data = load_data_func(json_file)
                return json2output(json_data)
        return json2output({})

    load_bnt.click(
        fn=load_data,
        inputs=[output_files],
        outputs=[
            audio_duration,
            voice_style,
            text_input,
            speed,
            pitch,
            volume,
            manual_seeds,
        ],
    )

    text2speech_bnt.click(
        fn=text2speech_process_func,
        inputs=[
            format,
            audio_duration,
            voice_style,
            text_input,
            speed,
            pitch,
            volume,
            manual_seeds,
        ],
        outputs=outputs + [input_params_json],
    )


def create_main_demo_ui(
    text2speech_process_func=dump_func,
    sample_data_func=dump_func,
    load_data_func=dump_func,
):
    with gr.Blocks(
        title="Chatterbox TTS Model DEMO",
    ) as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center;">Chatterbox: Text-to-Speech Generation</h1>
        """
        )
        with gr.Tab("text2speech"):
            create_text2speech_ui(
                gr=gr,
                text2speech_process_func=text2speech_process_func,
                sample_data_func=sample_data_func,
                load_data_func=load_data_func,
            )
    return demo


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    demo = create_main_demo_ui()
    demo.launch()
