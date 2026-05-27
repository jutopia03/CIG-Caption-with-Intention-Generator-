import gradio as gr

def check_path(video_file):
    print(f'타입: {type(video_file)}')
    print(f'값: {video_file}')
    if hasattr(video_file, 'name'):
        print(f'.name: {video_file.name}')
    return str(video_file)

with gr.Blocks() as demo:
    f = gr.File()
    out = gr.Textbox()
    f.change(check_path, inputs=f, outputs=out)

demo.launch()
