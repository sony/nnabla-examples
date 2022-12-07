import nnabla as nn
import numpy as np

from typing import Union, Optional, List
from io import BytesIO

from clip import CLIP

PRESET_HAIR = '\n'.join(f'A person with {color} hair' for color in [
                        'white', 'black', 'brown', 'red', 'pink', 'blue', 'blond'])
PRESET_EXPRESSION = '\n'.join(f'A {expression} person' for expression in [
                              'happy', 'sad', 'calm', 'angry', 'funny', 'crying'])
IMAGE_EXAMPLES = [
    'https://images.pexels.com/photos/5682847/pexels-photo-5682847.jpeg?auto=compress&cs=tinysrgb&w=600',
    'https://images.pexels.com/photos/2741036/pexels-photo-2741036.jpeg?auto=compress&cs=tinysrgb&w=600',
    'https://images.pexels.com/photos/1852300/pexels-photo-1852300.jpeg?auto=compress&cs=tinysrgb&w=600',
    'https://images.pexels.com/photos/13198518/pexels-photo-13198518.jpeg?auto=compress&cs=tinysrgb&w=600',
]


def read_image_from_bytes(b: BytesIO) -> np.ndarray:
    from PIL import Image
    pil_img = Image.open(b)
    np_img = np.asarray(pil_img, dtype=np.uint8)
    return np_img


def download_image(url: str, return_bytes: bool = False) -> Union[np.ndarray, BytesIO]:
    import requests
    # Download and read image as numpy array
    c = requests.get(url).content
    b = BytesIO(c)
    if return_bytes:
        return b
    np_img = read_image_from_bytes(b)
    return np_img


def parse_categories(texts: str) -> List[str]:
    return list(
        filter(
            lambda x: x,
            map(lambda x: x.strip(), texts.splitlines())))


class GradioCLIP(CLIP):

    def run_by_url(self, url: str, categories: str):
        img_bytes = download_image(url, return_bytes=True)
        np_img = read_image_from_bytes(img_bytes)
        categories = parse_categories(categories)
        img_bytes.seek(0)
        probs = self.run(img_bytes, categories)
        inds = probs.argsort()[::-1]
        labels = {categories[i]: float(probs[i]) for i in inds}
        return np_img, labels


def main():

    import gradio as gr
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn')
    nn.set_default_context(ctx)

    # Create layout
    with gr.Blocks() as demo:
        w_md = gr.Markdown('''
        # Zero-shot classification using nnabla CLIP

        1. Provide URL of an input image.
        2. Define categories for zero-shot classification. See category definition examples below. Each line reprsents a category.
        3. Click the "Run" button to run predicted scores.
        ''')
        with gr.Row():
            with gr.Column(scale=2):
                w_url = gr.Textbox(label='Image URL', lines=1,
                                   max_lines=1, value=IMAGE_EXAMPLES[0])
                w_categories = gr.Textbox(label='Zero-shot categories', lines=10, max_lines=100,
                                          value=PRESET_HAIR)
                w_predict = gr.Button(label='Run')
            with gr.Column(scale=3):
                w_image = gr.Image(label='Image')
            with gr.Column(scale=4):
                w_label = gr.Label(label='Prediction')

        # Set callbacks
        gradio_clip = GradioCLIP()
        w_preset_categories = gr.Examples(
            [PRESET_HAIR, PRESET_EXPRESSION],
            [w_categories],
            label='Category definition examples',
        )
        w_preset_images = gr.Examples(
            [[x] for x in IMAGE_EXAMPLES],
            inputs=[w_url],
            label='Example images',
        )

        w_url.change(download_image, inputs=[w_url], outputs=w_image)
        w_predict.click(gradio_clip.run_by_url, inputs=[
                        w_url, w_categories], outputs=[w_image, w_label])

    # Launch servers
    demo.queue(concurrency_count=1)
    demo.launch(server_name='0.0.0.0', server_port=8889)


if __name__ == '__main__':
    main()
