import nnabla as nn
import nnabla.functions as F
from PIL import Image

import clip

def main():
    with nn.auto_forward():
        clip.load('data/ViT-B-32.h5')

        image = clip.preprocess(Image.open("CLIP.png"))
        text = clip.tokenize(["a diagram", "a dog", "a cat"])

        image_features = clip.encode_image(image)
        text_features = clip.encode_text(text)
            
        logits_per_image, logits_per_text = clip.logits(image, text)
        probs = F.softmax(logits_per_image, axis=-1)

        print("Label probs:", probs.d)  # prints: [[0.9927937  0.00421068 0.00299572]]


if __name__ == "__main__":
    main()