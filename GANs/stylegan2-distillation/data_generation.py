from genericpath import exists
import os
import sys
import argparse

stylegan2_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(stylegan2_path)
from stylegan2.generate import synthesis
from stylegan2.networks import mapping_network
from stylegan2.ops import lerp, convert_images_to_uint8

import numpy as np
from tqdm import trange

import nnabla as nn
from nnabla.ext_utils import get_extension_context
import nnabla.functions as F
from nnabla.utils.image_utils import imsave

def generate_data(args):

    if not os.path.isfile(os.path.join(args.weights_path, 'gen_params.h5')):
        os.makedirs(args.weights_path, exist_ok=True)
        print("Downloading the pretrained tf-converted weights. Please wait...")
        url = "https://nnabla.org/pretrained-models/nnabla-examples/GANs/stylegan2/styleGAN2_G_params.h5"
        from nnabla.utils.data_source_loader import download
        download(url, os.path.join(args.weights_path, 'gen_params.h5'), False)

    nn.load_parameters(os.path.join(args.weights_path, 'gen_params.h5'))
    print('Loaded pretrained weights from tensorflow!')
    
    os.makedirs(args.save_image_path, exist_ok=True)
    
    batches = [args.batch_size for _ in range(args.num_images//args.batch_size)]
    if args.num_images%args.batch_size != 0:
        batches.append(args.num_images - (args.num_images//args.batch_size)*args.batch_size)
        
    for idx, batch_size in enumerate(batches):
        z = [F.randn(shape=(batch_size, 512)).data, F.randn(shape=(batch_size, 512)).data]

        for i in range(len(z)):
            z[i] = F.div2(z[i], F.pow_scalar(F.add_scalar(F.mean(
                z[i] ** 2., axis=1, keepdims=True), 1e-8, inplace=True), 0.5, inplace=True))

        # get latent code
        w = [mapping_network(z[0], outmaps=512, num_layers=8)]
        w += [mapping_network(z[1], outmaps=512, num_layers=8)]

        # truncation trick
        dlatent_avg = nn.parameter.get_parameter_or_create(
            name="dlatent_avg", shape=(1, 512))
        w = [lerp(dlatent_avg, _, 0.7) for _ in w]

        # Load direction
        attr_delta = nn.NdArray.from_numpy_array(np.load(args.attr_delta_path))
        attr_delta = F.reshape(attr_delta[0], (1,-1))
        w_plus = [w[i] + args.coeff*attr_delta for i in range(len(w))]
        w_minus = [w[i] - args.coeff*attr_delta for i in range(len(w))]

        constant_bc = nn.parameter.get_parameter_or_create(
                        name="G_synthesis/4x4/Const/const",
                        shape=(1, 512, 4, 4))
        constant_bc = F.broadcast(
            constant_bc, (batch_size,) + constant_bc.shape[1:])

        gen_plus = synthesis(w_plus, constant_bc, noise_seed=100, mix_after=7)
        gen_minus = synthesis(w_minus, constant_bc, noise_seed=100, mix_after=7)
        gen = synthesis(w, constant_bc, noise_seed=100, mix_after=7)
        
        image_plus = convert_images_to_uint8(gen_plus, drange=[-1, 1])
        image_minus = convert_images_to_uint8(gen_minus, drange=[-1, 1])
        image = convert_images_to_uint8(gen, drange=[-1, 1])

        for j in range(batch_size):
            filepath = os.path.join(args.save_image_path, f'image_{idx*batch_size+j}')
            # filepath = os.path.join(args.output_dir, f'{filename}_{i}.png')
            imsave(f'{filepath}_o.png', image_plus[j], channel_first=True)
            imsave(f'{filepath}_y.png', image_minus[j], channel_first=True)
            imsave(f'{filepath}.png', image[j], channel_first=True)
            print(f"Genetated. Saved {filepath}")
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-image-path', type=str, default='/home/krishna.wadhwani/gdown.pl/age-dataset-1024jpeg',
                        help="name of directory to save output image")
    parser.add_argument('--attr-delta-path', type=str, default='stylegan2directions/age.npy',
                        help="Path to npy file of attribute variation in stylegan2 latent space")
    parser.add_argument('--weights-path', type=str, default='./',
                        help="Path to store pretrained stylegan2 parameters")

    parser.add_argument('--batch-size', type=int, default=8,
                        help="Batch-size of 1 forward pass of the generator")
    parser.add_argument('--num-images', type=int, default=50000,
                        help="Number of images to generate.")
    
    parser.add_argument('--coeff', type=float, default=2.25,
                        help="coefficient of propagation in stylegan2 latent space")
    
    parser.add_argument('--context', type=str, default="cudnn",
                    help="context. cudnn is recommended.")
    
    args = parser.parse_args()
    
    assert args.num_images > args.batch_size, 'Number of images must be more than the batch-size'
    
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)
    
    generate_data(args)

if __name__ == '__main__':
    main()
