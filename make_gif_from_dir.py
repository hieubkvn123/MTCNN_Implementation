import os
import glob
import imageio

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='Path to the image directory')
args = vars(parser.parse_args())

extensions = ['jpg', 'jpeg', 'png', 'ppm']
images = []
image_paths = []

for ext in extensions:
    image_paths += glob.glob(os.path.join(args['dir'], f'*.{ext}'))

print(image_paths)
image_paths = {int(x.split('/')[-1].split('.')[0]):x for x in image_paths}

for filename in sorted(image_paths):
    images.append(imageio.imread(image_paths[filename]))

imageio.mimsave('media/pnet_conf_maps.gif', images, fps=2)
print('[INFO] GIF from image directory created')
