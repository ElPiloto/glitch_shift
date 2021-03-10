"""Turns an image into a gif and then mp4.

- Downloads image specified via IMG_URL at bottom.
- 
"""
import os

import imageio
import glob
import numpy as np

from skimage.color import rgb2hsv, hsv2rgb

np.random.seed(2222)


def read_to_hsv(fname):
  """Loads RGBA image to HSV discarding alpha."""
  img = imageio.imread(fname)
  if img.shape[-1] == 4:
    img = img[..., :-1]
  img = rgb2hsv(img)
  return img


def write_hsv(fname, img):
  """Writes HSV image to fname."""
  img = hsv2rgb(img)
  imageio.imwrite(fname, img)


def write_hsv_gif(fname, all_imgs):
  """Writes HSV image to fname."""
  all_imgs = [hsv2rgb(i) for i in all_imgs]
  all_imgs = [(255 * i).astype(np.uint8) for i in all_imgs]
  print('Converted to rbg')
  print('Writing out gif file')
  imageio.mimsave(fname, all_imgs)


def load_png_images(fnames=None):
  """Loads images specified or all PNG images."""
  if fnames is None:
    fnames = glob.glob("*.png")
    fnames += glob.glob("*.PNG")
    print(fnames)
  images = []
  for f in fnames:
    images.append(read_to_hsv(f))
  return images


def glitch_gif(images,
               fname='temp.gif',
               steps=50,
               write_backwards=False,
               make_reversible=False,
               cutoff_lines=100,
               downshift_freq=1):
  """Takes an image, gradually changes its color, and shifts lines downwards."""
  orig_img = images[0]
  # We cut off lines to hide us moving lines.
  shifted_images = [orig_img[:-cutoff_lines]]
  for i in range(1):
    for t in range(steps - 1):
      # Change hue.
      orig_img[:, :, 0] += 0.05
      orig_img[:, :, 0] = np.mod(orig_img[:, :, 0], 1.)
      # Shift lines downwards and put them back on top.
      if t % downshift_freq == 0:
        shifted_lines = orig_img[1:, :, :]
        first_line = orig_img[:1, :, :]
        orig_img = np.concatenate([shifted_lines, first_line], axis=0)
      img = np.copy(orig_img[:-cutoff_lines])
      if write_backwards:
        shifted_images.insert(0, img)
      else:
        shifted_images.append(img)
  if make_reversible:
    not_last = shifted_images[0:-3]
    shifted_images += not_last[::-1]
  print('writing gif')
  fname = fname.replace('png', 'gif')
  fname = fname.replace('jpg', 'gif')
  write_hsv_gif(fname, shifted_images)


fname = 'wall_drip.jpg'
IMG_URL = ('https://images.squarespace-cdn.com/content/v1/'
           '54bd5b83e4b076c29fcad1cb/1597876651943-R1S7E7S69COVANH10L47/'
           'ke17ZwdGBToddI8pDm48kJUF6o69OLkCFpTseF8Z5o57gQa3H78H3Y0txjaiv'
           '_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QHyNOqBUUEtDDsRW'
           'rJLTmTqyr0YSX1lQOSnWxEpwDIrirVpw8RlXeLrSBui8th2WB6vJrhxg9wvlbG'
           'jSvL1Q5/image-asset.jpeg?format=750w')

# Download image, the default URL points to:
# https://www.jenstark.com/2-d/txtr74jn72l93xj54w48ac53caeh8j
# This image is the copyright of Jen Start (jenstark.com)
os.system(f'wget -O {fname} {IMG_URL}')
images = load_png_images(fnames=[fname])
glitch_gif(images, fname, make_reversible=True)
# Convert output to mp4 because gifs are huge.
# Requires ffmpeg to be installed.
# os.system('ffmpeg -i wall_drip.gif wall_drip.mp4')
