import imageio
import glob
import numpy as np
from pygifsicle import optimize

from skimage.color import rgb2hsv, hsv2rgb

np.random.seed(102387)


def agg_random(x):
  num_pixels = x.shape[0]
  if num_pixels > 1:
    chosen_pixel = np.random.choice(range(num_pixels))
    return x[chosen_pixel]
  return x[0]


def agg_mean(x):
  return np.mean(x, axis=0)
  num_pixels = x.shape[0]


AGG_METHODS = {
    'random': agg_random,
    'mean': agg_mean,
}


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
  print('Optimizing gif')
  #optimize(fname)


def load_png_images(fnames=None):
  """Loads images specified or all PNG images."""
  if fnames is None:
    fnames = glob.glob("*.png")
    fnames += glob.glob("*.PNG")
    print(fnames)
  images = []
  for f in fnames:
    images.append(read_to_hsv(f))
  #img = images[0]
  # for i in range(5):
  # img = pixel_shift(img, shift_up=True, save=False)
  #img = pixel_shift(img, horizontal=True, shift_up=False, save=False)
  #img = pixel_shift(img, shift_up=True, save=True)
  return images


def _interpolate_channel(img,
                         start_x,
                         end_x,
                         start_y,
                         end_y,
                         vertical,
                         agg_fn,
                         channel=-1):
  if channel == -1:
    channel = np.random.randint(3)
  start_pixel = agg_fn(img[start_y, start_x:end_x, :])
  end_pixel = agg_fn(img[end_y, start_x:end_x, :])

  start_value = start_pixel[..., channel]
  end_value = end_pixel[..., channel]
  # end_value = 0.
  interpolation = np.linspace(start_value, end_value, num=end_y - start_y)

  # set all pixels to same base value
  #img[start_y:end_y, start_x:end_x, :] = end_pixel

  img[start_y:end_y, start_x:end_x, channel] = interpolation[:, None]
  return img


def _interpolate_multi_channel(img, start_x, end_x, start_y, end_y, vertical,
                               agg_fn, **unused_args):
  orig_start_y = start_y
  orig_end_y = end_y
  spacing = np.linspace(start_y, end_y, 4).astype(np.int)

  for channel in range(3):
    start_y = spacing[channel]
    end_y = spacing[channel + 1]
    channel = 1
    start_pixel = agg_fn(img[start_y, start_x:end_x, :])
    end_pixel = agg_fn(img[end_y, start_x:end_x, :])
    start_value = start_pixel[..., channel]
    end_value = end_pixel[..., channel]
    end_value = 1.
    interpolation = np.linspace(end_value, start_value, num=end_y - start_y)

    # set all pixels to same base value
    # img[start_y:end_y, start_x:end_x, :] = end_pixel

    img[start_y:end_y, start_x:end_x, channel] = interpolation[:, None]
  return img


def interpolate(img,
                start_x,
                end_x,
                start_y,
                end_y,
                vertical=True,
                agg_method='mean'):
  agg_fn = AGG_METHODS[agg_method]
  img = _interpolate_multi_channel(
      img,
      start_x,
      end_x,
      start_y,
      end_y,
      vertical,
      agg_fn,
      channel=2,
  )
  return img


def choose_shift_pixels(img, num_shifts, weight_by_variance=True):
  h, w, c = img.shape
  all_values = list(range(w))
  if weight_by_variance:
    std = np.std(img, axis=(0, 2))**4
    probabilities = std / np.sum(std)
  else:
    probabilities = np.ones(shape=(w,)) * 1. / w
  if False:
    new_values = []
    new_probs = []
    for v in all_values:
      if ((v > 85 and v < 105) or (v > 190 and v < 280) or
          (v > 335 and v < 345)):
        new_values.append(v)
  else:
    new_values = all_values
  shift_pixels = np.random.choice(
      new_values, size=num_shifts, replace=False)  #, p=new_probs)
  return shift_pixels


def choose_start_y(img, shift_pixels):
  h, w, c = img.shape
  if True:
    start_y = int(h * np.random.uniform(0.2, 0.3))
    return [start_y] * len(shift_pixels)
  start_y = []
  for p in shift_pixels:
    if p > 85 and p < 105:
      val = int(h * np.random.uniform(0.25, 0.28))
    elif p > 190 and p < 280:
      val = int(h * np.random.uniform(0.28, 0.3))
      # val = int(h * np.random.uniform(0.09, 0.12))
    elif p > 325 and p < 340:
      val = int(h * np.random.uniform(0.32, 0.34))
      # val = int(h * np.random.uniform(0.13, 0.18))
    else:
      val = int(h * np.random.uniform(0.45, 0.55))
    start_y.append(val)
  return start_y


def choose_deltas(img, shift_pixels):
  h, w, c = img.shape
  if True:
    start_y = int(h * np.random.uniform(0.2, 0.5))
    return [start_y] * len(shift_pixels)
  start_y = []
  for p in shift_pixels:
    if p > 85 and p < 105:
      val = int(h * np.random.uniform(0.25, 0.28))
    elif p > 190 and p < 280:
      val = int(h * np.random.uniform(0.28, 0.3))
      # val = int(h * np.random.uniform(0.09, 0.12))
    elif p > 325 and p < 340:
      val = int(h * np.random.uniform(0.22, 0.28))
      # val = int(h * np.random.uniform(0.13, 0.18))
    else:
      val = int(h * np.random.uniform(0.22, 0.28))
      #val = int(h * np.random.uniform(0.45, 0.55))
    start_y.append(val)
  return start_y


def modify_shift_pixels(shift_pixels):
  return shift_pixels


def modify_start_y(all_start_y):
  return all_start_y


def modify_deltas(all_deltas):
  all_deltas = [x - 1 for x in all_deltas]
  return all_deltas


def pixel_shift(
    img,
    num_shifts=200,
    shift_size=20,
    horizontal=False,
    shift_up=False,
    save=False,
    shift_pixels=None,
    all_start_y=None,
    all_deltas=None,
):
  if horizontal:
    img = np.transpose(img, [1, 0, 2])
  if shift_up:
    img = np.flip(img, axis=0)
  h, w, c = img.shape
  img = np.copy(img)
  if shift_pixels is None:
    shift_pixels = choose_shift_pixels(img, num_shifts)
    all_start_y = choose_start_y(img, shift_pixels)
    all_deltas = [
        int(h * np.random.uniform(0.6, 0.8)) for _ in range(num_shifts)
    ]
  else:
    shift_pixels = modify_shift_pixels(shift_pixels)
    all_start_y = modify_start_y(all_start_y)
    all_deltas = modify_deltas(all_deltas)
  for i, (start_y, p) in enumerate(zip(all_start_y, shift_pixels)):
    delta = all_deltas[i]
    end_y = min(h - 1, start_y + delta)
    img = interpolate(img, p, p + shift_size, start_y, end_y, vertical=True)

  # undo shift
  if shift_up:
    img = np.flip(img, axis=0)
  # undo horizontal
  if horizontal:
    img = np.transpose(img, [1, 0, 2])
  if save:
    write_hsv('temp.png', img)
  return img, shift_pixels, all_start_y, all_deltas


def glitch_gif(images,
               fname='temp.gif',
               steps=50,
               write_backwards=False,
               make_reversible=False):
  num_shifts = 100
  orig_img = images[0]
  shifted_images = [orig_img[:-100]]
  for i in range(1):
    # img, shift_pixels, all_start_y, all_deltas = pixel_shift(
    # orig_img, shift_up=True, num_shifts=num_shifts)
    # shifted_images.append(img)
    for t in range(steps - 1):
      orig_img[:, :, 0] += 0.05
      orig_img[:, :, 0] = np.mod(orig_img[:, :, 0], 1.)
      if t % 1 == 0:
        shifted_lines = orig_img[1:, :, :]
        first_line = orig_img[:1, :, :]
        orig_img = np.concatenate([shifted_lines, first_line], axis=0)
      img = np.copy(orig_img[:-100])
      # img, shift_pixels, all_start_y, all_deltas = pixel_shift(
      # # img,
      # orig_img,
      # shift_up=True,
      # num_shifts=num_shifts,
      # shift_pixels=shift_pixels,
      # all_start_y=all_start_y,
      # all_deltas=all_deltas)
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
images = load_png_images(fnames=[fname])
# images = load_png_images(fnames=['waterfall.jpg'])
# images = load_png_images(fnames=['vortex.jpg'])
# images = load_png_images(fnames=['killer_klowns.PNG'])
glitch_gif(images, fname, make_reversible=True)
