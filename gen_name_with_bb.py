import pickle
import numpy as np
import os
import scipy
import scipy.misc

subset = 'test'
db_path = '/home/yfeng23/lab/StackGAN/Data/birds/CUB_200_2011/'

with open(db_path + '../' + subset + '/class_info.pickle', 'rb') as f:
  class_id = pickle.load(f)
with open(db_path + '../' + subset + '/filenames.pickle', 'rb') as f:
  filenames = pickle.load(f)

with open(db_path + 'images.txt', 'r') as f:
  names = list(f)
  names = [i.split(' ')[1][:-1] for i in names]
  #names = [i[:-4] for i in names]

with open(db_path + 'bounding_boxes.txt', 'r') as f:
  bb = list(f)
  bb = [i.split(' ')[1:] for i in bb]
  bb = [map(float, i) for i in bb]
  bb = [map(int, i) for i in bb]

with open(db_path + '../' + subset + '/char-CNN-RNN-embeddings.pickle', 'rb') as f:
  emb = pickle.load(f)


def get_image(image_path, image_size, is_crop=False, bbox=None):
  out = transform(imread(image_path), image_size, is_crop, bbox)
  return out


def custom_crop(img, bbox):
  # bbox = [x-left, y-top, width, height]
  imsiz = img.shape  # [height, width, channel]
  # if box[0] + box[2] >= imsiz[1] or\
  #     box[1] + box[3] >= imsiz[0] or\
  #     box[0] <= 0 or\
  #     box[1] <= 0:
  #     box[0] = np.maximum(0, box[0])
  #     box[1] = np.maximum(0, box[1])
  #     box[2] = np.minimum(imsiz[1] - box[0] - 1, box[2])
  #     box[3] = np.minimum(imsiz[0] - box[1] - 1, box[3])
  center_x = int((2 * bbox[0] + bbox[2]) / 2)
  center_y = int((2 * bbox[1] + bbox[3]) / 2)
  R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
  y1 = np.maximum(0, center_y - R)
  y2 = np.minimum(imsiz[0], center_y + R)
  x1 = np.maximum(0, center_x - R)
  x2 = np.minimum(imsiz[1], center_x + R)
  img_cropped = img[y1:y2, x1:x2, :]
  return img_cropped


def transform(image, image_size, is_crop, bbox):
  image = colorize(image)
  if is_crop:
    image = custom_crop(image, bbox)
  #
  transformed_image = \
    scipy.misc.imresize(image, [image_size, image_size], 'bicubic')
  return np.array(transformed_image)


def imread(path):
  img = scipy.misc.imread(path)
  if len(img.shape) == 0:
    raise ValueError(path + " got loaded as a dimensionless array!")
  return img.astype(np.float)


def colorize(img):
  if img.ndim == 2:
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate([img, img, img], axis=2)
  if img.shape[2] == 4:
    img = img[:, :, 0:3]
  return img


with open(db_path + 'my' + subset + '.txt', 'w') as f:
  for i, c, e in zip(filenames, class_id, emb):
    idx = names.index(i + '.jpg')
    i = i.split('/')[1]
    f.write('%s,%d\n' % (i, c))
    na = os.path.split(i)[1]

    #np.save(db_path + 'sentence/' + na, e)

    # f_name = db_path + 'images/' + names[idx]
    # img = get_image(f_name, 304, is_crop=True, bbox=bb[idx])
    # img = img.astype('uint8')
    # scipy.misc.imsave(db_path + '/hr_imgs/' + na + '.png', img)
    # lr_img = scipy.misc.imresize(img, [76, 76], 'bicubic')
    # scipy.misc.imsave(db_path + '/lr_imgs/' + na + '.png', lr_img)
