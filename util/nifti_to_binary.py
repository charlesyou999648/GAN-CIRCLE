import argparse
import nibabel as nib
import numpy as np
import os
import tensorflow as tf

from glob import glob
from nilearn.image import reorder_img, resample_img

from parser import nifti_to_binary_parser

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def np_to_binary(array):
    shape = np.array(array.shape, np.int32)
    return shape.tobytes(), array.tobytes()

def write_to_tfrecord(array, tfrecord_file):
    shape, binary = np_to_binary(array)
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    features = tf.train.Features(feature={'shape': _bytes_feature(shape), 
                                          'array': _bytes_feature(binary)})
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())
    writer.close()

def read_from_tfrecord(filenames, shuffle=True):
    queue = tf.train.string_input_producer(filenames, shuffle=shuffle, 
                                           name='queue')
    reader = tf.TFRecordReader()
    _, record = reader.read(queue)
    
    features = {'shape': tf.FixedLenFeature([], tf.string), 
                'array': tf.FixedLenFeature([], tf.string)}
    tfrecord_features = tf.parse_single_example(record, features=features, 
                                                name='features')
    array = tf.decode_raw(tfrecord_features['array'], tf.float32)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    array = tf.reshape(array, shape)
    return array

def npy_to_nifti(subject, npy_path, affine_path, nifti_path, basename, axis):
    pattern = '{:s}*.npy'.format(basename)
    image_paths = sorted(glob(os.path.join(npy_path, pattern)))
    affine_path = os.path.join(affine_path, '{:s}.npy'.format(subject))
    
    affine = np.load(affine_path)
    image_list = []
    for path in image_paths:
        image = np.squeeze(np.load(path))
        image_list.append(image)
    
    array_data = np.stack(image_list, axis=axis)
    array_img = nib.Nifti1Image(array_data, affine)
    out_path = os.path.join(nifti_path, '{:s}.nii.gz'.format(basename))
    nib.save(array_img, out_path)

def resize(image, shape, interpolation='continuous'):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    output_shape = np.asarray(shape)
    spacing = input_shape / output_shape
    
    ras = reorder_img(image, resample=interpolation)
    affine = np.copy(ras.affine)
    affine[:3, :3] = ras.affine[:3, :3] * np.diag(spacing)
    
    resampled = resample_img(ras, target_affine=affine, target_shape=shape, 
                             interpolation=interpolation)
    return resampled

def main():
    args = nifti_to_binary_parser().parse_args()
    input_path = args.input
    affine_path = args.affine
    slice_path = args.slice
    x, y, z = args.x, args.y, args.z
    d = args.direction
    
    if not os.path.exists(affine_path):
        os.makedirs(affine_path)
    if not os.path.exists(slice_path):
        os.makedirs(slice_path)
    
    for filename in os.listdir(input_path):
        path = os.path.join(input_path, filename)
        subj_name, ext = os.path.splitext(filename)
        if ext == '.gz':
            subj_name = os.path.splitext(subj_name)[0]
        
        #Load image
        img = nib.load(path)
        img = resize(img, (x, y, z))
        affine = img.affine.astype(np.float32)
        data = img.get_data().astype(np.float32)
        
        #Transpose so that desired slices are in the first dimension
        axes = [d] + range(data.ndim)
        del axes[d + 1]
        data = data.transpose(axes)
        
        #Add channel dimension
        data = np.expand_dims(data, axis=3)
        
        #Save affine as .npy
        print('Saving {:s}...'.format(subj_name))
        np.save(os.path.join(affine_path, subj_name), affine)
        
        #Save slices as .tfrecord
        for i in range(data.shape[0]):
            slice_name = '{:s}-{:03d}'.format(subj_name, i)
            out = os.path.join(slice_path, '{:s}.tfrecord'.format(slice_name))
            write_to_tfrecord(data[i], out)
    
    print('Done.')

if __name__ == '__main__':
    main()

