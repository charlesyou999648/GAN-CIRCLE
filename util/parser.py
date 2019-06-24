import argparse

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        error = '{:s} is an invalid positive int value'.format(value)
        raise argparse.ArgumentTypeError(error)
    return ivalue

def positive_or_zero_int(value):
    ivalue = int(value)
    if ivalue < 0:
        error = '{:s} must be an int greater than or equal to 0'.format(value)
        raise argparse.ArgumentTypeError(error)
    return ivalue

def restricted_int(value):
    ivalue = int(value)
    if ivalue < 256:
        error = '{:s} must be an int greater than or equal to 256'.format(value)
        raise argparse.ArgumentTypeError(error)
    return ivalue

def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        error = '{:s} is an invalid positive float value'.format(value)
        raise argparse.ArgumentTypeError(error)
    return fvalue

def general_parser():
    parser = argparse.ArgumentParser(description='CycleGAN arguments.')
    general = parser.add_argument_group('general', 'General arguments')
    
    general.add_argument('-u', '--min_vox', action='store', nargs='?',
                         default=-193.0, type=float, 
                         help=('Minimum voxel intensity. Default: '
                               '-137.28571428571004'))
    general.add_argument('-v', '--max_vox', action='store', nargs='?',
                         default=19152.0, type=float, 
                         help=('Maximum voxel intensity. Default: '
                               '17662.535068643472'))
    general.add_argument('-n', '--name', action='store', nargs='?', 
                         default='gad', type=str, 
                         help='Dataset name. Default: "gad"')
    general.add_argument('-a', '--axis', action='store', nargs='?', default=1, 
                         type=positive_int, choices=range(0, 3), 
                         help=('Slice direction. Choose 0 for sagittal, 1 for '
                               'coronal, or 2 for axial. Default: 1'))
    
    return parser

def training_parser():
    parser = general_parser()
    parser.description = 'CycleGAN training arguments.'
    train = parser.add_argument_group('train', 'Training arguments')

    train.add_argument('-x', '--lambda_a', action='store', nargs='?',
                       default=1.0, type=positive_float, 
                       help=('Weight for forward cyclic loss (A -> B -> A). '
                             'Default: 10.0'))
    train.add_argument('-y', '--lambda_b', action='store', nargs='?',
                       default=1.0, type=positive_float, 
                       help=('Weight for backward cyclic loss (B -> A -> B). '
                             'Default: 10.0'))
    train.add_argument('-p', '--pool_size', action='store', nargs='?', default=50,
                       type=positive_int, 
                       help=('Number of fake images to store for calculating '
                             'loss. Deault: 50'))
    train.add_argument('-lg', '--base_lr_g', action='store', nargs='?', default=1e-5,
                       type=positive_or_zero_int, 
                       help=('Base learning rate. Default: 0.0002'))
    train.add_argument('-ld', '--base_lr_d', action='store', nargs='?', default=1e-5,
                       type=positive_or_zero_int,
                       help=('Base learning rate. Default: 0.0002'))
    train.add_argument('-e', '--epochs', action='store', nargs='?', 
                       default=150, type=positive_int, 
                       help='Max epochs. Default: 200')
    train.add_argument('-s', '--save', action='store', nargs='?', default=0, 
                       type=positive_or_zero_int, 
                       help=('Number of training images to save for random '
                             'subjects. Default: 0'))
    train.add_argument('-b', '--batch', action='store', nargs='?', default=64,
                       type=positive_int, help='Batch size. Default: 1')
    train.add_argument('-r', '--restore', action='store', nargs='?', type=str, 
                       help=('Name of folder containing checkpoints in '
                             'train/name/. Ignore for new training run.'))
    train.add_argument('-c', '--channel', action='store', nargs='?', type=int, default = 1, 
                       help=('Number of input/output channel'))

    return parser

def testing_parser():
    # parser = general_parser()
    parser = training_parser()
    parser.description = 'CycleGAN testing arguments.'
    test = parser.add_argument_group('test', 'Testing arguments')
    
    test.add_argument('-i', '--ids1', action='store', nargs='*', type=str, 
                      help='Subject IDs from group A to test.')
    test.add_argument('-j', '--ids2', action='store', nargs='*', type=str, 
                      help='Subject IDs from group B to test.')
    test.add_argument('-m', '--model', action='store', nargs='?', 
                      default='pretrained', type=str, 
                      help=('Name of folder containing checkpoints in '
                            'train/name/. Default: "pretrained"'))
    
    return parser

def nifti_to_binary_parser():
    parser = argparse.ArgumentParser(description='NIfTI to binary arguments.')
    
    optional = parser._action_groups.pop()
    optional.add_argument('-x', '--x', action='store', nargs='?', 
                          default=256, type=positive_int, 
                          help=('Output size x. Image will be resampled prior '
                                'to input into network. Assumes image is RAS '
                                'ordered. Default: 256'))
    optional.add_argument('-y', '--y', action='store', nargs='?', 
                          default=256, type=positive_int, 
                          help=('Output size y. Image will be resampled prior ' 
                                'to input into network. Assumes image is RAS '
                                'ordered. Default: 256'))
    optional.add_argument('-z', '--z', action='store', nargs='?', 
                          default=256, type=positive_int, 
                          help=('Output size z. Image will be resampled prior '
                                'to input into network. Assumes image is RAS '
                                'ordered. Default: 256'))
    optional.add_argument('-d', '--direction', action='store', nargs='?', 
                          default=1, type=positive_int, choices=range(0, 3), 
                          help=('Slice direction. Choose 0 for sagittal, 1 for '
                                'coronal, or 2 for axial. Default: 1'))
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', action='store', type=str, 
                          required=True, help='Input dir containing NIfTIs.')
    required.add_argument('-a', '--affine', action='store', type=str, 
                          required=True, help='Output dir for affines.')
    required.add_argument('-s', '--slice', action='store', type=str, 
                          required=True, help='Output dir for slices.')
    
    parser._action_groups.append(optional)
    return parser

if __name__=='__main__':

    tr_parser = training_parser().parse_args()
    print(tr_parser)
    te_parser = testing_parser().parse_args()
    print(te_parser)