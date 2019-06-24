import argparse
import os
import h5py
from models.cyclegan import CycleGAN
from util.parser import training_parser,testing_parser

def main():
    args = testing_parser().parse_args()

    name = args.name
    model = args.model
    
    #File paths
    train_dir = os.path.join('train/', name)

    f = h5py.File('./test_v2nds_2D.h5', 'r')
    test_data = f.get('LR')
    test_label = f.get('SR')

    args.w = test_data.shape[1]
    args.h = test_data.shape[2]
    args.c = test_data.shape[3]
    args.ow = test_label.shape[1]
    args.oh = test_label.shape[2]
    args.oc = test_label.shape[3]

    print(test_data,test_label)
    cyclegan = CycleGAN(args, False, None)

    ## b -> label  a -> data
    b2a, a2b, aba, bab = cyclegan.test(test_data.value,test_label.value)  # return array
    print(b2a.shape)
    print(a2b.shape)
    print(aba.shape)
    print(bab.shape)

if __name__ == '__main__':
    main()

