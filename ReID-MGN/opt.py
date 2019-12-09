import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="../../Market",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')

parser.add_argument('--augment',
                    action='append',
                    help='the data augmentation data path')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    type=int,
                    default=500,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=4,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    help='the batch size for test')

parser.add_argument('--random-erasing',
                    default=True,
                    help='whether to apply random erasing')

parser.add_argument('--use-centerloss',
                    default=False,
                    help='whether to use centerloss')

parser.add_argument('--bnn-neck',
                    default=False,
                    help='Whether to use bnnneck')

parser.add_argument('--label-smooth-ce',
                    default=False,
                    help='Whether to use Label Smoothing Cross Entropy or normal entropy')

opt = parser.parse_args()
