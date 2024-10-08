# Testing x2, x3, x4
python test.py -opt options/test/test_ESDANx2.yml
python test.py -opt options/test/test_ESDANx3.yml
python test.py -opt options/test/test_ESDANx4.yml


# Training x2, x3, x4
python train.py -opt options/train/train_ESDANx2.yml
python train.py -opt options/train/train_ESDANx3.yml
python train.py -opt options/train/train_ESDANx4.yml


# Training SRResNet_PA or RCAN_PA
python train.py -opt options/train/train_SRResNet.yml
python train.py -opt options/train/train_RCAN.yml
