from loader import MNIST
mndata = MNIST('../data/input/')
trn_img, trn_labels = mndata.load_training()
tst_img, tst_labels = mndata.load_testing()
