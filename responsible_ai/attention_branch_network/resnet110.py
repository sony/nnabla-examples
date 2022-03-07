import nnabla.functions as F
import nnabla.parametric_functions as PF


def resnet110(x, test=False):
    # Input:x -> 3,32,32

    # ImageAugmentation
    if not test:
        h = F.image_augmentation(x, shape=(3, 32, 32), min_scale=1, max_scale=1, angle=0.26,
                                 aspect_ratio=1, distortion=0, flip_lr=True, brightness=0, contrast=1, noise=0)
        h = F.random_shift(h, shifts=(4, 4))
        h = F.random_flip(h, axes=(3,))
    else:
        h = x

    # MulScalar_2
    h = F.mul_scalar(h, val=0.01735)
    # AddScalar_2
    h = F.add_scalar(h, val=-1.99)
    # Convolution -> 16,32,32
    h = PF.convolution(h, 16, (3, 3), (1, 1),
                       with_bias=False, name='Convolution')
    # RepeatStart_1
    for i in range(18):

        # BatchNormalization_2
        h1 = PF.batch_normalization(
            h, (1,), 0.5, 0.0001, not test, name='BatchNormalization_2[' + str(i) + ']')
        # ReLU_2
        h1 = F.relu(h1, True)
        # Convolution_2
        h1 = PF.convolution(h1, 16, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_2[' + str(i) + ']')
        # BatchNormalization_11
        h1 = PF.batch_normalization(
            h1, (1,), 0.5, 0.0001, not test, name='BatchNormalization_11[' + str(i) + ']')
        # ReLU_6
        h1 = F.relu(h1, True)
        # Convolution_3
        h1 = PF.convolution(h1, 16, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_3[' + str(i) + ']')

        # Add2 -> 16,32,32
        h2 = F.add2(h, h1, inplace=False)
        # RepeatEnd_3
        h = h2
    # BatchNormalization_3
    h2 = PF.batch_normalization(
        h2, (1,), 0.5, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_3
    h2 = F.relu(h2, inplace=True)

    # Convolution_15 -> 32,16,16
    h3 = PF.convolution(h2, 32, (3, 3), (1, 1), (2, 2),
                        with_bias=False, name='Convolution_15')

    # Convolution_4 -> 32,16,16
    h4 = PF.convolution(h2, 32, (3, 3), (1, 1), (2, 2),
                        with_bias=False, name='Convolution_4')
    # BatchNormalization_4
    h4 = PF.batch_normalization(
        h4, (1,), 0.5, 0.0001, not test, name='BatchNormalization_4')
    # ReLU_4
    h4 = F.relu(h4, True)
    # Convolution_10
    h4 = PF.convolution(h4, 32, (3, 3), (1, 1),
                        with_bias=False, name='Convolution_10')

    # Add2_5 -> 32,16,16
    h4 = F.add2(h4, h3, inplace=True)
    # RepeatStart
    for i in range(17):

        # BatchNormalization_5
        h5 = PF.batch_normalization(
            h4, (1,), 0.5, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # ReLU_5
        h5 = F.relu(h5, True)
        # Convolution_5
        h5 = PF.convolution(h5, 32, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_5[' + str(i) + ']')
        # BatchNormalization_6
        h5 = PF.batch_normalization(
            h5, (1,), 0.5, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_7
        h5 = F.relu(h5, True)
        # Convolution_6
        h5 = PF.convolution(h5, 32, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_6[' + str(i) + ']')

        # Add2_2 -> 32,16,16
        h6 = F.add2(h4, h5, inplace=False)
        # RepeatEnd
        h4 = h6
    # BatchNormalization_7
    h6 = PF.batch_normalization(
        h6, (1,), 0.5, 0.0001, not test, name='BatchNormalization_7')
    # ReLU_9
    h6 = F.relu(h6, inplace=True)

    # Convolution_17 -> 64,8,8
    h7 = PF.convolution(h6, 64, (3, 3), (1, 1), (2, 2),
                        with_bias=False, name='Convolution_17')

    # Convolution_7 -> 64,8,8
    h8 = PF.convolution(h6, 64, (3, 3), (1, 1), (2, 2),
                        with_bias=False, name='Convolution_7')
    # BatchNormalization_12
    h8 = PF.batch_normalization(
        h8, (1,), 0.5, 0.0001, not test, name='BatchNormalization_12')
    # ReLU_11
    h8 = F.relu(h8, True)
    # Convolution_11
    h8 = PF.convolution(h8, 64, (3, 3), (1, 1),
                        with_bias=False, name='Convolution_11')

    # Add2_6 -> 64,8,8
    h8 = F.add2(h8, h7, inplace=True)
    # RepeatStart_2
    for i in range(17):

        # BatchNormalization_8
        h9 = PF.batch_normalization(
            h8, (1,), 0.5, 0.0001, not test, name='BatchNormalization_8[' + str(i) + ']')
        # ReLU_8
        h9 = F.relu(h9, inplace=True)
        # Convolution_8
        h9 = PF.convolution(h9, 64, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_8[' + str(i) + ']')
        # BatchNormalization_9
        h9 = PF.batch_normalization(
            h9, (1,), 0.5, 0.0001, not test, name='BatchNormalization_9[' + str(i) + ']')
        # ReLU_10
        h9 = F.relu(h9, inplace=True)
        # Convolution_9
        h9 = PF.convolution(h9, 64, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_9[' + str(i) + ']')

        # Add2_3 -> 64,8,8
        h10 = F.add2(h8, h9, inplace=False)
        # RepeatEnd_2
        h8 = h10
    # BatchNormalization_13
    h10 = PF.batch_normalization(
        h10, (1,), 0.5, 0.0001, not test, name='BatchNormalization_13')
    # AveragePooling -> 64,1,1
    h10 = F.average_pooling(h10, kernel=(8, 8), stride=(8, 8))
    # ReLU_12
    h10 = F.relu(h10, inplace=True)
    # Affine -> 10
    h10 = PF.affine(h10, n_outmaps=(10,), name='Affine')
    return h10
