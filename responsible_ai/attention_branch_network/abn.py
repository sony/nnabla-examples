import nnabla.functions as F
import nnabla.parametric_functions as PF


def abn(x, test=False):
    # Input:x -> 3,32,32

    if not test:
        h1 = F.image_augmentation(x, shape=(3, 32, 32), min_scale=1, max_scale=1, angle=0.26,
                                  aspect_ratio=1, distortion=0, flip_lr=True, brightness=0, contrast=1, noise=0)
        h1 = F.random_shift(h1, (4, 4))
        h1 = F.random_flip(h1, (3,))
    else:
        h1 = x

    # MulScalar_2
    h1 = F.mul_scalar(h1, 0.01735)
    # AddScalar_2
    h1 = F.add_scalar(h1, -1.99)
    # Convolution -> 16,32,32
    h1 = PF.convolution(h1, 16, (3, 3), (1, 1),
                        with_bias=False, name='Convolution')
    # RepeatStart_1
    for i in range(18):

        # BatchNormalization_2
        h2 = PF.batch_normalization(
            h1, (1,), 0.5, 0.0001, not test, name='BatchNormalization_2[' + str(i) + ']')
        # ReLU_2
        h2 = F.relu(h2, True)
        # Convolution_2
        h2 = PF.convolution(h2, 16, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_2[' + str(i) + ']')
        # BatchNormalization_11
        h2 = PF.batch_normalization(
            h2, (1,), 0.5, 0.0001, not test, name='BatchNormalization_11[' + str(i) + ']')
        # ReLU_6
        h2 = F.relu(h2, True)
        # Convolution_3
        h2 = PF.convolution(h2, 16, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_3[' + str(i) + ']')

        # Add2 -> 16,32,32
        h3 = F.add2(h1, h2, False)
        # RepeatEnd_1
        h1 = h3
    # BatchNormalization_3
    h3 = PF.batch_normalization(
        h3, (1,), 0.5, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_3
    h3 = F.relu(h3, True)

    # Convolution_15 -> 32,16,16
    h4 = PF.convolution(h3, 32, (3, 3), (1, 1), (2, 2),
                        with_bias=False, name='Convolution_15')

    # Convolution_4 -> 32,16,16
    h5 = PF.convolution(h3, 32, (3, 3), (1, 1), (2, 2),
                        with_bias=False, name='Convolution_4')
    # BatchNormalization_4
    h5 = PF.batch_normalization(
        h5, (1,), 0.5, 0.0001, not test, name='BatchNormalization_4')
    # ReLU_4
    h5 = F.relu(h5, True)
    # Convolution_10
    h5 = PF.convolution(h5, 32, (3, 3), (1, 1),
                        with_bias=False, name='Convolution_10')

    # Add2_5 -> 32,16,16
    h5 = F.add2(h5, h4, True)
    # RepeatStart_2
    for i in range(17):

        # BatchNormalization_5
        h6 = PF.batch_normalization(
            h5, (1,), 0.5, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # ReLU_5
        h6 = F.relu(h6, True)
        # Convolution_5
        h6 = PF.convolution(h6, 32, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_5[' + str(i) + ']')
        # BatchNormalization_6
        h6 = PF.batch_normalization(
            h6, (1,), 0.5, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_7
        h6 = F.relu(h6, True)
        # Convolution_6
        h6 = PF.convolution(h6, 32, (3, 3), (1, 1), with_bias=False,
                            name='Convolution_6[' + str(i) + ']')

        # Add2_2 -> 32,16,16
        h7 = F.add2(h5, h6, False)
        # RepeatEnd_2
        h5 = h7
    # BatchNormalization_7
    h7 = PF.batch_normalization(
        h7, (1,), 0.5, 0.0001, not test, name='BatchNormalization_7')
    # ReLU_9
    h7 = F.relu(h7, True)

    # Convolution_17 -> 64,8,8
    h8 = PF.convolution(h7, 64, (3, 3), (1, 1), (2, 2),
                        with_bias=False, name='Convolution_17')

    # Convolution_7 -> 64,8,8
    h9 = PF.convolution(h7, 64, (3, 3), (1, 1), (2, 2),
                        with_bias=False, name='Convolution_7')
    # BatchNormalization_12
    h9 = PF.batch_normalization(
        h9, (1,), 0.5, 0.0001, not test, name='BatchNormalization_12')
    # ReLU_11
    h9 = F.relu(h9, True)
    # Convolution_11
    h9 = PF.convolution(h9, 64, (3, 3), (1, 1),
                        with_bias=False, name='Convolution_11')

    # Add2_6 -> 64,8,8
    h10 = F.add2(h9, h8, True)

    # RepeatStart_3
    for i in range(17):

        # BatchNormalization_14
        h11 = PF.batch_normalization(
            h10, (1,), 0.5, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')
        # ReLU_13
        h11 = F.relu(h11, True)
        # Convolution_16
        h11 = PF.convolution(
            h11, 64, (3, 3), (1, 1), with_bias=False, name='Convolution_16[' + str(i) + ']')
        # BatchNormalization_15
        h11 = PF.batch_normalization(
            h11, (1,), 0.5, 0.0001, not test, name='BatchNormalization_15[' + str(i) + ']')
        # ReLU_14
        h11 = F.relu(h11, True)
        # Convolution_18
        h11 = PF.convolution(
            h11, 64, (3, 3), (1, 1), with_bias=False, name='Convolution_18[' + str(i) + ']')

        # Add2_4 -> 64,8,8
        h12 = F.add2(h10, h11, False)
        # RepeatEnd_3
        h10 = h12
    # BatchNormalization_16
    h12 = PF.batch_normalization(
        h12, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # ReLU_15
    h12 = F.relu(h12, True)
    # Convolution_14
    h12 = PF.convolution(h12, 64, (1, 1), (0, 0), name='Convolution_14')
    # BatchNormalization_10
    h12 = PF.batch_normalization(
        h12, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # ReLU
    h12 = F.relu(h12, True)

    # Convolution_12 -> 10,8,8
    h13 = PF.convolution(h12, 10, (1, 1), (0, 0), name='Convolution_12')

    # Convolution_13 -> 1,8,8
    h14 = PF.convolution(h12, 1, (3, 3), (1, 1), name='Convolution_13')
    # BatchNormalization
    h14 = PF.batch_normalization(
        h14, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # GlobalAveragePooling -> 10,1,1
    h13 = F.global_average_pooling(h13)
    # Reshape -> 10
    h13 = F.reshape(h13, (h13.shape[0], 10,))
    # Sigmoid_Attention_Map
    h14 = F.sigmoid(h14)

    # Mul2 -> 64,8,8
    h15 = F.mul2(h14, h10)

    # Add2_7 -> 64,8,8
    h16 = F.add2(h10, h15, False)
    # RepeatStart_4
    for i in range(17):

        # BatchNormalization_8
        h17 = PF.batch_normalization(
            h16, (1,), 0.5, 0.0001, not test, name='BatchNormalization_8[' + str(i) + ']')
        # ReLU_8
        h17 = F.relu(h17, True)
        # Convolution_8
        h17 = PF.convolution(
            h17, 64, (3, 3), (1, 1), with_bias=False, name='Convolution_8[' + str(i) + ']')
        # BatchNormalization_9
        h17 = PF.batch_normalization(
            h17, (1,), 0.5, 0.0001, not test, name='BatchNormalization_9[' + str(i) + ']')
        # ReLU_10
        h17 = F.relu(h17, True)
        # Convolution_9
        h17 = PF.convolution(
            h17, 64, (3, 3), (1, 1), with_bias=False, name='Convolution_9[' + str(i) + ']')

        # Add2_3 -> 64,8,8
        h18 = F.add2(h16, h17, False)
        # RepeatEnd_4
        h16 = h18
    # BatchNormalization_13
    h18 = PF.batch_normalization(
        h18, (1,), 0.5, 0.0001, not test, name='BatchNormalization_13')
    # AveragePooling -> 64,1,1
    h18 = F.average_pooling(h18, (8, 8), (8, 8))
    # ReLU_12
    h18 = F.relu(h18, True)
    # Affine -> 10
    h18 = PF.affine(h18, (10,), name='Affine')
    return h13, h18
