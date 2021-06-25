
def build_rnet_model(batch_norm=True, dropout=False):
    inputs = Input(shape=(64, 64, 3))
    
    r_layer = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(inputs)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)
    r_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(r_layer)
    
    r_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)
    r_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(r_layer)

    r_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)
    r_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(r_layer)
    
    r_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)
    
    r_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)

    r_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(dropout) : r_layer = Dropout(0.5)(r_layer)

    r_layer_out1 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1))(r_layer)
    r_layer_out1 = Softmax(axis=3)(r_layer_out1)

    r_layer_out2 = Conv2D(4, activation='sigmoid', kernel_size=(1, 1), strides=(1, 1))(r_layer)
    
    r_layer_out1 = Reshape(target_shape=(2,), name='probability')(r_layer_out1)
    r_layer_out2 = Reshape(target_shape=(4,), name='bbox_regression')(r_layer_out2)

    r_net = Model(inputs, [r_layer_out1, r_layer_out2], name='R-Net')

    return r_net
