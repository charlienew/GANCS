def _glorot_initializer(prev_units, num_units, stddev_factor=1.0):
    """Initialization in the style of Glorot 2010.

    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
    stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
    return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)


def add_relu(prev_layer):
    """Adds a ReLU activation function to this model""" 
    out = tf.nn.relu(prev_layer)
    return out    

def _get_num_inputs(prev_layer):
        return int(prev_layer.get_shape()[-1])

def _glorot_initializer_conv2d(prev_units, num_units, mapsize, stddev_factor=1.0):
    """Initialization in the style of Glorot 2010.

    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

    stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
    return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)   


def add_batch_norm(prev_layer, scale=False):
    """Adds a batch normalization layer to this model.

    See ArXiv 1502.03167v3 for details."""

    # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
    out = tf.contrib.layers.batch_norm(prev_layer, scale=scale)

    return out

def add_conv2d(prev_layer, num_units, mapsize=1, stride=1, stddev_factor=1.0):
    """Adds a 2D convolutional layer."""

    assert len(prev_layer.get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        
    prev_units = _get_num_inputs()
            
            # Weight term and convolution
    initw  = _glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
    weight = tf.get_variable('weight', initializer=initw)
    

            # Bias term
    initb  = tf.constant(0.0, shape=[num_units])
    bias   = tf.get_variable('bias', initializer=initb)
            
    return weight, bias


def add_sum(prev_layer, term):
        """Adds a layer that sums the top layer with the given term"""

        
        prev_shape = prev_layer.get_shape()
        term_shape = term.get_shape()
            #print("%s %s" % (prev_shape, term_shape))
        assert prev_shape == term_shape and "Can't sum terms with a different size"
        out = tf.add(prev_layer, term)
        return out

def add_residual_block(prev_layer, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
    """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

    assert len(prev_layer.get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
    if num_units != int(prev_layer.get_shape()[3]):
        next_layer = add_conv2d(prev_layer, num_units, mapsize=1, stride=1, stddev_factor=1.)
    else:
        next_layer = prev_layer

    bypass = next_layer

        # Residual block
    for _ in range(num_layers):
        next_layer = add_batch_norm(next_layer)
        next_layer = add_relu(next_layer)
        next_layer = add_conv2d(next_layer, num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

    next_layer = add_sum(next_layer, bypass)
    next_layer = add_relu(next_layer)

    return next_layer


def _discriminator_model(disc_input, layer_output_skip=5, hybrid_disc=0):

    # update 05092017, hybrid_disc consider whether to use hybrid space for discriminator
    # to study the kspace distribution/smoothness properties

    # Fully convolutional model
    mapsize = 3
    layers  = [8, 16, 32, 64]#[64, 128, 256, 512]

    old_vars = tf.global_variables()#tf.all_variables() , all_variables() are deprecated

    # get discriminator input
    #disc_hybird = 2 * disc_input - 1
    #print(hybrid_disc, 'discriminator input dimensions: {0}'.format(disc_hybird.get_shape()))
    next_layer = disc_input       

    # discriminator network structure
    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0

        next_layer = add_conv2d(next_layer, nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        next_layer = add_batch_norm(next_layer)
        next_layer = model.add_relu()

    # Finalization a la "all convolutional net"
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_mean()

    new_vars  = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
    disc_vars = list(set(new_vars) - set(old_vars))

    #select output
    output_layers = [model.outputs[0]] + model.outputs[1:-1][::layer_output_skip] + [model.outputs[-1]]

    return model, disc_vars, output_layers