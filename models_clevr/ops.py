from __future__ import division
import tensorflow as tf

from .config import cfg

eps = 1e-20
inf = 1e30


def relu(inp):
    '''
    Performs a variant of ReLU based on cfg.RELU
        PRM for PReLU
        ELU for ELU
        LKY for Leaky ReLU
        otherwise, standard ReLU
    '''
    if cfg.RELU == "PRM":
        with tf.variable_scope(None, default_name="prelu"):
            alpha = tf.get_variable(
                "alpha", shape=inp.get_shape()[-1],
                initializer=tf.constant_initializer(0.25))
            pos = tf.nn.relu(inp)
            neg = - (alpha * tf.nn.relu(-inp))
            output = pos + neg
    elif cfg.RELU == "ELU":
        output = tf.nn.elu(inp)
    elif cfg.RELU == "SELU":
        output = tf.nn.selu(inp)
    elif cfg.RELU == "LKY":
        # output = tf.nn.leaky_relu(inp, cfg.RELU_ALPHA)
        output = tf.maximum(inp, cfg.RELU_ALPHA * inp)
    elif cfg.RELU == "STD":  # STD
        output = tf.nn.relu(inp)

    return output


activations = {
    "NON":      tf.identity,
    "TANH":     tf.tanh,
    "SIGMOID":  tf.sigmoid,
    "RELU":     relu,
    "ELU":      tf.nn.elu
}


def hingeLoss(labels, logits):
    maxLogit = tf.reduce_max(logits * (1 - labels), axis=1, keepdims=True)
    losses = tf.nn.relu((1 + maxLogit - logits) * labels)
    # reduce_max reduce sum will also work
    losses = tf.reduce_sum(losses, axis=1)
    return losses


def inter2logits(interactions, dim, sumMod="LIN", dropout=1.0, name="",
                 reuse=None):
    '''
    Transform vectors to scalar logits.

    Args:
        interactions: input vectors
        [batchSize, N, dim]

        dim: dimension of input vectors

        sumMod: LIN for linear transformation to scalars.
                SUM to sum up vectors entries to get scalar logit.

        dropout: dropout value over inputs (for linear case)

    Return matching scalar for each interaction.
    [batchSize, N]
    '''
    with tf.variable_scope("inter2logits" + name, reuse=reuse):
        if sumMod == "SUM":
            logits = tf.reduce_sum(interactions, axis=-1)
        else:  # "LIN"
            logits = linear(
                interactions, dim, 1, dropout=dropout, name="logits")
    return logits


def inter2att(interactions, dim, dropout=1.0, mask=None, sumMod="LIN", name="",
              reuse=None):
    '''
    Transforms vectors to probability distribution.
    Calls inter2logits and then softmax over these.

    Args:
        interactions: input vectors
        [batchSize, N, dim]

        dim: dimension of input vectors

        sumMod: LIN for linear transformation to scalars.
                SUM to sum up vectors entries to get scalar logit.

        dropout: dropout value over inputs (for linear case)

    Return attention distribution over interactions.
    [batchSize, N]
    '''
    with tf.variable_scope("inter2att" + name, reuse=reuse):
        logits = inter2logits(
            interactions, dim, dropout=dropout, sumMod=sumMod)
        if mask is not None:
            logits = expMask(logits, mask)
        attention = tf.nn.softmax(logits)
    return attention


def getWeight(shape, name=""):
    '''
    Initializes a weight matrix variable given a shape and a name.
    Uses random_normal initialization if 1d, otherwise uses xavier.
    '''
    with tf.variable_scope("weights"):
        initializer = tf.contrib.layers.xavier_initializer()
        # if len(shape) == 1: # good?
        #     initializer=tf.random_normal_initializer()
        W = tf.get_variable(
            "weight"+name, shape=shape, initializer=initializer)

    return W


def getKernel(shape, name=""):
    '''
    Initializes a weight matrix variable given a shape and a name. Uses xavier
    '''
    with tf.variable_scope("kernels"):
        initializer = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable(
            "kernel"+name, shape=shape, initializer=initializer)
    return W


def getBias(shape, name=""):
    '''
    Initializes a bias variable given a shape and a name.
    '''
    with tf.variable_scope("biases"):
        initializer = tf.zeros_initializer()
        b = tf.get_variable("bias"+name, shape=shape, initializer=initializer)
    return b


def linear(inp, inDim, outDim, dropout=1.0, batchNorm=None, addBias=True,
           bias=0.0, act="NON", actLayer=True, actDropout=1.0, retVars=False,
           name="", reuse=None):
    '''
    linear transformation.

    Args:
        inp: input to transform
        inDim: input dimension
        outDim: output dimension
        dropout: dropout over input
        batchNorm: if not None, applies batch normalization to inputs
        addBias: True to add bias
        bias: initial bias value
        act: if not None, activation to use after linear transformation
        actLayer: if True and act is not None, applies another linear
                  transformation on top of previous
        actDropout: dropout to apply in the optional second linear
                    transformation
        retVars: if True, return parameters (weight and bias)

    Returns linear transformation result.
    '''
    # batchNorm = {"decay": float, "train": Tensor}
    # actLayer: if activation is not non, stack another linear layer
    # maybe change naming scheme such that if name = "" than use it as
    # default_name (-->unique?)
    with tf.variable_scope("linearLayer" + name, reuse=reuse):
        W = getWeight((inDim, outDim) if outDim > 1 else (inDim, ))
        b = getBias((outDim, ) if outDim > 1 else ()) + bias

        if batchNorm is not None:
            inp = tf.contrib.layers.batch_norm(
                inp, decay=batchNorm["decay"], center=True, scale=True,
                is_training=batchNorm["train"], updates_collections=None)
            # tf.layers.batch_normalization, axis -1 ?

        inp = tf.nn.dropout(inp, dropout)

        if outDim > 1:
            output = multiply(inp, W)
        else:
            output = tf.reduce_sum(inp * W, axis=-1)

        if addBias:
            output += b

        output = activations[act](output)

        # good?
        if act != "NON" and actLayer:
            output = linear(output, outDim, outDim, dropout=actDropout,
                            batchNorm=batchNorm, addBias=addBias, act="NON",
                            actLayer=False, name=name + "_2", reuse=reuse)

    if retVars:
        return (output, (W, b))

    return output


def FCLayer(features, dims, batchNorm=None, dropout=1.0, act="RELU"):
    '''
    Computes Multi-layer feed-forward network.

    Args:
        features: input features
        dims: list with dimensions of network.
              First dimension is of the inputs, final is of the outputs.
        batchNorm: if not None, applies batchNorm
        dropout: dropout value to apply for each layer
        act: activation to apply between layers.
        NON, TANH, SIGMOID, RELU, ELU
    '''
    # no activation after last layer
    # batchNorm = {"decay": float, "train": Tensor}
    layersNum = len(dims) - 1

    for i in range(layersNum):
        features = linear(features, dims[i], dims[i+1], name="fc_%d" % i,
                          batchNorm=batchNorm, dropout=dropout)
        # not the last layer
        if i < layersNum - 1:
            features = activations[act](features)

    return features


def cnn(inp, inDim, outDim, batchNorm=None, dropout=1.0, addBias=True,
        kernelSize=None, stride=1, act="NON", name="", reuse=None):
    '''
    Computes convolution.

    Args:
        inp: input features
        inDim: input dimension
        outDim: output dimension
        batchNorm: if not None, applies batchNorm on inputs
        dropout: dropout value to apply on inputs
        addBias: True to add bias
        kernelSize: kernel size
        stride: stride size
        act: activation to apply on outputs
        NON, TANH, SIGMOID, RELU, ELU
    '''
    # batchNorm = {
    #    "decay": float, "train": Tensor, "center": bool, "scale": bool}
    # collections.namedtuple("batchNorm", ("decay", "train"))
    with tf.variable_scope("cnnLayer" + name, reuse=reuse):
        if kernelSize is None:
            kernelSize = 1
        kernelH = kernelW = kernelSize

        kernel = getKernel((kernelH, kernelW, inDim, outDim))
        b = getBias((outDim, ))

        if batchNorm is not None:
            inp = tf.contrib.layers.batch_norm(
                inp, decay=batchNorm["decay"], center=batchNorm["center"],
                scale=batchNorm["scale"], is_training=batchNorm["train"],
                updates_collections=None)

        inp = tf.nn.dropout(inp, dropout)

        output = tf.nn.conv2d(
            inp, filter=kernel, strides=[1, stride, stride, 1], padding="SAME")

        if addBias:
            output += b

        output = activations[act](output)

    return output


def CNNLayer(features, dims, batchNorm=None, dropout=1.0,
             kernelSizes=None, strides=None, act="RELU"):
    '''
    Computes Multi-layer convolutional network.

    Args:
        features: input features
        dims: list with dimensions of network.
              First dimension is of the inputs. Final is of the outputs.
        batchNorm: if not None, applies batchNorm
        dropout: dropout value to apply for each layer
        kernelSizes: list of kernel sizes for each layer. Default to
                     config.stemKernelSize
        strides: list of strides for each layer. Default to 1.
        act: activation to apply between layers.
        NON, TANH, SIGMOID, RELU, ELU
    '''
    # batchNorm = {
    #   "decay": float, "train": Tensor, "center": bool, "scale": bool}
    # activation after last layer
    layersNum = len(dims) - 1

    if kernelSizes is None:
        kernelSizes = [1 for i in range(layersNum)]

    if strides is None:
        strides = [1 for i in range(layersNum)]

    for i in range(layersNum):
        features = cnn(
            features, dims[i], dims[i+1], name="cnn_%d" % i,
            batchNorm=batchNorm,
            dropout=dropout, kernelSize=kernelSizes[i], stride=strides[i],
            act=act)

    return features


def expMask(seq, seqLength):
    '''
    Casts exponential mask over a sequence with sequence length.
    Used to prepare logits before softmax.
    '''
    maxLength = tf.shape(seq)[-1]
    mask = (tf.to_float(
        tf.logical_not(tf.sequence_mask(seqLength, maxLength)))) * (-inf)
    masked = seq + mask
    return masked


def att2Smry(attention, features):
    '''
    Sums up features using attention distribution to get a weighted average
    over them.
    '''
    return tf.reduce_sum(tf.expand_dims(attention, axis=-1)*features, axis=-2)


def mul(x, y, dim, dropout=1.0, proj=None, interMod="MUL", concat=None,
        mulBias=None, expandY=True, name="", reuse=None):
    '''
    "Enhanced" hadamard product between x and y:
    1. Supports optional projection of x, and y prior to multiplication.
    2. Computes simple multiplication, or a parametrized one, using diagonal
       of complete matrix (bi-linear)
    3. Optionally concatenate x or y or their projection to the multiplication
       result.

    Support broadcasting

    Args:
        x: left-hand side argument
        [batchSize, dim]

        y: right-hand side argument
        [batchSize, dim]

        dim: input dimension of x and y

        dropout: dropout value to apply on x and y

        proj: if not None, project x and y:
            dim: projection dimension
            shared: use same projection for x and y
            dropout: dropout to apply to x and y if projected

        interMod: multiplication type:
            "MUL": x * y
            "DIAG": x * W * y for a learned diagonal parameter W
            "BL": x' W y for a learned matrix W

        concat: if not None, concatenate x or y or their projection.

        mulBias: optional bias to stabilize multiplication (x * bias)(y * bias)

    Returns the multiplication result
    [batchSize, outDim] when outDim depends on the use of proj and cocnat
    arguments.
    '''
    # proj = {"dim": int, "shared": bool, "dropout": float}
    # # "act": str, "actDropout": float
    # # interMod = ["direct", "scalarW", "bilinear"] # "additive"
    # interMod = ["MUL", "DIAG", "BL", "ADD"]
    # concat = {"x": bool, "y": bool, "proj": bool}
    with tf.variable_scope("mul" + name, reuse=reuse):
        origVals = {"x": x, "y": y, "dim": dim}

        x = tf.nn.dropout(x, dropout)
        y = tf.nn.dropout(y, dropout)
        # projection
        if proj is not None:
            x = tf.nn.dropout(x, proj.get("dropout", 1.0))
            y = tf.nn.dropout(y, proj.get("dropout", 1.0))

            if proj["shared"]:
                xName, xReuse = "proj", None
                yName, yReuse = "proj", True
            else:
                xName, xReuse = "projX", None
                yName, yReuse = "projY", None

            x = linear(x, dim, proj["dim"], name=xName, reuse=xReuse)
            y = linear(y, dim, proj["dim"], name=yName, reuse=yReuse)
            dim = proj["dim"]
            projVals = {"x": x, "y": y, "dim": dim}
            proj["x"], proj["y"] = x, y

        if expandY:
            y = tf.expand_dims(y, axis=-2)
            # broadcasting to have the same shape
            y = tf.zeros_like(x) + y

        # multiplication
        if interMod == "MUL":
            if mulBias is None:
                mulBias = cfg.MUL_BIAS
            output = (x + mulBias) * (y + mulBias)
        elif interMod == "DIAG":
            W = getWeight((dim, ))  # change initialization?
            b = getBias((dim, ))
            output = x * W * y + b
        elif interMod == "BL":
            W = getWeight((dim, dim))
            b = getBias((dim, ))
            output = multiply(x, W) * y + b
        else:  # "ADD"
            output = tf.tanh(x + y)
        # concatenation

        if concat is not None:
            concatVals = projVals if concat.get("proj", False) else origVals
            if concat.get("x", False):
                output = tf.concat([output, concatVals["x"]], axis=-1)
                dim += concatVals["dim"]

            if concat.get("y", False):
                output = concat(output, concatVals["y"], expandY=expandY)
                dim += concatVals["dim"]

    return output, dim


def multiply(inp, W):
    '''
    Multiplies input inp of any depth by a 2d weight matrix.
    '''
    inDim = tf.shape(W)[0]
    outDim = tf.shape(W)[1]
    newDims = tf.concat([tf.shape(inp)[:-1], tf.fill((1,), outDim)], axis=0)

    inp = tf.reshape(inp, (-1, inDim))
    output = tf.matmul(inp, W)
    output = tf.reshape(output, newDims)

    return output


def concat(x, y, dim, mul=False, expandY=False):
    '''
    Concatenates x and y. Support broadcasting.
    Optionally concatenate multiplication of x * y
    '''
    if expandY:
        y = tf.expand_dims(y, axis=-2)
        # broadcasting to have the same shape
        y = tf.zeros_like(x) + y

    if mul:
        out = tf.concat([x, y, x * y], axis=-1)
        dim *= 3
    else:
        out = tf.concat([x, y], axis=-1)
        dim *= 2

    return out, dim


def generateVarDpMask(shape, keepProb):
    '''
    Generates a variational dropout mask for a given shape and a dropout
    probability value.
    '''
    randomTensor = tf.to_float(keepProb)
    randomTensor += tf.random_uniform(shape, minval=0, maxval=1)
    binaryTensor = tf.floor(randomTensor)
    mask = tf.to_float(binaryTensor)
    return mask


def applyVarDpMask(inp, mask, keepProb):
    '''
    Applies the a variational dropout over an input, given dropout mask
    and a dropout probability value.
    '''
    ret = (tf.div(inp, tf.to_float(keepProb))) * mask
    return ret


def createCell(hDim, reuse, cellType=None, act=None, projDim=None):
    '''
    Creates an RNN cell.

    Args:
        hdim: the hidden dimension of the RNN cell.

        reuse: whether the cell should reuse parameters or create new ones.

        cellType: the cell type
        RNN, GRU, LSTM, MiGRU, MiLSTM, ProjLSTM

        act: the cell activation
        NON, TANH, SIGMOID, RELU, ELU

        projDim: if ProjLSTM, the dimension for the states projection

    Returns the cell.
    '''
    if cellType is None:
        cellType = cfg.ENC_TYPE

    activation = activations.get(act, None)

    if cellType == "ProjLSTM":
        cell = tf.nn.rnn_cell.LSTMCell
        cell = cell(hDim, num_proj=projDim, reuse=reuse, activation=activation)
        return cell

    cells = {
        "RNN": tf.nn.rnn_cell.BasicRNNCell,
        "GRU": tf.nn.rnn_cell.GRUCell,
        "LSTM": lambda *args, **kwargs:
            tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', *args, **kwargs),
    }
    cell = cells[cellType](hDim, reuse=reuse, activation=activation)

    return cell


def fwRNNLayer(inSeq, seqL, hDim, cellType=None, dropout=1.0, varDp=None,
               name="", reuse=None):  # proj = None
    '''
    Runs an forward RNN layer.

    Args:
        inSeq: the input sequence to run the RNN over.
        [batchSize, sequenceLength, inDim]

        seqL: the sequence matching lengths.
        [batchSize, 1]

        hDim: hidden dimension of the RNN.

        cellType: the cell type
        RNN, GRU, LSTM, MiGRU, MiLSTM, ProjLSTM

        dropout: value for dropout over input sequence

        varDp: if not None, state and input variational dropouts to apply.
        dimension of input has to be supported (inputSize).

    Returns the outputs sequence and final RNN state.
    '''
    # varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
    with tf.variable_scope("rnnLayer" + name, reuse=reuse):
        batchSize = tf.shape(inSeq)[0]

        # passing reuse isn't mandatory
        cell = createCell(hDim, reuse, cellType)

        if varDp is not None:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                state_keep_prob=varDp["stateDp"],
                input_keep_prob=varDp["inputDp"],
                variational_recurrent=True, input_size=varDp["inputSize"],
                dtype=tf.float32)
        else:
            inSeq = tf.nn.dropout(inSeq, dropout)

        initialState = cell.zero_state(batchSize, tf.float32)

        outSeq, lastState = tf.nn.dynamic_rnn(
            cell, inSeq, sequence_length=seqL, initial_state=initialState,
            swap_memory=True)

        if isinstance(lastState, tf.nn.rnn_cell.LSTMStateTuple):
            lastState = lastState.h

    return outSeq, lastState


def biRNNLayer(inSeq, seqL, hDim, cellType=None, dropout=1.0, varDp=None,
               name="", reuse=None):
    '''
    Runs an bidirectional RNN layer.

    Args:
        inSeq: the input sequence to run the RNN over.
        [batchSize, sequenceLength, inDim]

        seqL: the sequence matching lengths.
        [batchSize, 1]

        hDim: hidden dimension of the RNN.

        cellType: the cell type
        RNN, GRU, LSTM, MiGRU, MiLSTM

        dropout: value for dropout over input sequence

        varDp: if not None, state and input variational dropouts to apply.
        dimension of input has to be supported (inputSize).

    Returns the outputs sequence and final RNN state.
    '''
    # varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
    with tf.variable_scope("birnnLayer" + name, reuse=reuse):
        batchSize = tf.shape(inSeq)[0]

        with tf.variable_scope("fw"):
            cellFw = createCell(hDim, reuse, cellType)
        with tf.variable_scope("bw"):
            cellBw = createCell(hDim, reuse, cellType)

        if varDp is not None:
            cellFw = tf.contrib.rnn.DropoutWrapper(
                cellFw,
                state_keep_prob=varDp["stateDp"],
                input_keep_prob=varDp["inputDp"],
                variational_recurrent=True, input_size=varDp["inputSize"],
                dtype=tf.float32)
            cellBw = tf.contrib.rnn.DropoutWrapper(
                cellBw,
                state_keep_prob=varDp["stateDp"],
                input_keep_prob=varDp["inputDp"],
                variational_recurrent=True, input_size=varDp["inputSize"],
                dtype=tf.float32)
        else:
            inSeq = tf.nn.dropout(inSeq, dropout)

        initialStateFw = cellFw.zero_state(batchSize, tf.float32)
        initialStateBw = cellBw.zero_state(batchSize, tf.float32)

        (outSeqFw, outSeqBw), (lastStateFw, lastStateBw) = \
            tf.nn.bidirectional_dynamic_rnn(
                cellFw, cellBw, inSeq, sequence_length=seqL,
                initial_state_fw=initialStateFw,
                initial_state_bw=initialStateBw,
                swap_memory=True)

        if isinstance(lastStateFw, tf.nn.rnn_cell.LSTMStateTuple):
            lastStateFw = lastStateFw.h  # take c?
            lastStateBw = lastStateBw.h

        outSeq = tf.concat([outSeqFw, outSeqBw], axis=-1)
        lastState = tf.concat([lastStateFw, lastStateBw], axis=-1)

    return outSeq, lastState


def RNNLayer(inSeq, seqL, hDim, bi=None, cellType=None, dropout=1.0,
             varDp=None, name="", reuse=None):
    '''
    Runs an RNN layer by calling biRNN or fwRNN.

    Args:
        inSeq: the input sequence to run the RNN over.
        [batchSize, sequenceLength, inDim]

        seqL: the sequence matching lengths.
        [batchSize, 1]

        hDim: hidden dimension of the RNN.

        bi: true to run bidirectional rnn.

        cellType: the cell type
        RNN, GRU, LSTM, MiGRU, MiLSTM

        dropout: value for dropout over input sequence

        varDp: if not None, state and input variational dropouts to apply.
        dimension of input has to be supported (inputSize).

    Returns the outputs sequence and final RNN state.
    '''
    # varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
    with tf.variable_scope("rnnLayer" + name, reuse=reuse):
        if bi is None:
            bi = cfg.ENC_BI

        rnn = biRNNLayer if bi else fwRNNLayer

        if bi:
            hDim = int(hDim / 2)

    return rnn(inSeq, seqL, hDim, cellType=cellType, dropout=dropout,
               varDp=varDp)
