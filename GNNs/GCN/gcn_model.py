import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def gcn(A_hat, X, num_classes=7, dropout=0.5):
    """
    Two layer GCN model.
    """

    H = gcn_layer(A_hat, X, out_features=16,
                  name='gcn_layer_0', dropout=dropout)
    H = gcn_layer(A_hat, H, out_features=num_classes,
                  name='gcn_layer_1', dropout=dropout, activation=F.softmax)

    return H


def gcn_layer(A_hat, X, out_features, name, dropout=0.5, activation=F.relu):
    '''
    GCN layer

    Parameters
    ----------
    A_hat: nnabla.Variable
      Normalized graph Llaplacian
    X: nnabla.Variable
      Feature matrix
    out_features: int
      Number of dimensions of output
    name: str
      Name of parameter scope
    dropout: float
      Parameter of dropout. If 0, not to use dropout
    activaton: nnabla.functons
      Activation function

    Returns
    -------
    H: nnabla.Variable
      Output of GCN layer
    '''

    with nn.parameter_scope(name):
        if dropout > 0:
            X = F.dropout(X, dropout)

        H = PF.affine(X, (out_features, ), with_bias=False)
        H = F.dot(A_hat, H)

        if activation is not None:
            H = activation(H)

    return H
