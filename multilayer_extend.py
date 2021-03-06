import numpy as np
from collections import OrderedDict
from layers import *
from numerical_gradient import numerical_gradient
#from my_module.layers import *
#from my_module.numerical_gradient import *

class MultiLayerNetExtend:

    def __init__( self, input_size, output_size, hidden_size_list, activation, loss, weight_init_std, use_batchnorm, threshold = None, outlayer_identity = False ):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len( hidden_size_list )
        self.output_size = output_size
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.threshold = threshold
        self.outlayer_identity = outlayer_identity
        ##Initialize weights
        self.__init_weight( weight_init_std )
        ##Make layers
        self.layers = OrderedDict()
        for idx in range( 1, self.hidden_layer_num+2 ):
            self.layers[ "Affine" + str( idx ) ] = Affine( self.params[ "W" + str( idx ) ], self.params[ "b" + str( idx ) ] )
            if use_batchnorm and idx != self.hidden_layer_num+1:
            #if use_batchnorm:
                self.layers[ "BatchNorm" + str( idx ) ] =\
                    BatchNormalization( self.params[ "gamma" + str( idx ) ], self.params[ "beta" + str( idx ) ], 1e-7 )
            ##Determine Activation Function
            if idx == self.hidden_layer_num+1 and self.outlayer_identity:
                self.layers["Activation"+str(idx)] = Identity()
                continue
            if activation is "oddrelu":
                self.layers["Activation" + str(idx)] = OddRelu(self.threshold)
            if activation is "relu":
                self.layers[ "Activation" + str( idx ) ] = Relu()
            if activation is "sigmoid":
                self.layers[ "Activation" + str( idx ) ] = Sigmoid()
            if activation is "tanh":
                self.layers[ "Activation" + str( idx ) ] = Tanh()
            if activation is "softsign":
                self.layers["Activation"+str(idx)] = Softsign()
            if activation is "tanhexp":
                self.layers["Activation"+str(idx)] = TanhExp()
            if activation is "origin1":
                self.layers["Activation"+str(idx)] = Origin1()
            if activation is "origin2":
                self.layers["Activation"+str(idx)] = Origin2()
            if activation is "identity":
                self.layers["Activation"+str(idx)] = Identity()

        ##Determine Loss Function
        if loss is "OriginalLoss":
            self.lastlayer = OriginalLoss()
        if loss is "MSE_RE2":
            self.lastlayer = MSE_RelativeError2()
        if loss is "MAE_sqrt":
            self.lastlayer = MAE_sqrt()
        if loss is "MAE":
            self.lastlayer = MAE()
        if loss is "MAE_log":
            self.lastlayer = MAE_log()
        if loss is "MSE_log":
            self.lastlayer = MSE_log()
        if loss is "MSE_RE":
            self.lastlayer = MSE_RelativeError()
        if loss is "MSE_AE":
            self.lastlayer = MSE_AbsoluteError()


    def __init_weight( self, weight_init_std ):
        """
        Initial weight setting
        """
        all_size_list = [ self.input_size ] + self.hidden_size_list + [ self.output_size ]
        for idx in range( 1, len( all_size_list ) ):
            #He's initial value
            if str( weight_init_std ).lower() in ( "relu", "he" ):
                scale = np.sqrt( 2.0 / all_size_list[ idx - 1 ] )
            #Xavier's initial value
            if str( weight_init_std ).lower() in ( "sigmoid", "xavier", "tanh" ):
                scale = np.sqrt( 1.0 / all_size_list[ idx - 1 ] )
            if idx == len(all_size_list)-1 and self.outlayer_identity:
                scale = 1.0
            self.params[ "W" + str( idx ) ] = scale * np.random.randn( all_size_list[ idx - 1 ], all_size_list[ idx ] )
            self.params[ "b" + str( idx ) ] = np.zeros( all_size_list[ idx ] )
            if self.use_batchnorm and idx != len(all_size_list)-1:
            #if self.use_batchnorm:
                self.params[ "gamma" + str( idx ) ] = np.ones( all_size_list[ idx ] )
                self.params[ "beta" + str( idx ) ] = np.zeros( all_size_list[ idx ] )


    def predict( self, x, is_training ):
        for layer in self.layers.values():
            x = layer.forward( x, is_training )
        return x


    def loss( self, x, t, is_training ):
        y = self.predict( x, is_training )
        return self.lastlayer.forward( y, t )


    def accuracy( self, x, t, is_training ):
        y = self.predict( x, is_training )
        accuracy = np.abs((y-t)/t)
        accuracy = np.sum(accuracy)/y.size
        return accuracy


    def numerical_gradient( self, x, t ):
        loss_W = lambda W: self.loss( x, t , is_training = True )
        grads = {}
        for idx in range( 1, self.hidden_layer_num + 2 ):
            grads[ "W" + str( idx ) ] =\
                            numerical_gradient( loss_W, self.params[ "W" + str( idx ) ] )
            grads[ "b" + str( idx ) ] =\
                            numerical_gradient( loss_W, self.params[ "b" + str( idx ) ] )
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
            #if self.use_batchnorm:
                grads[ "gamma" + str( idx ) ] =\
                            numerical_gradient( loss_W, self.params[ "gamma" + str( idx ) ] )
                grads[ "beta" + str( idx ) ] =\
                            numerical_gradient( loss_W, self.params[ "beta" + str( idx ) ] )
        return grads


    def gradient( self, x, t, is_training ):
        ##Forward
        self.loss( x, t, is_training )

        ##Backward
        dout = 1
        dout = self.lastlayer.backward( dout )

        layers = list( self.layers.values() )
        layers.reverse()
        for layer in layers:
            dout = layer.backward( dout )

        ##Update gradients
        grads = {}
        for idx in range( 1, self.hidden_layer_num + 2 ):
            grads[ "W" + str( idx ) ] = self.layers[ "Affine" + str( idx ) ].dW
            grads[ "b" + str( idx ) ] = self.layers[ "Affine" + str( idx ) ]. db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
            #if self.use_batchnorm:
                grads[ "gamma" + str( idx ) ] = self.layers[ "BatchNorm" + str( idx ) ].dgamma
                grads[ "beta" + str( idx ) ] = self.layers[ "BatchNorm" + str( idx ) ].dbeta

        return grads
