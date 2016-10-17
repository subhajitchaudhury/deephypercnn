import theano.tensor as T
from lasagne import layers
from nolearn.lasagne import NeuralNet,BatchIterator
import numpy as np
import theano
import cPickle
import lasagne
from lasagne.updates import nesterov_momentum,adagrad
from nolearn.lasagne import TrainSplit
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from nolearn.lasagne import TrainSplit 
from prepData import load_matfile
import sys
sys.setrecursionlimit(1500)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        #bs = Xb.shape[0]
        #indices = np.random.choice(bs, bs / 2, replace=False)
        bs = Xb.shape[0]
        num_ch= np.int(2*bs / 3)
        indices = np.random.choice(bs,num_ch, replace=False)
        
        ind1=   indices[0:num_ch/2]
        ind2=   indices[num_ch/2:]        
                
        Xb[ind1] = Xb[ind1, :, :, ::-1]
        
        Xb[ind2] = Xb[ind2, :, ::-1,:]
        
        return Xb, yb
        
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        
#        meanDataFile='32in_32out-retrainsig25_denoise_meanData.dat'
#        fp=open(meanDataFile,'rb')
#        self.meanX,mean_y=cPickle.load(fp);
#        fp.close()
#        self.img_clean=convert_to_rgb(cv2.imread('./images/Lena512rgb.png'))
        

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()
            
        #f = open("allfonts_inter_impainter_cnn.dump", "wb")
        #cPickle.dump(nn, f, -1)
        #f.close()
        
def network():
    
    cr=30;
    net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', ConvLayer),
        ('conv2', ConvLayer),
        ('conv3', ConvLayer),
        ('conv4', ConvLayer),
        ('hidden1', layers.DenseLayer),
        ('hidden2', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
        
    input_shape=(None, cr, 5, 5),
    conv1_num_filters=3*cr, conv1_filter_size=(3, 3), conv1_pad=1,
    conv2_num_filters=6*cr, conv2_filter_size=(3, 3), conv2_pad=1,
    conv3_num_filters=6*cr, conv3_filter_size=(3, 3),
    conv4_num_filters=9*cr, conv4_filter_size=(3, 3),
    hidden1_num_units=6*cr,
    hidden2_num_units=3*cr,
    output_num_units=16, output_nonlinearity=softmax,
    update=adagrad,
    update_learning_rate=theano.shared(np.float32(0.005)),
    #update_momentum=theano.shared(np.float32(0.9)),

    regression=False,
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.005, stop=0.005),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],
    train_split=TrainSplit(eval_size=0.1), 
    batch_iterator_train=FlipBatchIterator(batch_size=512),
    max_epochs=2000,
    verbose=2,
    )
    return net
    
    
def train_network(X,y,prefix):
    net=network()
    net.fit(X, y)
    # Save models.
    f = open(prefix+"_indian_pines.dump", "wb")
    cPickle.dump(net, f, -1)
    f.close()
    
    return net
    
    
if __name__=='__main__':
    X,y=load_matfile(filename='./data/indian_pines_data_pca.mat')
    #X,y=load_matfile(filename='./data/indian_pines_data.mat')
    X=X.transpose(3,2,1,0)
    y=np.squeeze(y)-1
    
    num_pix=len(X)
    train_size=np.round(0.8*num_pix)
    
    X_train=X[:train_size]
    y_train=y[:train_size]
    
    X_test=X[train_size:]
    y_test=y[train_size:]
    
        
    
    std=np.std(X_train)
    mean=np.mean(X_train)
    
    X_train=(X_train-mean)/std    
    X_test=(X_test-mean)/std    
    
    
    prefix='test2'
    net=train_network(X_train,y_train,prefix)
    
    y_pred=net.predict(X_test)
    acc=np.sum(y_pred==y_test)*1.0/len(y_test)
    
    print 'Test Accuracy= ',acc*100,' %'