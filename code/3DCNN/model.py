from __future__ import division
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, Conv3D, MaxPooling3D, MaxPooling2D
from keras.layers.merge import dot, multiply
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model

import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt
import seaborn as sns
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

num_words = int(config['3DCNN']['num_words'])
num_train_negatives = int(config['3DCNN']['num_train_negatives'])
path_model_folder = config['3DCNN']['path_model_folder']
num_epochs = int(config['3DCNN']['num_epochs'])

class CNN():

    def __init__(self, num_used_hists, num_words, num_word_embedding_dims):
        self.num_used_hists = num_used_hists
        self.num_words = num_words
        self.num_word_embedding_dims = num_word_embedding_dims
        self.model = None

    def create_model(self):

        user_input0 = Input(shape=(1, self.num_used_hists, self.num_words, self.num_word_embedding_dims)) # 3d
        article_input0 = Input(shape=(1, self.num_words, self.num_word_embedding_dims)) # 2d

        user_input = Dropout(0.2)(user_input0)
        article_input = Dropout(0.2)(article_input0)

        conv3d = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', data_format='channels_first')(user_input)
        maxpooling3d = MaxPooling3D(pool_size=(2, 2, 2))(conv3d)
        conv3d2 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(maxpooling3d)
        maxpooling3d2 = MaxPooling3D(pool_size=(2, 2, 2))(conv3d2)
        flatten3d = Flatten()(maxpooling3d2)

        conv2d = Conv2D(64, kernel_size=(3, 3), activation='relu', data_format='channels_first')(article_input)
        maxpooling2d = MaxPooling2D(pool_size=(2, 2))(conv2d)
        conv2d2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(maxpooling2d)
        maxpooling2d2 = MaxPooling2D(pool_size=(2, 2))(conv2d2)
        flatten2d = Flatten()(maxpooling2d2)

        element_wise_product0 = multiply([flatten3d, flatten2d])
        element_wise_product = Dropout(0.5)(element_wise_product0)
        
        fc = Dense(128, activation='relu')(element_wise_product)
        output = Dense(1, activation='sigmoid')(fc)

        self.model0 = Model(inputs=[user_input0, article_input0], outputs = output)
        self.model = multi_gpu_model(self.model0, gpus=4)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])



    def plot_acc_loss(self, history):
        epochs = [i + 1 for i in range(len(history.history['accuracy']))]
        plt.figure()
        plt.plot(epochs, history.history['accuracy'],'b',label='Training acc')
        plt.plot(epochs, history.history['val_accuracy'],'r',label='Validation acc')
        plt.title('Traing and Validation accuracy')
        plt.legend()
        plt.ylim(0.6, 1)
        plt.xlim(0, num_epochs)
        plt.xlabel('epochs')
        plt.savefig('model_acc.png')

        plt.figure()
        plt.plot(epochs, history.history['loss'],'b',label='Training loss')
        plt.plot(epochs, history.history['val_loss'],'r',label='Validation loss')
        plt.title('Traing and Validation loss')
        plt.legend()
        plt.ylim(0.1, 0.6)
        plt.xlim(0, num_epochs)
        plt.xlabel('epochs')
        plt.savefig('model_loss.png')

    def fit_model(self, inputs, outputs, batch_size, num_epochs):
        filepath="%s/3dcnn_word_%d_neg_%d_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.model"%(path_model_folder, num_words, num_train_negatives)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = self.model.fit(inputs, outputs, validation_split = 0.2, batch_size = batch_size, epochs = num_epochs, callbacks = callbacks_list, verbose=1)
        self.plot_acc_loss(history)


    def get_model_summary(self):
        print(self.model.summary())
