import os
import sys
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from keras import Sequential
import function_sandbox as fs
import numpy as np
import pandas as pd



train_dir_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Data\Dogs_Cats_Update\Partition_Data\train')
val_dir_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Data\Dogs_Cats_Update\Partition_Data\validate')
test_dir_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Data\Dogs_Cats_Update\Partition_Data\Test')


vgg_16_weights_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\vgg_16\weights\weights')
vgg_16_results_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\vgg_16\results')

vgg_19_weights_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\vgg_19\weights\weights')
vgg_19_results_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\vgg_19\results')

resnet50_weights_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\resnet50\weights\weights')
resnet50_results_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\resnet50\results')

inceptionV3_weights_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\inception_v3\weights\weights')
inceptionV3_results_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\inception_v3\results')

resnetV2_weights_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\resnet_v2\weights\weights')
resnetV2_results_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\resnet_v2\results')

mobile_net_weights_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\mobilenet\weights\weights')
mobile_net_results_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\mobilenet\results')

xception_weights_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\xception\weights\weights')
xception_results_path = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Charlies_Final_Project\model_data\xception\results')


class Models:

    def __init__(self):
        self._model = Sequential()
        self._history = ""
        self._path_to_weights = ""
        self._accuracy = 0
        self._model_name = " "
        self._path_to_results = ""

    def summarize_results(self):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Binary Loss Function')
        pyplot.plot(self._history.history['loss'], color='purple', label='train')
        pyplot.plot(self._history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(214)
        pyplot.title('Classification Accuracy')
        pyplot.plot(self._history.history['accuracy'], color='green', label='train')
        pyplot.plot(self._history.history['val_accuracy'], color='blue', label='test')

        # save plot to file
        pyplot.savefig(os.path.join(self._path_to_results, fs.get_time_stamp()))
        pyplot.close()

    def unlock_layers_weights(self):
        # Charlies favorite program
        for layer in self._model.layers:
            layer.trainable = True
        opt = SGD(lr=0.0005, momentum=0.9)
        self._model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    def lock_layers_weights(self):
        # Charlies least favorite program
        for layer in self._model.layers:
            layer.trainable = False

    def train_model(self):
        # create data generator
        datagen = ImageDataGenerator(featurewise_center=True)
        # specify imagenet mean values for centering
        datagen.mean = [123.68, 116.779, 103.939]
        # prepare iterator
        train_gen = datagen.flow_from_directory(train_dir_path, class_mode='binary', batch_size=24, target_size=(224, 224))
        test_gen = datagen.flow_from_directory(val_dir_path, class_mode='binary', batch_size=24, target_size=(224, 224))

        self._history = self._model.fit_generator(generator=train_gen, validation_data=test_gen, epochs=1, verbose=1)

    def test_model(self):
        # create data generator
        datagen = ImageDataGenerator(featurewise_center=True)
        # specify imagenet mean values for centering
        datagen.mean = [123.68, 116.779, 103.939]
        test_gen = datagen.flow_from_directory(test_dir_path, class_mode='binary', batch_size=24, target_size=(224, 224))
        # evaluate model
        _, acc = self._model.evaluate_generator(test_gen, verbose=0)
        self._accuracy = acc
        print('Accuracy on Test Set> %.3f' % (acc * 100.0))

    def deploy_model(self):
        # create data generator
        datagen = ImageDataGenerator(featurewise_center=True)
        # specify imagenet mean values for centering
        datagen.mean = [123.68, 116.779, 103.939]

        test_gen = datagen.flow_from_directory(test_dir_path, class_mode='binary',
                                               target_size=(224, 224), shuffle=False)

        predictions = self._model.predict(test_gen)
        classes = fs.get_sub_dirs(test_dir_path)
        class_list = []
        for individual_class in classes:
            class_list.append(str(os.path.basename(os.path.normpath(individual_class))))
        classifications = [['Image # ', 'Actual Class', self.get_model_name() + 'Predicted Class']]
        for img_dir in classes:
            class_name = str(os.path.basename(os.path.normpath(img_dir)))
            for img in fs.get_sub_dirs(img_dir):
                image_name = str(os.path.basename(os.path.normpath(img)))
                if predictions.item(0) > .5:
                    pre_class = class_list[1]
                else:
                    pre_class = class_list[0]
                predictions = np.delete(predictions, 0)
                classifications.append([image_name, class_name, pre_class])
        return classifications

    # print(predictions)

    def save_model_weights(self):
        try:
            self._model.save_weights(self._path_to_weights, overwrite=True)
            print(self._model_name + ' weights saved to: ' + str(self._path_to_weights))
        except IOError:
            print("ERROR: MODEL WEIGHTS UNSAVED")

    def get_model_name(self):
        return self._model_name

    def print_model_summary(self):
        try:
            self._model.summary()
        except:
            print('ERROR:This model has not yet been built.')

    def get_accuracy(self):
        return self._accuracy * 100

    def get_path_to_weights(self):
        return self._path_to_weights

    def load_model_weights(self):
        try:
            self._model.load_weights(self._path_to_weights)
            print('PREVIOUS MODEL WEIGHTS LOADED: ' + str(self._path_to_weights))
        except IOError:
            print("ERROR: FAILURE TO LOAD MODEL WEIGHTS")


class VGG16BasedModel(Models):
    def __init__(self):
        Models.__init__(self)
        self._model_name = "VGG16"
        self._path_to_weights = vgg_16_weights_path
        self._path_to_results = vgg_16_results_path
        self._model = self.build_model()
        self.load_model_weights()

    @staticmethod
    def build_model():
        # Load model
        model = VGG16(include_top=False, input_shape=(224, 224, 3))

        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False

        # add custom Layers layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)

        # define new model
        model = Model(inputs=model.inputs, outputs=output)

        # compile
        opt = SGD(lr=0.0005, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model


class VGG19BasedModel(Models):
    def __init__(self):
        Models.__init__(self)
        self._model_name = 'VGG19'
        self._path_to_weights = vgg_19_weights_path
        self._path_to_results = vgg_19_results_path
        self._model = self.build_model()
        self.load_model_weights()

    @staticmethod
    def build_model():
        # load model
        model = VGG19(include_top=False, input_shape=(224, 224, 3))

        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False

        # add custom Layers layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)

        # define new model
        model = Model(inputs=model.inputs, outputs=output)

        # compile
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model


class ResNet50BasedModel(Models):
    def __init__(self):
        Models.__init__(self)
        self._model_name = 'RESNET50'
        self._path_to_weights = resnet50_weights_path
        self._path_to_results = resnet50_results_path
        self._model = self.build_model()
        self.load_model_weights()

    @staticmethod
    def build_model():
        # load model
        model = ResNet50(include_top=False, input_shape=(224, 224, 3))

        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False

        # add custom Layers layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)

        # define new model
        model = Model(inputs=model.inputs, outputs=output)

        # compile
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model


class InceptionV3BasedModel(Models):
    def __init__(self):
        Models.__init__(self)
        self._model_name = 'INCEPTIONV3'
        self._path_to_weights = inceptionV3_weights_path
        self._path_to_results = inceptionV3_results_path
        self._model = self.build_model()
        self.load_model_weights()

    @staticmethod
    def build_model():
        # load model
        model = InceptionV3(include_top=False, input_shape=(224, 224, 3))

        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False

        # add custom Layers layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)

        # define new model
        model = Model(inputs=model.inputs, outputs=output)

        # compile
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model


class ResNetV2BasedModel(Models):
    def __init__(self):
        Models.__init__(self)
        self._model_name = 'RESNETV2'
        self._path_to_weights = resnetV2_weights_path
        self._path_to_results = resnetV2_results_path
        self._model = self.build_model()
        self.load_model_weights()

    @staticmethod
    def build_model():
        # load model
        model = InceptionResNetV2(include_top=False, input_shape=(224, 224, 3))

        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False

        # add custom Layers layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)

        # define new model
        model = Model(inputs=model.inputs, outputs=output)

        # compile
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model


class MobileNetBasedModel(Models):
    def __init__(self):
        Models.__init__(self)
        self._model_name = 'MOBILENET'
        self._path_to_weights = mobile_net_weights_path
        self._path_to_results = mobile_net_results_path
        self._model = self.build_model()
        self.load_model_weights()

    @staticmethod
    def build_model():
        # load model
        model = MobileNet(include_top=False, input_shape=(224, 224, 3))

        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False

        # add custom Layers layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)

        # define new model
        model = Model(inputs=model.inputs, outputs=output)

        # compile
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model


class XceptionBasedModel(Models):
    def __init__(self):
        Models.__init__(self)
        self._model_name = 'XCEPTION'
        self._path_to_weights = xception_weights_path
        self._path_to_results = xception_results_path
        self._model = self.build_model()
        self.load_model_weights()

    @staticmethod
    def build_model():
        # Load model
        model = Xception(include_top=False, input_shape=(224, 224, 3))

        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False

        # add custom Layers layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)

        # define new model
        model = Model(inputs=model.inputs, outputs=output)

        # compile
        opt = SGD(lr=0.0005, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model

