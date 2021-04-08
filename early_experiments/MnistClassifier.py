import tensorflow as tf 
import tensorflow.keras.layers as layers


class MnistClassifier():
    def __init__(self):

        self.model = self.build_model()


    def build_model(self):
        model = tf.keras.Sequential()
        
        model.add(layers.Conv2D(28, (5,5), activation=tf.keras.activations.relu , input_shape=(28,28,1)))
        print(model.output_shape)
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        print(model.output_shape)

        model.add(layers.Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(layers.Dropout(0.2))
        print(model.output_shape)

        model.add(layers.Dense(10,activation=tf.keras.activations.softmax))
        print(model.output_shape)

        return model


    def train(self, x_train, y_train, epochs=10):
        self.model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=epochs)


    def evaluate(self, x_test, y_test):
        self.model.evaluate(x_test, y_test)

        
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshaping the array to 4-dims so that it can work with the Keras API
    print(x_train.shape[0])
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    #input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    model = MnistClassifier()
    model.train(x_train, y_train, epochs=3)
    model.evaluate(x_test, y_test)