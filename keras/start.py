import numpy as np

#1,2
#3,4
a = np.array([[1,2],[3,4]])
sum0 = np.sum(a, axis=0) #axis=0 X轴
sum1 = np.sum(a, axis=1) #axis=1 Y轴
sum2 = np.sum(a)
# sum3 = np.sum(a, axis=2)

print(sum0)
print(sum1)
print(sum2)
# print(sum3)

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.layers import merge
from keras.utils import to_categorical

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(labels, num_classes=10)
# one_hot_labels=labels
print(one_hot_labels)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)