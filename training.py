from lib.file_operate import read_np_lists

import numpy as np
import matplotlib.pyplot as plt

data_x = read_np_lists('HOG_data/PHOG.txt')
amount = len(data_x) 
print("containing "+str(amount)+" positive samples")
 
#train_x = data_x[0:int(amount*0.8)]
train_x = data_x[20:]
val_x = data_x[int(amount*0.8):amount]

train_y = np.array([1]*len(train_x))
val_y = np.array([1]*len(val_x))

data_nx = read_np_lists('HOG_data/NHOG.txt')
namount = len(data_nx) 
print("containing "+str(namount)+" negitive samples")
 
train_nx = data_nx[0:int(namount*0.5)]
val_nx = data_nx[int(namount*0.5):namount]

train_ny = np.array([0]*len(train_nx))
val_ny = np.array([0]*len(val_nx))

train_x = np.concatenate((train_x,train_nx))
val_x = np.concatenate((val_x,val_nx))

train_y = np.concatenate((train_y,train_ny))
val_y = np.concatenate((val_y,val_ny))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


batch_size = 200


model = Sequential()
model.add(Dense(50, activation='tanh',input_shape=(5832,)))
model.add(Dropout(0.2))
model.add(Dense(20, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='tanh'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 50
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=EPOCHS, batch_size=batch_size)
loss, acc = model.evaluate(val_x, val_y, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

if(input("是否保存模型： y/n?")=="y"):
    model.save("trained_models/nnet/model.h5")
