from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class EarlyStoppingByLoss(keras.callbacks.Callback):
    def __init__(self, monitor, value):
        super().__init__()
        self.monitor = monitor
        self.value = value

    #callback會呼叫此function
    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get(self.monitor)
        #判斷是否有達到停止標準 如果有 就停止訓練
        if current_loss is not None and current_loss < self.value:
            print(f"Epoch {epoch + 1}: Stopping training as loss is less than {self.value}")
            self.model.stop_training = True

#打開模型
with open('bpnn_model.json', 'r') as file:
    model_json = file.read()
#建立模型
trained_model = keras.models.model_from_json(model_json)
#匯入權重
trained_model.load_weights('bpnn_model_weights.h5')

fs=1/1e2
x_min=0
x_max=10
#定重新訓練的X與y值
x_relearn=np.arange(x_min , x_max , fs)
y=np.sin(2*x_relearn)
#建立停止條件
stop_condition=EarlyStoppingByLoss(monitor='loss',value=1e-4)

#retrain again the model
trained_model.compile(optimizer='adam', loss='mean_squared_error')
history = trained_model.fit(x_relearn, y, epochs=100,callbacks=[stop_condition]) 

# 給新的X進行訓練
x_new = np.arange(x_min,x_max,fs)  
y_pred = trained_model.predict(x_new)

# 繪出cost function
plt.figure(1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.figure(2)#畫出結果
plt.plot(x_relearn, y, label='Original Data')
plt.plot(x_new, y_pred, label='Model Prediction', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()