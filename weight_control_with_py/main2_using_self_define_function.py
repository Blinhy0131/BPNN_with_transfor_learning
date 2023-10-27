from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class CallBackNeed2Do(keras.callbacks.Callback):
    def __init__(self, monitor, value, layer1, layer2):
        super().__init__()
        self.monitor = monitor
        self.value = value
        self.layer1 = layer1
        self.layer2 = layer2
        
    def change_weight(self):
        #讀取模型權重
        weight_ram=model.layers[1].get_weights()[0]
        #替換成所要的數值
        weight_ram[50:100,:]=self.layer1[50:100,:]
        #副寫模型權重
        model.layers[1].set_weights(weight_ram)
        
        weight_ram=model.layers[2].get_weights()[0]
        weight_ram[50:100,:]=self.layer2[50:100,:]
        model.layers[2].set_weights(weight_ram)
        
        
    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get(self.monitor)
        #判斷是否有達到停止標準 如果有 就停止訓練
        if current_loss is not None and current_loss < self.value:
            print(f"Epoch {epoch + 1}: Stopping training as loss is less than {self.value}")
            self.model.stop_training = True
        #呼叫chenge_weight function
        self.change_weight

fs=1/1e2
x_min=0
x_max=10
x=np.arange(x_min , x_max , fs)
y=np.sin(2*x)
# 讀取模型
with open('bpnn_model.json', 'r') as file:
    model_json = file.read()
#建立模型與讀取權重
trained_model = keras.models.model_from_json(model_json)
trained_model.load_weights('bpnn_model_weights.h5')

# 建立訓練模型 模型名稱叫做model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),    # input
    keras.layers.Dense(100, activation='tanh'),  # 第一層
    keras.layers.Dense(100, activation='tanh'),   # 第二層
    keras.layers.Dense(100, activation='tanh'),   # 第二層
    keras.layers.Dense(100, activation='tanh'),   # 第二層
    keras.layers.Dense(1)             # output
])

#讀取訓練好的模型權重並且保存
layer1_weight=trained_model.layers[1].get_weights()[0]
layer2_weight=trained_model.layers[2].get_weights()[0]

adam = keras.optimizers.Adam(epsilon=1e-7)
# 建立模型 使用adam優化 使用平方差誤差函數
model.compile(optimizer=adam, loss='mean_squared_error')

# 設定callback要做的事
call_back=CallBackNeed2Do(monitor='loss',value=1e-4,layer1=layer1_weight,layer2=layer2_weight)
# 建立訓練模型 訓練epochs次D
history = model.fit(x, y, epochs=100,callbacks=[call_back])

# 給新的X進行訓練
x_new = np.arange(x_min, x_max,fs)  
y_pred = model.predict(x_new)

# 繪出cost function
plt.figure(1)
plt.plot(history.history['loss'])
plt.title('Training Loss (stop at mse<1e-4)')
plt.figure(2)#畫出結果
plt.plot(x, y, label='Original Data')
plt.plot(x_new, y_pred, label='Model Prediction', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()