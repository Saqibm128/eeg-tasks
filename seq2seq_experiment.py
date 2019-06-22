from keras.models import Model
from keras.layers import Input, LSTM, Dense
import data_reader as read
import pandas as pd
from seq2seq.models import Seq2Seq

edfRawData = read.EdfDataset("dev_test", "01_tcp_ar", num_files=20)
data = read.EdfFFTDatasetTransformer(edfRawData, window_size=pd.Timedelta(seconds=1), precache=True)

model = Seq2Seq(batch_input_shape=(16, 7, 5), hidden_dim=10, output_length=8, output_dim=20, depth=4)
model.compile(loss='mse', optimizer='rmsprop')
