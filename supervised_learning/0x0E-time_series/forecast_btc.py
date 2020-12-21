#!/usr/bin/env python3
"""Script to creates, trains, and validates a
keras model for the forecasting of BTC"""
import numpy as np
import tensorflow as tf


class WindowGenerator():
    """Window Generator Class"""
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """Class constructor"""
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        a = self.input_slice
        self.input_indices = np.arange(self.total_window_size)[a]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        b = self.labels_slice
        self.label_indices = np.arange(self.total_window_size)[b]

    def split_window(self, features):
        """Split window method"""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        c = self.label_columns
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in c],
                axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """Plot method"""
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel('{} [normed]'.format(plot_col))
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,
                                                                 None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :,
                            label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        """Make dataset method"""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        """Train property"""
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """Val Property"""
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """Test property"""
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """example property"""
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


class Baseline(tf.keras.Model):
    """Baseline Class"""
    def __init__(self, label_index=None):
        """constructor method"""
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        """call method"""
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

    def build_model():
        """Build model method"""
        mol = tf.keras.layers.LSTM(24, input_shape=[24, 7],
                                   return_sequences=True)
        mol1 = tf.keras.layers.Dense(units=1)
        lstm_model = tf.keras.models.Sequential([mol, mol1])
        lstm_model.summary()
        return lstm_model

    def compile_and_fit(model, window, patience=2, epochs=500):
        """compile and fit method"""
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])
        history = model.fit(window.train, epochs=epochs,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        print(model.summary())
        return history

    def forecasting(train, validation, test):
        """forecasting method"""
        window = WindowGenerator(input_width=24, label_width=24, shift=1,
                                 train_df=train, val_df=validation,
                                 test_df=test,
                                 label_columns=['Close'])
        column_indices = window.column_indices
        print(window)
        val_performance = {}
        performance = {}
        baseline = Baseline(label_index=column_indices['Close'])
        baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.MeanAbsoluteError()])
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(units=1)])
        history = Baseline.compile_and_fit(lstm_model, window)
        val_performance['LSTM'] = lstm_model.evaluate(window.val)
        performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)
        window.plot(lstm_model)
