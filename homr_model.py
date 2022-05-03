import argparse
import tensorflow as tf
from homr_dataset import HOMRDataset


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, stride: int) -> None:
        super().__init__()

        self._filters, self._kernel, self._stride = filters, kernel_size, stride

        self._conv = tf.keras.layers.Conv1D(filters, kernel_size, stride, padding="same",
                                            data_format='channels_first')
        self._bn = tf.keras.layers.BatchNormalization()
        self._relu = tf.keras.layers.ReLU()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = self._relu(self._bn(self._conv(inputs)))
        return output

    def get_config(self) -> dict[str, int]:
        return {'filters': self._filters,
                'kernel_size': self._kernel,
                'stride': self._stride}


class RNNBlock(tf.keras.layers.Layer):
    def __init__(self, units: int, dropout_rate: float) -> None:
        super().__init__()

        self._units, self._dropout_rate = units, dropout_rate

        self._rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units, return_sequences=True),
                                                  merge_mode='sum')
        self._dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = self._dropout(self._rnn(inputs))
        return output

    def get_config(self) -> dict[str, int]:
        return {'units': self._units,
                'dropout': self._dropout_rate}


class FFNClassifier(tf.keras.layers.Layer):
    def __init__(self, units: int, num_classes: int, l2_rate: float, dropout_rate: float) -> None:
        super().__init__()

        self._units, self._classes, self._l2, self._dropout = units, num_classes, l2_rate, dropout_rate

        self._fc_layer = tf.keras.layers.Dense(units, activation=tf.nn.relu,
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate))
        self._logits = tf.keras.layers.Dense(num_classes + 1)
        self._dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = self._logits(self._dropout(self._fc_layer(inputs)))
        return output

    def get_config(self) -> dict[str, int]:
        return {'units': self._units,
                'classes': self._classes,
                'dropout': self._dropout,
                'l2': self._l2}


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Define computational graph of out model - how data flows through layers

        inputs = tf.keras.layers.Input(shape=[args.target_img_H, args.target_img_W, 1], dtype=tf.float32)
        hidden = tf.reshape(inputs, tf.shape(inputs)[:-1])

        # Stacking convolutional layers
        for _ in range(args.conv_layers):
            hidden = ConvBlock(filters=args.conv_dim, kernel_size=5, stride=2)(hidden)
        hidden = tf.transpose(hidden, [0, 2, 1])

        # Stacking recurrent layers
        for _ in range(args.rnn_layers):
            hidden = RNNBlock(units=args.rnn_cell_dim, dropout_rate=args.dropout_rate)(hidden)

        # Adding final fully connected layer for temporal classification
        logits = FFNClassifier(units=args.rnn_cell_dim, num_classes=len(HOMRDataset.MARKS), l2_rate=args.l2,
                               dropout_rate=args.dropout_rate)(hidden)

        super().__init__(inputs=inputs, outputs=logits)

        # We compile the model with the CTC loss and EditDistance metric.
        cosine_decay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=args.learning_rate,
                                                                 decay_steps=tf.math.ceil(
                                                                     HOMRDataset.train_n / args.batch_size * args.epochs))

        self.compile(optimizer=tf.optimizers.Adam(learning_rate=cosine_decay, clipnorm=0.01),
                     loss=self.ctc_loss,
                     metrics=[HOMRDataset.EditDistanceMetric()])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        self._ctc_beam = args.ctc_beam

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.Tensor) -> tf.Tensor:
        """Computes Connectionist Temporal Classification (CTC) loss given target labels and model output."""

        batch_size, width = tf.shape(logits)[0], tf.shape(logits)[1]

        batch_loss = tf.nn.ctc_loss(labels=tf.cast(tf.sparse.from_dense(gold_labels.to_tensor()), tf.int32),
                                    logits=logits, label_length=None,
                                    logit_length=tf.cast(tf.fill([batch_size], width), tf.int32),
                                    logits_time_major=False, blank_index=len(HOMRDataset.MARKS))

        return tf.math.reduce_mean(batch_loss)

    def ctc_decode(self, logits: tf.Tensor) -> tf.RaggedTensor:
        """Transforms model output(logits) to labels."""

        batch_size, width = tf.shape(logits)[0], tf.shape(logits)[1]
        if self._ctc_beam > 1:
            predictions = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, [1, 0, 2]),
                                                        sequence_length=tf.cast(tf.fill([batch_size], width), tf.int32),
                                                        beam_width=self._ctc_beam)[0][0]
        else:
            predictions = tf.nn.ctc_greedy_decoder(tf.transpose(logits, [1, 0, 2]),
                                                   sequence_length=tf.cast(tf.fill([batch_size], width), tf.int32))[0][0]

        length = tf.unique_with_counts(predictions.indices[:, 0])[-1]
        predictions = tf.RaggedTensor.from_tensor(tf.sparse.to_dense(predictions), lengths=length)
        return predictions

    # We override the `train_step` method, because we do not want to
    # evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)
