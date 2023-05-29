import tensorflow as tf
import logging
from keras.callbacks import ModelCheckpoint
from ..utils.data import strip_standardize_py
import numpy as np
from tqdm import tqdm


class StabilityCallback(tf.keras.callbacks.Callback):
    def __init__(self, clip_norm=1.0, clip_value=1e-6):
        super(StabilityCallback, self).__init__()
        self.clip_norm = clip_norm
        self.clip_value = clip_value

    def on_train_batch_begin(self, batch, logs=None):
        # Clip the gradients by norm
        if self.clip_norm is not None:
            loss_fn = self.model.loss
            with tf.GradientTape() as tape:
                tape.watch(self.model.inputs)
                outputs = self.model(self.model.inputs, training=True)
                loss = loss_fn(y_true=outputs, y_pred=self.model.targets[0])
                gradients = tape.gradient(loss, self.model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Clip the predicted values to avoid NaNs
        if self.clip_value is not None:
            predicted_values = self.model.predict_on_batch(self.model.inputs)
            predicted_values = tf.clip_by_value(predicted_values, self.clip_value, 1.0 - self.clip_value)
            self.model._feed_targets = predicted_values


# take 2
def sequential_model(
        encoder: tf.keras.layers.experimental.preprocessing.TextVectorization,
        embedding_dim: int = 128,
        rnn_units: int = 256,
        batch_size: int = 64,
        max_tokens: int = 5000,
        max_length: int = 1000) -> tf.keras.Model:
    """Create a sequential RNN model for text generation"""
    input_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int64)
    regularizer = tf.keras.regularizers.l2(0.0001)
    model = tf.keras.Sequential()
    model.add(input_layer)
    # model.add(encoder)
    model.add(tf.keras.layers.Embedding(
        encoder.vocabulary_size(),
        embedding_dim,
        input_length=max_length))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        rnn_units*2,
        return_sequences=True,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        bias_regularizer=regularizer,
        dropout=0.2,
        # recurrent_dropout=0.3 # this stops CUDNN from being used
        )))
    # model.add(tf.keras.layers.Reshape((max_length, rnn_units)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        rnn_units,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        bias_regularizer=regularizer,
        dropout=0.2,
        # recurrent_dropout=0.3 # this stops CUDNN from being used
        )))
    model.add(tf.keras.layers.Dense(rnn_units))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(max_length))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # perp = Perplexity(input_dim=max_tokens, input_shape=(None, max_tokens, max_length))
    metrics = tf.metrics.Accuracy()  # for now...
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


# take 3
def rnn_model(
        encoder: tf.keras.layers.experimental.preprocessing.TextVectorization,
        embedding_dim: int = 128,
        rnn_units: int = 256,
        max_length: int = 1000,
        batch_size: int = 32,
        softmax: bool = True) -> tf.keras.Model:
    """Create an RNN model for text generation"""
    # define the input shape
    input_shape = (max_length,)

    # define the input layer
    input_layer = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

    # define the embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=encoder.vocabulary_size(),
        output_dim=embedding_dim,
        input_length=max_length,
        mask_zero=True)(input_layer)

    regularizer = tf.keras.regularizers.l2(0.0001)

    # loss = tfa.seq2seq.SequenceLoss()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # metrics = tf.metrics.Accuracy()
    optimizer = tf.keras.optimizers.experimental.AdamW()
    # define the first bidirectional GRU layer
    gru_layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        units=rnn_units,
        return_sequences=True,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        bias_regularizer=regularizer,
        stateful=False,
        recurrent_initializer='glorot_uniform',
        dropout=0.1,
        ))(embedding_layer)

    # define the second bidirectional GRU layer
    gru_layer_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        units=rnn_units,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        bias_regularizer=regularizer,
        recurrent_initializer='glorot_uniform',
        return_sequences=False,
        stateful=False,
        dropout=0.1
        ))(gru_layer_1)

    # dense_layer = tf.keras.layers.Dense(units=rnn_units//2)(gru_layer_2)

    # define the output layer with linear activation
    output_layer = tf.keras.layers.Dense(units=encoder.vocabulary_size(), activation='softmax')(gru_layer_2)

    # define the model with input and output layers
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # compile the model with categorical_crossentropy loss and Adam optimizer
    model.compile(loss=loss, optimizer=optimizer)

    # print the model summary
    model.summary(expand_nested=True)

    return model


def perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    perplexity = tf.exp(tf.keras.backend.mean(cross_entropy))
    return perplexity


def get_prediction_mask(encoder):
    vocab = encoder.get_vocabulary()
    inverted_vocab = {v: k for k, v in enumerate(vocab)}
    skip_ids = [inverted_vocab['<START>'], inverted_vocab['<END>'], inverted_vocab['<ITEM>'], inverted_vocab['[UNK]']]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=np.array(sorted(skip_ids)).reshape(-1, 1),
        # Match the shape to the vocabulary
        dense_shape=[len(vocab)])

    return tf.sparse.to_dense(sparse_mask)


def generate_text_sequence(model, encoder, prompt, temperature=1.0, min_length=0, max_n=10):
    # make sure temperature is within bounds
    temperature = temperature if temperature > 0 else 10e-6
    temperature = temperature if temperature < 1 else 1.0
    prompt = strip_standardize_py(prompt)
    prompt_tokens = len(prompt.split())
    vocab = encoder.get_vocabulary()
    prediction_mask = get_prediction_mask(encoder)

    last_token = -1
    current_loop = 0
    max_loops = 20
    while len(prompt.split()) - prompt_tokens <= min_length:
        current_loop += 1
        if current_loop > max_loops:
            break
        encoded_prompt = encoder([prompt])
        output_sequence = model.predict(encoded_prompt, verbose=0)
        output_sequence = tf.squeeze(output_sequence, axis=0)
        output_sequence = output_sequence / temperature
        output_sequence = output_sequence + prediction_mask
        # output_sequence = tf.nn.softmax(output_sequence, axis=-1)
        if max_n > 1:
            _, top_n_indices = tf.math.top_k(output_sequence, k=max_n)
            top_n_indices = np.array(top_n_indices)
            selected_indices = np.random.choice(max_n, size=output_sequence.shape[0])
            selected_indices = top_n_indices[np.arange(len(selected_indices)), selected_indices]
        else:
            # just take argmax
            selected_indices = tf.math.argmax(output_sequence, axis=-1)
        generated_text = ''

        for i in range(len(selected_indices)):
            if selected_indices[i] == last_token:
                last_token = selected_indices[i]
                continue
            generated_text = generated_text.strip() + ' ' + vocab[selected_indices[i]]
            last_token = selected_indices[i]
        prompt = prompt.strip() + ' ' + generated_text.strip()
    prompt = prompt.replace('<ITEM>', '')
    prompt = prompt.replace('<START>', '')
    return prompt


def generate_text_by_word(model, encoder, prompt, max_len=300, temperature=1.0):
    # make sure temperature is within bounds
    temperature = temperature if temperature > 0 else 10e-6
    temperature = temperature if temperature < 1 else 1.0
    prompt = strip_standardize_py(prompt)
    prediction_mask = get_prediction_mask(encoder)

    for i in tqdm(range(len(prompt.split(' ')), max_len)):
        generated_text = encoder([prompt])
        predictions = model.predict(generated_text, verbose=0)
        predictions = predictions[:, -1, :]
        predictions = predictions / temperature
        # predicted_id = tf.nn.softmax(predictions, axis=-1)
        predictions = predictions + prediction_mask
        # predicted_id = tf.argmax(predicted_id, axis=-1)
        predicted_id = tf.random.categorical(predictions, num_samples=1)
        predicted_id = tf.squeeze(predicted_id, axis=-1)
        predicted_word = encoder.get_vocabulary()[predicted_id[0]]
        if predicted_word == '<END>' or i == max_len - 1:
            break
        prompt += ' ' + predicted_word
    # remove any '<ITEM>', '<START>' tokens
    prompt = prompt.replace('<ITEM>', '')
    prompt = prompt.replace('<START>', '')

    return prompt


# take 4
def rnn_model_context(
        encoder: tf.keras.layers.experimental.preprocessing.TextVectorization,
        embedding_dim: int = 256,
        rnn_units: int = 512,
        max_length: int = 300,
        batch_size: int = 64) -> tf.keras.Model:

    article_comment_shape = (max_length, )
    # input layers
    article_comment_input = tf.keras.layers.Input(shape=article_comment_shape, batch_size=batch_size, dtype=tf.int16)
    # embedding layers
    vocab_size = encoder.vocabulary_size()
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        mask_zero=True)
    embedded_article_comment = embedding_layer(article_comment_input)
    # bi-directional GRU layer
    gru_layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        units=rnn_units,
        return_sequences=True,
        stateful=False,
        recurrent_initializer='glorot_uniform',
        dropout=0.1,
        ))(embedded_article_comment)
    dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))(gru_layer_1)
    # define the model with input and output layers
    model = tf.keras.models.Model(inputs=article_comment_input, outputs=dense_layer)
    # compile the model with categorical_crossentropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[perplexity])
    # print the model summary
    model.summary(expand_nested=True)
    return model


# take 1
class OverKillerRNN(tf.keras.Model):
    """OverKillerRNN is a RNN model that can be used to train a model to generate text."""

    _encoder: tf.keras.layers.experimental.preprocessing.TextVectorization
    _batch_size: int
    _layers: int = 0

    _max_length: int = 0
    _vocab_size: int = 0
    _rnn_units: int = 0
    _embedding_dim: int = 0

    def __init__(
            self,
            encoder: tf.keras.layers.experimental.preprocessing.TextVectorization,
            embedding_dim: int = 128,
            rnn_units: int = 256,
            batch_size: int = 64,
            max_length: int = 0,
            **kwargs):
        super(OverKillerRNN, self).__init__(**kwargs)
        self._batch_size = batch_size
        self.encoder = encoder
        self._vocab_size = encoder.vocabulary_size()
        self._max_length = max_length
        self._rnn_units = rnn_units
        self._embedding_dim = embedding_dim
        self._layers = 0

        # variable length int sequences
        self._query_input = tf.keras.Input(shape=(self._batch_size, ), dtype=tf.int16, name='query_input')
        self._reply_input = tf.keras.Input(shape=(None,), dtype=tf.int16, name='reply_input')
        # Embedding layer
        self._embedding_layer = self._embedding(self._vocab_size, self._embedding_dim)
        self._query_embedding = self._embedding_layer(self._query_input)
        self._reply_embedding = self._embedding_layer(self._reply_input)
        # Bidirectional layer
        self._bidirectional_layer = self._rnn_bidirectional(self._rnn_units, return_state=True)
        self._bidirectional_layer_2 = self._rnn_bidirectional(self._rnn_units, return_state=False)
        # self._query_seq_encoding = self._bidirectional_layer(self._query_embedding)
        # self._reply_seq_encoding = self._bidirectional_layer(self._reply_embedding)

        # Self-attention layer
        # self._self_attention_layer = self._self_attention(use_scale=True)
        # self._query_reply_attention_seq = self._self_attention_layer([self._query_seq_encoding, self._reply_seq_encoding])
        # self._query_encoding = tf.keras.layers.GlobalAveragePooling1D()(self._query_seq_encoding)
        # self._query_reply_attention = tf.keras.layers.GlobalAveragePooling1D()(self._query_reply_attention_seq)
        # self._input_layer = tf.keras.layers.Concatenate()([self._query_encoding, self._query_reply_attention])
        # Dense layer
        self._dense_layer = self._dense(self._vocab_size)

        # Dropout layer
        self._dropout_layer = self._dropout(0.5)

        # Dense layer
        self._dense_layer_2 = self._dense(self._vocab_size)

    def _rnn_bidirectional(self, rnn_units: int, lstm: bool = False, **kwargs) -> tf.keras.layers.Bidirectional:
        name = f'bidirectional_{self._layers}'
        self._layers += 1

        if lstm:
            return tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(rnn_units, return_sequences=True, name=name, **kwargs))
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(rnn_units, return_sequences=True, name=name, **kwargs))

    def _self_attention(self, **kwargs) -> tf.keras.layers.AdditiveAttention:
        name = f'self_attention_{self._layers}'
        self._layers += 1
        return tf.keras.layers.AdditiveAttention(name=name, **kwargs)

    def _dense(self, units: int, **kwargs) -> tf.keras.layers.Dense:
        name = f'dense_{self._layers}'
        self._layers += 1
        return tf.keras.layers.Dense(units, name=name, **kwargs)

    def _embedding(self, in_dim: int, out_dim: int, **kwargs) -> tf.keras.layers.Embedding:
        name = f'embedding_{self._layers}'
        self._layers += 1
        return tf.keras.layers.Embedding(in_dim, out_dim, name=name, mask_zero=True, **kwargs)

    def _dropout(self, rate: float, **kwargs) -> tf.keras.layers.Dropout:
        name = f'dropout_{self._layers}'
        self._layers += 1
        return tf.keras.layers.Dropout(rate, name=name, **kwargs)

    def _rnn(self, rnn_units: int, lstm: bool = False, **kwargs) -> tf.keras.layers.RNN:
        name = f'rnn_{self._layers}'
        self._layers += 1
        if lstm:
            return tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, name=name, **kwargs)
        return tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True, name=name, **kwargs)

    def call(self, inputs: tf.Tensor, states=None, return_state=False, training=False) -> tf.Tensor:
        x = inputs
        x = self.encoder(inputs)
        x = self._embedding_layer(x)
        if states is None:
            states = self._bidirectional_layer.layer.get_initial_state(x)
        # # Add additional features
        # additional_features = kwargs.get("additional_features")
        # if additional_features is not None:
        #     x = tf.concat([x, additional_features], axis=-1)

        x, states = self._bidirectional_layer(x, initial_state=[states], training=training)
        # x = self._bidirectional_layer_2(x)
        # x = self._self_attention_layer(x)
        x = self._dense_layer(x)
        x = self._dropout_layer(x)
        x = self._dense_layer_2(x)

        if return_state:
            return x, states
        return x  # type: ignore


def save_best_callback(model_save_path, monitor="perplexity"):
    return ModelCheckpoint(
        filepath=model_save_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        save_freq=1500,
        verbose=1)






































    # def __init__(
    #         self,
    #         encoder: tf.keras.layers.experimental.preprocessing.TextVectorization,
    #         embedding_dim: int,
    #         rnn_units: int,
    #         batch_size: int):
    #     super(OverKillerRNN, self).__init__()
    #     self.batch_size = batch_size
    #     self.encoder = encoder
    #     self.vocab_size = len(encoder.get_vocabulary())
    #     self.embedding = tf.keras.layers.Embedding(self.vocab_size, embedding_dim)
    #     self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
    #     self.lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
    #     self.birnn = tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True))
    #     self.dense = tf.keras.layers.Dense(self.vocab_size)

    # def call(self, inputs, states=None, return_state=False, training=False):
    #     x = self.encoder(inputs)
    #     x = self.embedding(x, training=training)
    #     if states is None:
    #         states = self.gru.get_initial_state(x)
    #     x, state = self.gru(x, initial_state=states, training=training)  # type: ignore
    #     x, state = self.lstm(x, initial_state=states, training=training)  # type: ignore
    #     x, state = self.birnn(x, initial_state=states, training=training)  # type: ignore
    #     x = self.dense(x, training=training)

    #     if return_state:
    #         return x, state
    #     else:
    #         return x

    # def generate_text(self, start_string: str, num_generate: int = 1000) -> str:
    #     input_eval = [start_string]
    #     input_eval = self.encoder(input_eval)
    #     text_generated = []

    #     temperature = 1.0

    #     self.reset_states()
    #     for _ in range(num_generate):
    #         predictions = self(input_eval)
    #         predictions = tf.squeeze(predictions, 0)

    #         predictions = predictions / temperature
    #         predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

    #         input_eval = tf.expand_dims([predicted_id], 0)
    #         text_generated.append(self.encoder.get_vocabulary()[predicted_id])

    #     return (start_string + ''.join(text_generated))
