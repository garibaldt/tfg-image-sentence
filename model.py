from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense
from keras.layers import BatchNormalization, Embedding, Activation, Reshape
from keras.layers.merge import Add
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2

def model(max_token_length, vocabulary_size, rnn='lstm' ,num_image_features=4096,
        hidden_size=512, embedding_size=512, regularizer=1e-8):

    # word embedding
    text_input = Input(shape=(max_token_length, vocabulary_size), name='text')
    # masking
    text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
    # time distributed: applies a layer to every temporal slice of an input
    text_to_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='text_embedding'))(text_mask)

    text_dropout = Dropout(.5, name='text_dropout')(text_to_embedding)

    # image embedding
    image_input = Input(shape=(max_token_length, num_image_features),
                                                        name='image')
    image_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='image_embedding'))(image_input)
    image_dropout = Dropout(.5,name='image_dropout')(image_embedding)

    # language model
    recurrent_inputs = [text_dropout, image_dropout]
    # merge text and image for the model
    merged_input = Add()(recurrent_inputs)
    # implement model using lstm or gru
    if rnn == 'lstm':
        recurrent_network = LSTM(units=hidden_size,
                                recurrent_regularizer=l2(regularizer),
                                kernel_regularizer=l2(regularizer),
                                bias_regularizer=l2(regularizer),
                                return_sequences=True,
                                name='recurrent_network')(merged_input)

    elif rnn == 'gru':
        recurrent_network = GRU(units=hidden_size,
                                recurrent_regularizer=l2(regularizer),
                                kernel_regularizer=l2(regularizer),
                                bias_regularizer=l2(regularizer),
                                return_sequences=True,
                                name='recurrent_network')(merged_input)
    else:
        raise Exception('Invalid rnn name')

    output = TimeDistributed(Dense(units=vocabulary_size,
                                    kernel_regularizer=l2(regularizer),
                                    activation='softmax'),
                                    name='output')(recurrent_network)

    inputs = [text_input, image_input]
    model = Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    model = model(16, 1024)
    plot_model(model, './my_model.png')
