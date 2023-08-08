import pandas as pd
from training_class import DCNN
import sys
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import random
from bert import bert_tokenization

dataset = pd.read_csv(sys.argv[1])
data_clean = list(dataset['data_clean'])
data_labels = list(dataset['data_labels'])
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1',
                               trainable = False)

min_token = 7
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

def encode_sentence(sent):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

data_inputs = [encode_sentence(sentence) for sentence in data_clean]

data_with_len = [
        [
            sent, data_labels[i], len(sent)
        ]
        for i, sent in enumerate(data_inputs)
    ]
random.shuffle(data_with_len)
data_with_len.sort(key=lambda x: x[2])
sorted_all = [
            (sent_lab[0], sent_lab[1])
            for sent_lab in data_with_len if sent_lab[2] > min_token
        ]
BATCH_SIZE = 32
NB_BATCHES = len(sorted_all) // BATCH_SIZE
NB_BATCHES_TEST = NB_BATCHES // 10

all_dataset = tf.data.Dataset.from_generator(lambda: sorted_all, 
                                            output_types = (tf.int32, tf.int32))
all_batched = all_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,),()))

test_dataset = all_batched.take(NB_BATCHES_TEST)
train_dataset = all_batched.skip(NB_BATCHES_TEST)

VOCAB_SIZE = len(tokenizer.vocab)
EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2
DROPOUT_RATE = 0.2
NB_EPOCHS = 2

Dcnn = DCNN(
    vocab_size = VOCAB_SIZE,
    emb_dim = EMB_DIM,
    nb_filters = NB_FILTERS,
    FFN_units = FFN_UNITS,
    nb_classes = NB_CLASSES,
    dropout_rate = DROPOUT_RATE
)

if  NB_CLASSES == 2:
    Dcnn.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
else:
    Dcnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['sparse_categorical_accuracy'])
        
checkpoint_path = './'
ckpt = tf.train.Checkpoint(Dcnn= Dcnn)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                    max_to_keep=1)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Lastest checkpoint restored')

class MyCustomCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        ckpt_manager.save()
        print(f"Checkpoint save at {checkpoint_path}")


history = Dcnn.fit(
    train_dataset,
    epochs = NB_EPOCHS,
    callbacks = [MyCustomCallBack()]
)

results = Dcnn.evaluate(test_dataset)
print(results)