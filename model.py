import pathlib
import glob
import collections
import tensorflow as tf
import numpy as np
import pandas as pd
import pretty_midi as pm
from typing import Optional


class Model():
    def __init__(self):

        #seed
        self.seed = 42
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        
        #init var
        self.key_order = ['pitch', 'step', 'duration']
        self.vocab_size = 128
        self.sequence_length = 20

        #training data
        self.data_dir = pathlib.Path('data/maestro-v2.0.0')
        if not self.data_dir.exists():
            tf.keras.utils.get_file(
                'maestro-v2.0.0-midi.zip',
                origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
                extract=True,
                cache_dir='.', cache_subdir='data',
                )
        
        #prepare data
        self.filenames = glob.glob(str(self.data_dir/'**/*.mid*'))
        print(len(self.filenames), ' training files available.')
        self.sample_file = self.filenames[1]
        self.raw_notes = Model.midi_to_notes(self.sample_file)
        self.raw_notes.head()
        self.midi_data = pm.PrettyMIDI(self.sample_file)
        self.instrument = self.midi_data.instruments[0]
        self.instrument_name = pm.program_to_instrument_name(self.instrument.program)

        self.all_notes = []
        for f in self.filenames[:10]:
            notes = Model.midi_to_notes(f)
            self.all_notes.append(notes)
        self.all_notes = pd.concat(self.all_notes)
        self.n_notes = len(self.all_notes)

        self.train_notes = np.stack([self.all_notes[key] for key in self.key_order], axis=1)
        self.notes_ds = tf.data.Dataset.from_tensor_slices(self.train_notes)
        self.seq_ds = self.create_sequences(self.notes_ds, self.sequence_length, self.vocab_size)
        buffer_size = self.n_notes - self.sequence_length  # the number of items in the dataset
        self.train_ds = (self.seq_ds
            .shuffle(buffer_size)
            .batch(64, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))

        #Input
        self.input_shape = (self.sequence_length, 3)
        self.learning_rate = 0.004
        self.inputs = tf.keras.Input(self.input_shape)
        
        #Hidden - LSTM
        self.x = tf.keras.layers.LSTM(128)(self.inputs)
        
        #Output -Dense
        self.outputs = {
            'pitch': tf.keras.layers.Dense(128, name='pitch')(self.x),
            'step': tf.keras.layers.Dense(1, name='step')(self.x),           
            'duration': tf.keras.layers.Dense(1, name='duration')(self.x),
        }

        #define model
        self.model = tf.keras.Model(self.inputs, self.outputs)
        
        #loss and optimizer
        self.loss = {
            'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'step': self.mse_with_positive_pressure,
            'duration': self.mse_with_positive_pressure,
        }
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        #compile and evaluate model
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.summary()
        self.losses = self.model.evaluate(self.train_ds, return_dict=True)
        self.model.compile(
            loss=self.loss,
            loss_weights={
                'pitch': 0.05,
                'step': 1.0,
                'duration':1.0,
            },
            optimizer=self.optimizer,
            )
        self.model.evaluate(self.train_ds, return_dict=True)
        
        #training checkpoints
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./training_checkpoints/ckpt_{epoch}',
                save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True),
            ]

    def train_model(self, epochs: int):
        self.epochs = epochs
        self.history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            callbacks=self.callbacks,
            )

    def midi_to_notes(midi_file: str) -> pd.DataFrame:
        #prepare midi_notes for tf
        if midi_file == '': return pd.DataFrame()
        midi_data = pm.PrettyMIDI(midi_file)
        instrument = midi_data.instruments[0]
        notes = collections.defaultdict(list)

        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

    def notes_to_midi(
        notes: pd.DataFrame,
        velocity: int = 100,
        instrument_name: Optional[str] = 'Acoustic Grand Piano',
        out_file: Optional[str] = 'new.mid' 
        ) -> pm.PrettyMIDI:
        #transfer tf_data to midi_data
        midi_data = pm.PrettyMIDI()
        instrument = pm.Instrument(
            program=pm.instrument_name_to_program(
                instrument_name))

        prev_start = 0
        for i, note in notes.iterrows():
            start = float(prev_start + note['step'])
            end = float(start + note['duration'])
            note = pm.Note(
                velocity=velocity,
                pitch=int(note['pitch']),
                start=start,
                end=end,
            )
            instrument.notes.append(note)
            prev_start = start

        midi_data.instruments.append(instrument)
        midi_data.write(out_file)
        return midi_data
    
    def create_sequences(
        self,
        dataset: tf.data.Dataset, 
        seq_length: int,
        vocab_size = 128,
    ) -> tf.data.Dataset:
        #Returns TF Dataset of sequence and label examples. From pd to tf
        seq_length = seq_length+1

        # Take 1 extra for the labels
        windows = dataset.window(seq_length, shift=1, stride=1,
                                    drop_remainder=True)

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(seq_length, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        # Normalize note pitch
        def scale_pitch(x):
            x = x/[vocab_size,1.0,1.0]
            return x

        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            labels_dense = sequences[-1]
            labels = {key:labels_dense[i] for i,key in enumerate(self.key_order)}

            return scale_pitch(inputs), labels

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    
    def mse_with_positive_pressure(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        #custom mean square error loss function
        mse = (y_true - y_pred) ** 2
        positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
        return tf.reduce_mean(mse + positive_pressure)
    
    def predict_next_note(
        self,
        notes: np.ndarray, 
        model: tf.keras.Model, 
        temperature: float = 1.0
    ) -> tuple[int, float, float]:
        #Generates a note as a tuple of (pitch, step, duration), using a trained sequence model.

        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = model.predict(inputs)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)

        # `step` and `duration` values should be non-negative
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration)

    def predict_notes(self, num_predictions: int = 120):
        temperature = 2.0

        sample_notes = np.stack([self.raw_notes[key] for key in self.key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training
        # sequences
        input_notes = (
            sample_notes[:self.sequence_length] / np.array([self.vocab_size, 1, 1]))

        generated_notes = []
        prev_start = 0
        for _ in range(num_predictions):
            pitch, step, duration = self.predict_next_note(input_notes, self.model, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(
            generated_notes, columns=(*self.key_order, 'start', 'end'))

        self.out_pm = Model.notes_to_midi(generated_notes)
        return generated_notes

def main():
    mo = Model()
    mo.train_model(5)
    mo.predict_notes()


if __name__=='__main__':
    main()