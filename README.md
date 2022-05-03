# homr_competition
Handwritten Optical Music Recognition (HOMR).

In this project we implemet a Convolutional Recursive Neural Network for the task of human handwriting recognition. Namely algorithm learns to recognize handwritten musical notes.

## Project layout

- homr_competition.py - Main driving script. Trains the model and saves predictions for the test dataeset in a text file.
- homr_model.py - Defines neural network used.
- homr_dataset.py - Project dataset definition;
- homr_dataset_utils.py - Minor utility functions for building data pipeline.

## Dataset & evaluation

The inputs are grayscale images of monophonic scores starting with a clef, key signature, and a time signature, followed by several staves. The dataset is downloaded automatically via homr_dataset.py module if missing (is has ~500MB, so it might take a while). Sample training data together with labels can be found [here](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/demos/homr_train.html).

The evaluation is performed by computing edit distance of predicted sequence to the target (true) sequence, normalized by its length. The smaller the average edit distance is, the better.

## Neural Network Architecture

Images are fed through two one-dimensional convolutional layers followed by two recurrent GRU layers followed by two fully connected layers. The model is trained using CTC loss function, so that it can learn to align its outputs to target labels, which generally are of different length. To prevent overfitting we use batch normalization, dropout and l2 regularization.

## Model parameters

For all model parameters there are sensible defaults (set at the top of homr_competition.py), so the project can be executed simply by running homr_competition.py script without parameters.

## Results

With default parameters the proposed model achives 0.02 avg. edit distance for the validation set.
