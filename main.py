# Importing Main Libraries For Empathy Prediction Problem Set
import torch
import numpy as np

torch.cuda.current_device()
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
import warnings
from torch.utils.data import DataLoader
from data_embeddings import EmpathyDataLoading, DataProcessing
from lstm_models import LSTM_fix_input, LSTM_var_input, LSTM_glove_vecs_input
from training_testing_criterion import train, get_criterion_optimizer_scheduler

warnings.filterwarnings('ignore')
np.random.seed(1)

# Model training using GPU in CUDA Environment
print("Whether Cuda is Available: {}".format(torch.cuda.is_available()))

# File Names
label_message_file = "/media/HDD_2TB.1/Empathy-Predictions/labeled_messages.csv"
empathies = "/media/HDD_2TB.1/Empathy-Predictions/empathies.csv"


# Method that calls to train different types of LSTMs
def training_LSTMs(model, epochs, learning_rate, loss_weights, device, train_queue, valid_queue):
    model = model.to(device)
    criterion, optimizer, scheduler = get_criterion_optimizer_scheduler(model, epochs, learning_rate,
                                                                        loss_weights, device)

    for epoch in range(epochs):
        scheduler.step()
        train(model, device, train_queue, valid_queue, optimizer, epoch, criterion)


# Main Method
def main():
    # Object 'Data' is created by Class Named -> 'DataProcessing'. file names are parameters
    data_obj = DataProcessing(label_message_file, empathies)

    # Method describes messages lengths, number of words in Corpus 
    data_obj.describe_counts()

    # Method for MultiLabel Encoding, weighted label weights for label data imbalance
    output_size, loss_weights = data_obj.label_binarizer_get_weights()

    # Get X and Y
    X = data_obj.get_X_data()
    y = data_obj.get_Y_data()
    print('\n')

    # Training and Testing Split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)

    # Baseline Classifier using Support Vector Classifier to calculate Accuracy and Area Under Curve 
    print("*** Baseline AUC Scores ***")
    acc_svm, roc_svm = data_obj.modelling("SVC", X_train, X_valid, y_train, y_valid)
    print("SVM Modelling --> Validation Acc. : %.3f, Validation AUC Score : %.3f" % (acc_svm, roc_svm))
    acc_RF, roc_RF = data_obj.modelling("RandomForest", X_train, X_valid, y_train, y_valid)
    print("Random Forest Modelling --> Validation Acc. : %.3f, Validation AUC Score : %.3f" % (acc_RF, roc_RF))
    print("** Statistical Method performed better then Baseline **")

    # 'EmpathyDataLoading' Class for training and validation data to load while run time
    train_ds = EmpathyDataLoading(X_train, y_train)
    valid_ds = EmpathyDataLoading(X_valid, y_valid)

    vocab_size = len(data_obj.words)
    epochs = 1001
    batch_size = 1000
    learning_rate = 0.3

    # Data Loader for train and test
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_queue = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # CUDA Environment Conf.
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    torch.manual_seed(1)
    cudnn.enabled = True
    torch.cuda.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # LSTMs models with fixed Input length, var Input length, using Stanford Glove Vec Representations
    print('\n')
    print('-----------LSTMs Fixed Input Length--------------')
    model_fix_len = LSTM_fix_input(vocab_size, 48, 96, output_size)
    training_LSTMs(model_fix_len, epochs, learning_rate, loss_weights, device, train_queue, valid_queue)

    print('\n')
    print('-----------LSTMs Var Input Length--------------')
    model_var_len = LSTM_var_input(vocab_size, 48, 96, output_size)
    training_LSTMs(model_var_len, epochs, learning_rate, loss_weights, device, train_queue, valid_queue)

    print('\n')
    print('-----------LSTMs with Glove Representations--------------')
    word_vecs = data_obj.load_glove_vectors()
    pretrained_weights, vocab, vocab2index = data_obj.get_emb_matrix(word_vecs)
    model_glove = LSTM_glove_vecs_input(vocab_size, 50, 96, pretrained_weights, output_size)
    training_LSTMs(model_glove, epochs, learning_rate, loss_weights, device, train_queue, valid_queue)


if __name__ == '__main__':
    main()
