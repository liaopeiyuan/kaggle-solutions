'''
Author: Yaqi Zhang
Date: 12/17/17
University of Wisconsin-Madison
'''
from keras.models import load_model
from resnetmodel_build import *
import numpy as np
import sys
import os

from keras.models import Sequential
from keras.layers import Dense

def mostcommon(array):
    '''return the most common value of an array'''
    return np.bincount(array).argmax()

def weighted_vote(x_test, models, accuracy_records, num_classes=10):
    ''' return final_predict based on weighted_vote of all the learners in models
        weight is the the accuracy of each learner
    '''
    n_learners = len(models)
    n_tests = x_test.shape[0]
    # final_predict = np.zeros((n_tests, 1), dtype="int64")
    probs = np.zeros((n_tests, num_classes))
    for i in range(n_learners):
        accuracy = accuracy_records[i]
        model = models[i]
        # probs = probs + accuracy*model.predict_proba(x_test)
        probs = probs + accuracy*model.predict(x_test)
    return np.argmax(probs, axis=1)

def majority_vote(x_test, models, accuracy_records):
    ''' return final_predict based on majority vote of all the learners in models
    '''
    n_learners = len(models)
    n_tests = x_test.shape[0]
    predictions = np.zeros((n_tests, n_learners), dtype="int64")
    for i in range(n_learners):
        model = models[i]
        predictions[:, i] = predict(model, x_test) # each column stores one learner's prediction
    final_predict = np.zeros((n_tests, 1), dtype="int64")
    for i in range(n_tests):
        final_predict[i] = mostcommon(predictions[i, :])
    return final_predict

def cross_validation():
    '''use different parameters and pick the best one to be the final learner'''
    pass


def adaboost_original(version, n, n_learners, epochs_lst, batch_size, sample_ratio=3, filename="temp.txt", file_prefix=""):
    ''' adaboost of multi classification'''
    num_classes = 10
    K = float(num_classes)
    data_augmentation = True

    (x_train, y_train), (x_test, y_test) = load_data()
    y_train_old = y_train[:]
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    # weights = 1.0/n_trains*np.ones(n_trains) # initialize instance weights
    weights = [1.0/n_trains for k in range(n_trains)]
    M = sample_ratio*n_trains # >> sample a large (>> m) unweighted set of instance according to p
    test_accuracy_records = []
    alphas = []
    for i in range(n_learners):
        # weights = weights/sum(weights)
        sum_weights = sum(weights)
        weights = [weight/sum_weights for weight in weights]
        epochs = epochs_lst[i]
        # model = build_model(x_train, num_classes)

        train_picks = np.random.choice(n_trains, M, weights)

        x_train_i = x_train[train_picks, :]
        y_train_i = y_train[train_picks, :]
        # model, history = build_resnet(x_train_i, y_train_i, x_test, y_test, input_shape, batch_size, epochs, num_classes, n, version, data_augmentation)
        # model, history = train(x_train_i, y_train_i, x_test, y_test, model, batch_size, epochs)
        model, history = build_resnet(x_train_i, y_train_i, x_test, y_test, batch_size, epochs, n, version, data_augmentation, "adaboost-model-"+str(i))

        print("model " + str(i))
        predicts = predict(model, x_train_i)
        y_ref = y_train_old[train_picks, :].reshape((M, ))
        num_error = np.count_nonzero(predicts - y_ref)
        error = float(num_error)/M
        w_changed = np.zeros(n_trains)
        assert error < 0.5
        alpha = np.log((1 - error)/error) # + np.log(K - 1)
        for j in range(M):
            index = train_picks[j]
            if predicts[j] != y_ref[j] and w_changed[index] == 0:
                w_changed[index] = 1
                weights[index] = weights[index] * np.exp(alpha)
        alphas.append(alpha)
        print("alpha = " + str(alpha))
        models.append(model) # save base learner
        scores = evaluate(model, x_test, y_test)
        test_accuracy_records.append(scores[1])

    save_final_models(models, "adaboost-original")

    # final_predict = majority_vote(x_test, models, alphas)
    final_predict = weighted_vote(x_test, models, alphas)
    print(final_predict.shape)
    errors = np.count_nonzero(final_predict.reshape((n_tests, )) - y_test_old.reshape((n_tests,)))

    filename = file_prefix + filename
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    print('sample ratio = %f' % (sample_ratio))
    print('ensemble test accuracy: %f' % ((n_tests - errors)/float(n_tests)))
    out_file.write('sample ratio = %f\n' % (sample_ratio))
    out_file.write('ensemble test accuracy: %f\n' % ((n_tests - errors)/float(n_tests)))

    for i in range(n_learners):
        print("learner %d (epochs = %d): %0.6f" % (i, epochs_lst[i], test_accuracy_records[i]))
        out_file.write("learner %d (epochs = %d): %0.6f\n" % (i, epochs_lst[i], test_accuracy_records[i]))
    out_file.close()




def adaboost(version, n, n_learners, epochs_lst, batch_size, sample_ratio=3, filename="temp.txt", file_prefix=""):
    ''' adaboost of multi classification'''
    num_classes = 10
    K = float(num_classes)
    data_augmentation = True

    (x_train, y_train), (x_test, y_test) = load_data()
    y_train_old = y_train[:]
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    # weights = 1.0/n_trains*np.ones(n_trains) # initialize instance weights
    weights = [1.0/n_trains for k in range(n_trains)]
    M = sample_ratio*n_trains # >> sample a large (>> m) unweighted set of instance according to p
    test_accuracy_records = []
    alphas = []
    for i in range(n_learners):
        # weights = weights/sum(weights)
        sum_weights = sum(weights)
        weights = [weight/sum_weights for weight in weights]
        epochs = epochs_lst[i]
        # model = build_model(x_train, num_classes)

        train_picks = np.random.choice(n_trains, M, weights)

        x_train_i = x_train[train_picks, :]
        y_train_i = y_train[train_picks, :]
        # model, history = build_resnet(x_train_i, y_train_i, x_test, y_test, input_shape, batch_size, epochs, num_classes, n, version, data_augmentation)
        # model, history = train(x_train_i, y_train_i, x_test, y_test, model, batch_size, epochs)
        model, history = build_resnet(x_train_i, y_train_i, x_test, y_test, batch_size, epochs, n, version, data_augmentation, "adaboost-model-"+str(i))

        print("model " + str(i))
        predicts = predict(model, x_train_i)
        y_ref = y_train_old[train_picks, :].reshape((M, ))
        num_error = np.count_nonzero(predicts - y_ref)
        error = float(num_error)/M
        w_changed = np.zeros(n_trains)
        alpha = np.log((1 - error)/error) + np.log(K - 1)
        for j in range(M):
            index = train_picks[j]
            if predicts[j] != y_ref[j] and w_changed[index] == 0:
                w_changed[index] = 1
                weights[index] = weights[index] * np.exp(alpha)
        alphas.append(alpha)
        print("alpha = " + str(alpha))
        models.append(model) # save base learner
        scores = evaluate(model, x_test, y_test)
        test_accuracy_records.append(scores[1])

    save_final_models(models, "adaboost")

    # final_predict = majority_vote(x_test, models, alphas)
    final_predict = weighted_vote(x_test, models, alphas)
    print(final_predict.shape)
    errors = np.count_nonzero(final_predict.reshape((n_tests, )) - y_test_old.reshape((n_tests,)))

    filename = file_prefix + filename
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    print('sample ratio = %f' % (sample_ratio))
    print('ensemble test accuracy: %f' % ((n_tests - errors)/float(n_tests)))
    out_file.write('sample ratio = %f\n' % (sample_ratio))
    out_file.write('ensemble test accuracy: %f\n' % ((n_tests - errors)/float(n_tests)))

    for i in range(n_learners):
        print("learner %d (epochs = %d): %0.6f" % (i, epochs_lst[i], test_accuracy_records[i]))
        out_file.write("learner %d (epochs = %d): %0.6f\n" % (i, epochs_lst[i], test_accuracy_records[i]))
    out_file.close()
    ## check diversity
    # for i in range(n_learners):
    #    for j in range(i+1, n_learners):
    #        print("the diversity between %d and %d is %f" %(i, j, diversity(x_test, y_test, y_test_old, models[i], models[j])))

def diversity(x_data, y_data, y_data_old, model1, model2):
    scores1 = evaluate(model1, x_data, y_data)
    accuracy1 = scores1[1]
    scores2 = evaluate(model2, x_data, y_data)
    accuracy2 = scores2[1]
    predicts1 = predict(model1, x_data)
    predicts2 = predict(model2, x_data)
    n_train = x_data.shape[0]
    num_correct = 0
    for j in range(n_train):
        if predicts1[j] == y_data_old[j] or predicts2[j] == y_data_old[j]:
            num_correct += 1
    combined_acc = num_correct/n_train
    diver = combined_acc - max(accuracy1, accuracy2)
    return diver

def save_final_models(models, name_prefix):
    for i, model in enumerate(models):
        model.save(name_prefix + "." + str(i) + ".final.hdf5")

def bagging_train_model(version, n, n_learners, epochs_lst, batch_size, votefuns, filename="temp.txt", file_prefix="", random=True):
    '''bagging, use unique model, can use multiple vote functions, votefuns are vote
       functions list
    '''
    # Training parameters
    data_augmentation = True
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data()
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    train_accuracy_records = []
    test_accuracy_records = []
    for i in range(n_learners):
        epochs = epochs_lst[i]
        if random:
            train_picks = np.random.choice(n_trains, n_trains)
            x_train_i = x_train[train_picks, :]
            y_train_i = y_train[train_picks, :]
        else:
            x_train_i = x_train
            y_train_i = y_train
        model, history = build_resnet(x_train, y_train, x_test, y_test, batch_size, epochs, n, version, data_augmentation, "bagging-model-"+str(i))

        print("model %d finished" % (i))
        scores = model.evaluate(x_train, y_train, verbose=1)
        train_accuracy_records.append(scores[1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_accuracy_records.append(scores[1])
        models.append(model) # save base learner
    save_final_models(models, "bagging")

    filename = file_prefix + filename
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    out_file.write("Random = " + str(random) + "\n")
    print("Random = " + str(random))
    for votefun in votefuns:
        # get weighted vote or majority vote based on the votefun
        final_predict = votefun(x_test, models, train_accuracy_records)
        print(final_predict.shape)
        errors = np.count_nonzero(final_predict.reshape((n_tests, )) - y_test_old.reshape((n_tests,)))
        out_file.write("votefun is\n")
        out_file.write(str(votefun) + "\n")
        out_file.write('ensemble test accuracy: %0.6f \n' % ((n_tests - errors)/float(n_tests)))
        print("votefun is ")
        print(votefun)
        print('ensemble test accuracy: %0.6f' % ((n_tests - errors)/float(n_tests)))
        for i in range(n_learners):
            print("learner %d (epochs = %d): %0.6f" % (i, epochs_lst[i], test_accuracy_records[i]))
            out_file.write("learner %d (epochs = %d): %0.6f\n" % (i, epochs_lst[i], test_accuracy_records[i]))
    out_file.close()

def bagging_loading_model(n_learners, saved_model_files, votefuns, filename="temp.txt", file_prefix=""):
    '''load models from saved files
       votefuns are vote functions list
    '''
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data()
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    train_accuracy_records = []
    test_accuracy_records = []
    for i in range(n_learners):
        model_file = saved_model_files[i]
        model = load_model(model_file)
        print("model " + str(i))
        scores = evaluate(model, x_test, y_test)
        test_accuracy_records.append(scores[1])
        scores = evaluate(model, x_train, y_train)
        train_accuracy_records.append(scores[1])
        models.append(model) # save base learner

    filename = file_prefix + filename
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    for votefun in votefuns:
        # get weighted vote or majority vote based on the votefun
        final_predict = votefun(x_test, models, train_accuracy_records)

        errors = np.count_nonzero(final_predict.reshape((n_tests, )) - y_test_old.reshape((n_tests,)))
        out_file.write("votefun is\n")
        out_file.write(str(votefun) + "\n")
        out_file.write('ensemble test accuracy: %0.6f \n' % ((n_tests - errors)/float(n_tests)))
        print("votefun is ")
        print(votefun)
        print('ensemble test accuracy: %0.6f' % ((n_tests - errors)/float(n_tests)))
        for i in range(n_learners):
            print("learner %d (model_file = %s): %0.6f" % (i, saved_model_files[i], test_accuracy_records[i]))
            out_file.write("learner %d (model_file = %s): %0.6f\n" % (i, saved_model_files[i], test_accuracy_records[i]))
    out_file.close()



def split(x_train, y_train):
    '''split x_train and y_train to train set and validation set'''
    x_train_new = x_train[:45000, :]
    x_val = x_train[-5000:, :]
    y_train_new = y_train[:45000, :]
    y_val = y_train[-5000:, :]
    return (x_train_new, y_train_new), (x_val, y_val)


def stack_train_model_super(version, n, n_learners, epochs_lst, batch_size, meta_epochs=40, filename="temp.txt"):
    '''stacking multiple saved models'''
    # Training parameters
    data_augmentation = True
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data()
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    (x_train, y_train), (x_val, y_val) = split(x_train, y_train)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    n_vals = x_val.shape[0]
    test_accuracy_records = []
    for i in range(n_learners):
        epochs = epochs_lst[i]
        model, history = build_resnet(x_train, y_train, x_test, y_test, batch_size, epochs, n, version, data_augmentation, "super-model-"+str(i))
        print("model %d finished" % (i))
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_accuracy_records.append(scores[1])
        # test_accuracy_records.append(history.history['val_acc'][-1])
        models.append(model) # save base learner
    save_final_models(models, "super")

    # construct meta learning problem
    # meta_x_train = np.zeros((n_trains, n_learners*num_classes), dtype="float32")
    meta_x_train = np.zeros((n_vals, n_learners*num_classes), dtype="float32")
    meta_x_test = np.zeros((n_tests, n_learners*num_classes), dtype="float32")
    for i in range(n_learners):
        # meta_x_train[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_train, verbose=0)
        meta_x_train[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_val, verbose=0)
        meta_x_test[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_test, verbose=0)
    # meta_y_train = y_train # use one hot encode
    meta_y_train = y_val
    meta_y_test = y_test
    super_model = meta_model(n_learners, num_classes)
    # callbacks
    save_dir = os.path.join(os.getcwd(), 'super_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # model_name="stack-{epoch:03d}-{val_acc:.4f}.hdf5"
    model_name = "best_super.hdf5"
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
    callbacks_list = [checkpoint]

    super_model.fit(meta_x_train, meta_y_train, batch_size=128, epochs=meta_epochs, validation_data=(meta_x_test, meta_y_test), shuffle=True, callbacks=callbacks_list)
    super_model.load_weights(filepath)
    scores = super_model.evaluate(meta_x_test, meta_y_test, verbose=1)
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    print('Stack test accuracy: ', scores[1])
    out_file.write('Stack test accuracy: %0.6f\n' % (scores[1]))
    for i in range(n_learners):
        print("learner %d (epochs = %d): %0.6f" % (i, epochs_lst[i], test_accuracy_records[i]))
        out_file.write("learner %d (epochs = %d): %0.6f\n" % (i, epochs_lst[i], test_accuracy_records[i]))
    out_file.close()


def stack_train_model(version, n, n_learners, epochs_lst, batch_size, meta_epochs=40, filename="temp.txt"):
    '''stacking multiple saved models'''
    # Training parameters
    data_augmentation = True
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data()
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    test_accuracy_records = []
    for i in range(n_learners):
        epochs = epochs_lst[i]
        model, history = build_resnet(x_train, y_train, x_test, y_test, batch_size, epochs, n, version, data_augmentation, "stack-model-"+str(i))
        print("model %d finished" % (i))
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_accuracy_records.append(scores[1])
        # test_accuracy_records.append(history.history['val_acc'][-1])
        models.append(model) # save base learner
    save_final_models(models, "stack")

    # construct meta learning problem
    meta_x_train = np.zeros((n_trains, n_learners*num_classes), dtype="float32")
    meta_x_test = np.zeros((n_tests, n_learners*num_classes), dtype="float32")
    for i in range(n_learners):
        meta_x_train[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_train, verbose=0)
        meta_x_test[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_test, verbose=0)
    meta_y_train = y_train # use one hot encode
    meta_y_test = y_test
    super_model = meta_model(n_learners, num_classes)
    # callbacks
    save_dir = os.path.join(os.getcwd(), 'stacking_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # model_name="stack-{epoch:03d}-{val_acc:.4f}.hdf5"
    model_name = "best_stack.hdf5"
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
    callbacks_list = [checkpoint]

    super_model.fit(meta_x_train, meta_y_train, batch_size=128, epochs=meta_epochs, validation_data=(meta_x_test, meta_y_test), shuffle=True, callbacks=callbacks_list)
    super_model.load_weights(filepath)
    scores = super_model.evaluate(meta_x_test, meta_y_test, verbose=1)
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    print('Stack test accuracy: ', scores[1])
    out_file.write('Stack test accuracy: %0.6f\n' % (scores[1]))
    for i in range(n_learners):
        print("learner %d (epochs = %d): %0.6f" % (i, epochs_lst[i], test_accuracy_records[i]))
        out_file.write("learner %d (epochs = %d): %0.6f\n" % (i, epochs_lst[i], test_accuracy_records[i]))
    out_file.close()


def stack_loading_model_super(saved_model_files, meta_epochs=40, filename="temp.txt"):
    '''stacking multiple saved models'''
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data()
    y_test_old = y_test[:] # save for error calculation
    y_train_old = y_train[:]
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    (x_train, y_train), (x_val, y_val) = split(x_train, y_train)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    n_vals = x_val.shape[0]
    test_accuracy_records = []
    n_learners = len(saved_model_files)
    for i in range(n_learners):
        model_file = saved_model_files[i]
        model = load_model(model_file)
        print("model " + str(i))
        scores = evaluate(model, x_test, y_test)
        test_accuracy_records.append(scores[1])
        models.append(model) # save base learner
    # construct meta learning problem
    # meta_x_train = np.zeros((n_trains, n_learners*num_classes), dtype="float32")
    meta_x_train = np.zeros((n_vals, n_learners*num_classes), dtype="float32")
    meta_x_test = np.zeros((n_tests, n_learners*num_classes), dtype="float32")
    for i in range(n_learners):
        # meta_x_train[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_train, verbose=0)
        meta_x_train[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_val, verbose=1)
        meta_x_test[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_test, verbose=0)
    # meta_y_train = y_train # use one hot encode
    meta_y_train = y_val
    meta_y_test = y_test
    super_model = meta_model(n_learners, num_classes)
    # callbacks
    save_dir = os.path.join(os.getcwd(), 'stacking_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # model_name="stack-{epoch:03d}-{val_acc:.4f}.hdf5"
    model_name = "best_stack.hdf5"
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    print(meta_x_train.shape)
    print(meta_y_train.shape)

    super_model.fit(meta_x_train, meta_y_train, batch_size=128, epochs=meta_epochs, validation_data=(meta_x_test, meta_y_test), shuffle=True, callbacks=callbacks_list)
    super_model.load_weights(filepath);
    scores = super_model.evaluate(meta_x_test, meta_y_test, verbose=1)
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    print('Stack test accuracy: ', scores[1])
    out_file.write('Stack test accuracy: %0.6f\n' % (scores[1]))
    for i in range(n_learners):
        print("learner %d (model_file = %s): %0.6f" % (i, saved_model_files[i], test_accuracy_records[i]))
        out_file.write("learner %d (model_file = %s): %0.6f\n" % (i, saved_model_files[i], test_accuracy_records[i]))
    out_file.close()


def stack_loading_model(saved_model_files, meta_epochs=40, filename="temp.txt"):
    '''stacking multiple saved models'''
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data()
    y_test_old = y_test[:] # save for error calculation
    y_train_old = y_train[:]
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    test_accuracy_records = []
    n_learners = len(saved_model_files)
    for i in range(n_learners):
        model_file = saved_model_files[i]
        model = load_model(model_file)
        print("model " + str(i))
        scores = evaluate(model, x_test, y_test)
        test_accuracy_records.append(scores[1])
        models.append(model) # save base learner
    # construct meta learning problem
    meta_x_train = np.zeros((n_trains, n_learners*num_classes), dtype="float32")
    meta_x_test = np.zeros((n_tests, n_learners*num_classes), dtype="float32")
    for i in range(n_learners):
        meta_x_train[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_train, verbose=0)
        meta_x_test[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_test, verbose=0)
    meta_y_train = y_train # use one hot encode
    meta_y_test = y_test
    super_model = meta_model(n_learners, num_classes)
    # callbacks
    save_dir = os.path.join(os.getcwd(), 'stacking_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # model_name="stack-{epoch:03d}-{val_acc:.4f}.hdf5"
    model_name = "best_stack.hdf5"
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    super_model.fit(meta_x_train, meta_y_train, batch_size=128, epochs=meta_epochs, validation_data=(meta_x_test, meta_y_test), shuffle=True, callbacks=callbacks_list)
    super_model.load_weights(filepath);
    scores = super_model.evaluate(meta_x_test, meta_y_test, verbose=1)
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    print('Stack test accuracy: ', scores[1])
    out_file.write('Stack test accuracy: %0.6f\n' % (scores[1]))
    for i in range(n_learners):
        print("learner %d (model_file = %s): %0.6f" % (i, saved_model_files[i], test_accuracy_records[i]))
        out_file.write("learner %d (model_file = %s): %0.6f\n" % (i, saved_model_files[i], test_accuracy_records[i]))
    out_file.close()

def test1():
    n_learners = 3
    batch_size = 32
    epochs_lst = [120, 120, 120]
    version = 1
    n = 3
    votefuns = [weighted_vote,  majority_vote]
    bagging_train_model(version, n, n_learners, epochs_lst, batch_size, votefuns, "resnet-bagging.txt", file_prefix="",random=False)

def test2():
    votefuns = [weighted_vote,  majority_vote]
    saved_model_files = ['bagging.0.final.hdf5', 'bagging.1.final.hdf5', 'bagging.2.final.hdf5']
    n_learners = len(saved_model_files)
    bagging_loading_model(n_learners, saved_model_files, votefuns, "resnet-bagging.txt")

def test3():
    n_learners = 3
    batch_size = 32
    epochs_lst = [5, 5, 5]
    version = 1
    n = 3
    votefuns = [weighted_vote,  majority_vote]
    bagging_train_model(version, n, n_learners, epochs_lst, batch_size, votefuns, "resnet-bagging.txt")

def meta_model(n_learners, num_classes):
    # create model
    model = Sequential()
    in_dim = n_learners * num_classes
    print(in_dim)
    model.add(Dense(n_learners*num_classes, input_dim = in_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def snapshot_train_model(version, n, epochs, batch_size, M, alpha_zero, name_prefix):
    num_classes = 10
    data_augmentation = True
    (x_train, y_train), (x_test, y_test) = load_data()
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    build_resnet_snapshot(M, alpha_zero, x_train, y_train, x_test, y_test, batch_size, epochs, n, version, data_augmentation, name_prefix)

def snapshot_ensemble(version, n, epochs, batch_size, M, alpha_zero, name_prefix, meta_epochs):
    snapshot_train_model(version, n, epochs, batch_size, M, alpha_zero, name_prefix)
    saved_model_files = []
    for i in range(M):
        saved_model_files.append("snapshot_models/%s-" % (name_prefix) + str(i+1) + ".h5")
    print(saved_model_files)
    stack_loading_model(saved_model_files, meta_epochs, filename="resnet-snapshot.txt")

def adaboost_test(version, n, batch_size):
    n_learners = 3
    epochs_lst = [70, 70, 70]
    sample_ratio = 3
    adaboost(version, n, n_learners, epochs_lst, batch_size, sample_ratio, "resnet-adaboost.txt")


def adaboost_original_test(version, n, batch_size):
    n_learners = 3
    epochs_lst = [70, 70, 70]
    sample_ratio = 3
    adaboost_original(version, n, n_learners, epochs_lst, batch_size, sample_ratio, "resnet-adaboost-original.txt")


def stack_test(version, n, batch_size):
    n_learners = 3
    epochs_lst = [200, 200, 200]
    meta_epochs = 200
    stack_train_model(version, n, n_learners, epochs_lst, batch_size, meta_epochs, filename="resnet-stack.txt")

def super_test(version, n, batch_size):
    n_learners = 3
    epochs_lst = [200, 200, 200]
    meta_epochs = 200
    stack_train_model_super(version, n, n_learners, epochs_lst, batch_size, meta_epochs, filename="resnet-super.txt")

def bagging_test(version, n, batch_size, random):
    n_learners = 3
    epochs_lst = [200, 200, 200]
    votefuns = [weighted_vote,  majority_vote]
    bagging_train_model(version, n, n_learners, epochs_lst, batch_size, votefuns, "resnet-bagging.txt", file_prefix="", random=random)


def snapshot_test(version, n, batch_size):
    epochs = 360
    M = 3
    alpha_zero = 1e-3*2
    name_prefix = "Model"
    meta_epochs = 200
    snapshot_ensemble(version, n, epochs, batch_size, M, alpha_zero, name_prefix, meta_epochs)

if __name__ == "__main__":
    version = 1 # 2
    n = 3 # 3, 5, 7, 9, 18
    batch_size = 32
    if len(sys.argv) != 2:
        print("Usage: python resnet_ensemble.py test_type")
        print("test type = 1: adaboost")
        print("test type = 2: stacking")
        print("test type = 3: bagging (original alg: random pick with replacement)")
        print("test type = 4: bagging (new alg: pick all)")
        print("test type = 5: snapshot")
        print("test type = 6: super learner")
        print("test type = 7: adaboost original test")
        sys.exit()
    test_type = int(sys.argv[1])
    if test_type == 1:
        adaboost_test(version, n, batch_size)
    elif test_type == 2:
        stack_test(version, n, batch_size)
    elif test_type == 3:
        bagging_test(version, n, batch_size, random=True)
    elif test_type == 4:
        bagging_test(version, n, batch_size, random=False)
    elif test_type == 5:
        snapshot_test(version, n, batch_size)
    elif test_type == 6:
        super_test(version, n, batch_size)
    elif test_type == 7:
        adaboost_original_test(version, n, batch_size)
    else:
        print("Unknown test type")
        sys.exit()