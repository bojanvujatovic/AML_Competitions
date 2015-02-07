'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime
from csv import reader, DictReader
from math import exp, log, sqrt
from os import chdir
from copy import deepcopy
import multiprocessing


# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there

def main_loop(l1, l2, train, test, isTesting, holdoutSize):
    ##############################################################################
    # parameters #################################################################
    ##############################################################################

    # A, paths
    chdir('/Users/miljan/Documents/ML Master/Applied Machine Learning/Avazu Click-Through Prediction Rate/Data/new')
    submission = 'submission_new' + datetime.today().isoformat() + '.csv'  # path of to be outputted submission file

    # B, model
    alpha = .1  # learning rate
    beta = 1.   # smoothing parameter for adaptive learning rate
    L1 = l1     # L1 regularization, larger value means more regularized
    L2 = l2     # L2 regularization, larger value means more regularized

    # C, feature/hash trick
    with open(train, 'r') as t:
        rd = DictReader(t)
        D = int(rd.fieldnames[-1].split(':')[1])
    interaction = False     # whether to enable poly2 feature interactions

    # D, training/validation
    epoch = 1      # learn training data for N passes
    holdafter = None   # data after date N (exclusive) are used as validation
    holdout = holdoutSize  # use every N training instance for holdout validation


    ##############################################################################
    # class, function, generator definitions #####################################
    ##############################################################################

    class ftrl_proximal(object):
        ''' Our main algorithm: Follow the regularized leader - proximal

            In short,
            this is an adaptive-learning-rate sparse logistic-regression with
            efficient L1-L2-regularization

            Reference:
            http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
        '''

        def __init__(self, alpha, beta, L1, L2, D, interaction):
            # parameters
            self.alpha = alpha
            self.beta = beta
            self.L1 = L1
            self.L2 = L2

            # feature related parameters
            self.D = D
            self.interaction = interaction

            # model
            # n: squared sum of past gradients
            # z: weights
            # w: lazy weights
            self.n = [0.] * D
            self.z = [0.] * D
            self.w = {}

        def _indices(self, x):
            ''' A helper generator that yields the indices in x

                The purpose of this generator is to make the following
                code a bit cleaner when doing feature interaction.
            '''

            # first yield index of the bias term
            yield 0

            # then yield the normal indices
            for index in x:
                yield index

            # now yield interactions (if applicable)
            if self.interaction:
                D = self.D
                L = len(x)

                x = sorted(x)
                for i in xrange(L):
                    for j in xrange(i+1, L):
                        # one-hot encode interactions with hash trick
                        yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

        def predict(self, x):
            ''' Get probability estimation on x

                INPUT:
                    x: features

                OUTPUT:
                    probability of p(y = 1 | x; w)
            '''

            # parameters
            alpha = self.alpha
            beta = self.beta
            L1 = self.L1
            L2 = self.L2

            # model
            n = self.n
            z = self.z
            w = {}

            # wTx is the inner product of w and x
            wTx = 0.
            for i in self._indices(x):
                sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

                # build w on the fly using z and n, hence the name - lazy weights
                # we are doing this at prediction instead of update time is because
                # this allows us for not storing the complete w
                if sign * z[i] <= L1:
                    # w[i] vanishes due to L1 regularization
                    w[i] = 0.
                else:
                    # apply prediction time L1, L2 regularization to z and get w
                    w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

                wTx += w[i]

            # cache the current w for update stage
            self.w = w

            # bounded sigmoid function, this is the probability estimation
            return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

        def update(self, x, p, y):
            ''' Update model using x, p, y

                INPUT:
                    x: feature, a list of indices
                    p: click probability prediction of our model
                    y: answer

                MODIFIES:
                    self.n: increase by squared gradient
                    self.z: weights
            '''

            # parameter
            alpha = self.alpha

            # model
            n = self.n
            z = self.z
            w = self.w

            # gradient under logloss
            g = p - y

            # update z and n
            for i in self._indices(x):
                sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
                z[i] += g - sigma * w[i]
                n[i] += g * g


    def logloss(p, y):
        ''' FUNCTION: Bounded logloss

            INPUT:
                p: our prediction
                y: real answer

            OUTPUT:
                logarithmic loss of p given y
        '''

        p = max(min(p, 1. - 10e-15), 10e-15)
        return -log(p) if y == 1. else -log(1. - p)


    def data(path, D, trainFlag):
        ''' GENERATOR: Apply hash-trick to the original csv row
                       and for simplicity, we one-hot-encode everything

            INPUT:
                path: path to training or testing file
                D: the max index that we can hash to

            YIELDS:
                ID: id of the instance, mainly useless
                x: a list of hashed and one-hot-encoded 'indices'
                   we only need the index since all values are either 0 or 1
                y: y = 1 if we have a click, else we have y = 0
        '''

        rd = reader(open(path), delimiter=',')
        header = rd.next()

        for t, row in enumerate(rd):

            y = 0.
            ID = 'UNK'
            if trainFlag:
                y = float(row[0])
            else:
                ID = row[0]

            # build 
            x = []
            for key in row[1:]:
                # one-hot encode everything with hash trick
                index = abs(hash(key)) % D
                x.append(index)
            # x = map(int, row[1:])
            yield t, ID, x, y


    ##############################################################################
    # start training #############################################################
    ##############################################################################

    start = datetime.now()

    # initialize ourselves a learner
    learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

    # start training
    for e in xrange(epoch):
        loss = 0.
        count = 0

        for t, ID, x, y in data(train, D, True):  # data is a generator
            #    t: just a instance counter
            # date: you know what this is
            #   ID: id provided in original data
            #    x: features
            #    y: label (click)

            # step 1, get prediction from learner
            p = learner.predict(x)

            if (holdout and t % holdout == 0):
                # step 2-1, calculate validation loss
                #           we do not train with the validation data so that our
                #           validation loss is an accurate estimation
                #
                # holdafter: train instances from day 1 to day N
                #            validate with instances from day N + 1 and after
                #
                # holdout: validate with every N instance, train with others
                loss += logloss(p, y)
                count += 1
            else:
                # step 2-2, update learner with label (click) information
                learner.update(x, p, y)

        if isTesting:
            print('Epoch %d finished, validation logloss: %f, elapsed time: %s' % (
                e, loss/count, str(datetime.now() - start)))


    ##############################################################################
    # start testing, and build Kaggle's submission file ##########################
    ##############################################################################

    if isTesting:
        print('Starting testing')
        with open(submission, 'w') as outfile:
            outfile.write('id,click\n')
            for t, ID, x, y in data(test, D, False):
                p = learner.predict(x)
                outfile.write('%s,%s\n' % (ID, str(p)))
        return loss/count
        print('Done testing')

def greedy_feature_selection():
    l_features = []
    current_min = 1
    counter = 1
    while True:
        run_results = []
        print('Starting with %d feature(s) \n' % counter)
        for i in xrange(0, 23):
            print('Feature(s):')
            testing_features = deepcopy(l_features)
            testing_features.append(i)
            print(testing_features)
            res = main_loop(testing_features)
            run_results.append(res)
        minimum = min(run_results)
        # if there are improvements with some features, add them
        if minimum < current_min:
            current_min = minimum
            best_new = [i for i, j in enumerate(run_results) if j == minimum]
            print('\nBest %d . feature is: ' % counter)
            print(best_new)
            l_features += best_new
            print('Current best features are: ')
            print(l_features)
            counter += 1
            print('\n')
        # otherwise stop the selection of features
        else:
            break
    print('\nFeatures selected are:')
    print(l_features)


def select_features(d_row, l_features):
    l_row = d_row.items()
    results = []
    for i in l_features:
        results.append(l_row[i])
    return {v[0]: v[1] for v in results}


def select_l1_l2():
    x1 = 0
    x2 = 0.125
    y1 = 0
    y2 = 0.125

    incr = 3
    mainctr = 0
    while mainctr < 7:
        print('\n\nStarting iteration %d' % mainctr)
        print('\nCalculating l1')
        baseX2 = main_loop([2, 3, 9, 22, 1, 16, 0, 15, 20, 19, 8, 17, 7, 14, 13, 12, 10], x2, (y1 + y2) / 2.0)
        print baseX2
        ctr = 0
        while ctr < incr:
            print('\nCount is %d ' % ctr)
            c = (x1 + x2) / 2.0
            res = main_loop([2, 3, 9, 22, 1, 16, 0, 15, 20, 19, 8, 17, 7, 14, 13, 12, 10], c, (y1 + y2) / 2.0)
            print('Res is %f' % res)
            print('c is %f, x1 is %f, x2 is %f' % (c, x1, x2))
            if res < baseX2:
                x2 = c
                baseX2 = res
            else:
                x1 = c
            ctr += 1

        print('\nCalculating l2')
        baseY2 = main_loop([2, 3, 9, 22, 1, 16, 0, 15, 20, 19, 8, 17, 7, 14, 13, 12, 10], (x1+x2)/2.0, y2)
        print baseY2
        ctr = 0
        while ctr < incr:
            print('\nCount is %d ' % ctr)
            d = (y1 + y2) / 2.0
            res = main_loop([2, 3, 9, 22, 1, 16, 0, 15, 20, 19, 8, 17, 7, 14, 13, 12, 10], (x1 + x2) / 2.0, d)
            print('Res is %f' % res)
            print('d is %f, y1 is %f, y2 is %f' % (d, y1, y2))
            if res < baseY2:
                y2 = d
                baseY2 = res
            else:
                y1 = d
            ctr += 1
        if incr < 8:
            incr += 1
        mainctr += 1
        print c, d


if __name__ == '__main__':
    print('Training and validating!\n')
    # validate on 10% holdout
    p1 = multiprocessing.Process(target=main_loop, args=(0.01, 0.01, './train_svmlight_app', './test_svmlight_app', True, 10, ))
    p2 = multiprocessing.Process(target=main_loop, args=(0.01, 0.01, './train_svmlight_site', './test_svmlight_site', True, 10, ))
    p1.start()
    p2.start()

    # retrain on full, and create submission file
    # p3 = multiprocessing.Process(target=main_loop, args=(0.01, 0.01, './train_svmlight_app', './test_svmlight_app', True, 100000, ))
    # p4 = multiprocessing.Process(target=main_loop, args=(0.01, 0.01, './train_svmlight_site', './test_svmlight_site', True, 100000, ))
    # p3.start()
    # p4.start()


