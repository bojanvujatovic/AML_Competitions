from sklearn.linear_model import LogisticRegression
from csv import reader
from numpy import genfromtxt
from itertools import izip
from os import chdir

chdir('/mnt/new')

model = 'app' # 'site'

print '### BOOSTING ' + model + 'PREDICTIONS\n\n'

print 'Loading training data...'
train = genfromtxt('merged_train_preds_' + model + '_10000.csv', delimiter=',')
click = train[:, 0]
train = train[:, 1:]

print 'Training...'
reg = LogisticRegression()
reg.fit(train, click)

print 'Loading test data...'
test = genfromtxt('merged_submission_' + model + '_10000.csv', delimiter=',', skip_header=1)
ids = test[:, 0]
test = test[:, 1:]

print 'Predicting...'
pred = reg.predict(test)

print 'Writing submission file...'
with open('boosted_submission_' + model + '.csv', 'w') as out:
	out.write('id,click\n')
	for ID, pred in izip(ids, pred):
		out.write('%s,%s\n' % (ID, str(pred)))
