from math import exp, sqrt, log
from os import chdir
from csv import reader
from random import random
from datetime import datetime

chdir('/mnt/new')


model = 'app'
n = 60 if 'app' else 56
w = [0.0] * n
nEpochs = 5

def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

print '\n\nTRAINING...'
for epoch in xrange(1, nEpochs + 1):
	with open('merged_train_preds_%s.csv' % model, 'r') as train:
		r = reader(train)
		error = 0.0
		count = 0
		start = datetime.now()
		for t, sample in enumerate(r):
			y = float(sample[0])
			feats = map(float, sample[1:])
			# feats = map(float, sample[1:])
			# wTx = sum([w[i] * feats[i] for i in xrange(n)])
			wTx = 0.0
			for i in xrange(n):
				wTx += w[i] * feats[i]
			p = 1.0 / (1.0 + exp(-wTx))
			# w = [w[i] - (1 / sqrt(epoch)) * ((p - y) * w[i]) for i in xrange(n)]
			for i in xrange(n):
				w[i] -= (1 / sqrt(epoch)) * ((p - y) * feats[i])
			if random() < 1e-4:
				error += logloss(p, y)
				count += 1

			if t % 500000 == 0:
				print t
		print 'Epoch %d finished with error %f in %s.\n\n' % (epoch, error / count, str(datetime.now() - start))

print '\n\nTESTING...'
with open('merged_submission_%s.csv' % model, 'r') as test, open('boosted_submission_' + model + '.csv', 'w') as out:
	r = reader(test)
	r.next()
	out.write('id,click\n')
	for t, sample in enumerate(r):
		ID = sample[0]
		# feats = map(float, sample[1:])
		# wTx = sum([w[i] * feats[i] for i in xrange(n)])
		wTx = 0.0
		for i in xrange(n):
			wTx += w[i] * float(sample[i+1])
		p = 1.0 / (1.0 + exp(-wTx))
		p = 1.0 / (1.0 + exp(-wTx))
		out.write('%s,%f\n' % (ID, p))
		if t % 500000 == 0:
			print t
