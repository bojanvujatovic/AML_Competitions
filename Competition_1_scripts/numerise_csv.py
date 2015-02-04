from csv import DictReader
from datetime import datetime

TRAIN_PATH = './train'
TEST_PATH = './test'
FEATS2REMOVE = set(['id', 'click', 'device_ip', 'device_id'])

train_num_path = TRAIN_PATH + '_num'
test_num_path = TEST_PATH + '_num'

convertion = {}

def convert(orig_path, num_path):
	print '\n\n#### CONVERTING', orig_path
	with open(num_path, 'w') as num:
		reader = DictReader(open(orig_path))
		features = list(reader.fieldnames)
		features.remove('id')
		num.write('%s\n' % ','.join(features))
		for t, row in enumerate(reader):

			sample = []
			for key in features:
				if key not in convertion:
					convertion[key] = {}
				if row[key] not in convertion[key]:
					convertion[key][row[key]] = str(len(convertion[key]))
				sample.append(convertion[key][row[key]])
			num.write('%s\n' % ','.join(sample))

			if t % 500000 == 0:
				print t

convert(TRAIN_PATH, train_num_path)
convert(TEST_PATH, test_num_path)

print '\n\n\n'
