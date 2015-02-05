from csv import DictReader
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import *
from pprint import pprint

TRAIN_PATH = './train'
TEST_PATH = './test'
FEATS2REMOVE = set(['id', 'click', 'hour', 'device_ip', 'device_id', 'device_model', 'C15', 'C16'])
NEWFEATURES = ['day', 'hour_country', 'C15_16']
MIN_FREQ = 5

train_num_path = TRAIN_PATH + '_num'
test_num_path = TEST_PATH + '_num'

convertion = {}
freqs = {}
null_values = {
	'site_id': '85f751fd',
	'site_domain': 'c4e18dd6',
	'site_category': '50e219e0',
	'app_id': 'ecad2386',
	'app_domain': '7801e8d9',
	'app_category': '07d7df22'
}

def transformrow(row):
	# feature engineering
	row['day'] = str(datetime(int(row['hour'][0:2]), int(row['hour'][2:4]), int(row['hour'][4:6])).weekday())
	row['hour_country'] = str(int(row['hour'][6:]) / 8) + '_' + row['C20']
	row['C15_16'] = row['C15'] + '_' + row['C16']
	return row

def initialise_reader(orig_path):
	reader = DictReader(open(orig_path))
	features = list(reader.fieldnames)
	for feat in FEATS2REMOVE:
		if feat in features:
			features.remove(feat)
	for feat in NEWFEATURES:
		features.append(feat)
	return reader, features

def encode(orig_path, num_path, train):
	with open(num_path, 'w') as num:
		# compute frequencies
		if train:
			print '\n\n#### COUNTING FREQUENCIES'
			reader, features = initialise_reader(orig_path)
			for feat in features:
				freqs[feat] = {}
			for t, row in enumerate(reader):
				row = transformrow(row)
				for feat in features:
					if row[feat] not in freqs[feat]:
						freqs[feat][row[feat]] = 0
					freqs[feat][row[feat]] += 1
				if t % 500000 == 0:
					print t

		print '\n\n#### CONVERTING', orig_path
		reader, features = initialise_reader(orig_path)
		for feat in features:
			convertion[feat] = {'UNK': 0}
		y = []
		X = []
		# num.write('%s\n' % ','.join(features))
		for t, row in enumerate(reader):
			# label
			if 'click' in row:
				y.append(int(row['click']))
				# num.write('%s,' % row['click'])
			else:
				y.append(-1)
				# num.write('%s,' % str(-1))

			row = transformrow(row)

			sample = []
			for key in features:
				if train:
					if freqs[key][row[key]] > MIN_FREQ \
						and (key not in null_values or row[key] != null_values[key]):
						if row[key] not in convertion[key]:
							convertion[key][row[key]] = len(convertion[key])
						sample.append(convertion[key][row[key]])
					else:
						sample.append(convertion[key]['UNK'])
				else:
					if row[key] in convertion[key] and freqs[key][row[key]] > MIN_FREQ \
						and (key not in null_values or row[key] != null_values[key]):
						sample.append(convertion[key][row[key]])
					else:
						sample.append(convertion[key]['UNK'])
			X.append(sample)
			# num.write('%s\n' % ','.join(sample))

			if t % 500000 == 0:
				print t
	return X, y	

train, click = encode(TRAIN_PATH, train_num_path, train=True)
enc = OneHotEncoder()
train.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
train = enc.fit_transform(train)
train = train[:-1]
dump_svmlight_file(train, click, TRAIN_PATH + '_svmlight')

test, foo = encode(TEST_PATH, test_num_path, train=False)
test = enc.transform(test)
dump_svmlight_file(test, foo, TEST_PATH + '_svmlight')

print '\n\n\n'