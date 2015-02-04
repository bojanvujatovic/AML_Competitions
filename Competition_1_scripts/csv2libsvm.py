from csv import DictReader
from datetime import datetime

train_csv_path = './train'
train_libsvm_path = train_csv_path + '_libsvm'
test_csv_path = './test'
test_libsvm_path = test_csv_path + '_libsvm'
test_libsvm_ids_path = test_csv_path + '_libsvm_ids'

# features = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id',
# 	'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type',
# 	'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

D = 2 ** 20

feats2remove = ['id', 'device_ip', 'device_id']

def encode(row, D):

    ID = row['id']

    # get label
    y = None
    if 'click' in row:
        if row['click'] == '1':
            y = 1
        else:
        	y = 0
        del row['click']

    # extract date
    date = int(row['hour'][4:6])

    # engineer features
    row['day'] = str(datetime(int(row['hour'][0:2]), int(row['hour'][2:4]), int(row['hour'][4:6])).weekday())
    row['hour'] = row['hour'][6:]

    # remove features
    for feature in feats2remove:
    	del row[feature]

    # build x
    x = []
    for key in row:
        value = row[key]

        # one-hot encode everything with hash trick
        index = abs(hash(key + '_' + value)) % D
        x.append(index)

    return ID, x, y


###############
# CONVERT
###############

def convert(csv_path, libsvm_path, libsvm_ids_path, test, max_elements=None):
	if test:
		print '\n\n##### CONVERTING TEST SET' 
	else:
		print '\n\n##### CONVERTING TRAIN SET' 
	ids = []
	with open(libsvm_path, 'w') as libsvm:
		for t, row in enumerate(DictReader(open(csv_path))):
			ID, x, y = encode(row, D)
			if y is not None:
				libsvm.write('%d' % y)
			else:
				libsvm.write('%d' % -1)
				ids.append(ID)

			for idx in sorted(x):
				libsvm.write(' %d:%d' % (idx, 1))
			libsvm.write('\n')
			if t % 500000 == 0:
				print t
			if max_elements and t == max_elements:
				break

	if test:
		with open(libsvm_ids_path, 'w') as libsvm_ids:
			libsvm_ids.write('id\n')
			for ID in ids:
				libsvm_ids.write('%s\n' % ID)

convert(train_csv_path, train_libsvm_path, None, False, 1000000)
convert(test_csv_path, test_libsvm_path, test_libsvm_ids_path, True)

print '\n\n\n'

