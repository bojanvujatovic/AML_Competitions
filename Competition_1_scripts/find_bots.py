from csv import DictReader
from datetime import datetime

train_path = '/Users/hmourit/Downloads/train'
test_path = '/Users/hmourit/Downloads/test'
bots_path = '/Users/hmourit/Downloads/bots'
click_bound = 1000


start = datetime.now()

ips = {}
for t, row in enumerate(DictReader(open(train_path))):
    if row['device_ip'] not in ips:
        ips[row['device_ip']] = [0, 0]
    ips[row['device_ip']][int(row['click'])] += 1
    if t % 500000 == 0:
        print t

bots = set([])
count = 0
with open(bots_path, 'w') as outfile:
    outfile.write('ip,click_rate\n')
    for ip in ips:
    	if ips[ip][1] > click_bound:
    		bots.add(ip)
    		count += (ips[ip][0] + ips[ip][1])
    		click_rate = 1. * ips[ip][1] / (ips[ip][0] + ips[ip][1])
        	outfile.write('%s,%s\n' % (ip, str(click_rate)))

n_bots = len(bots)
print '\n\n', n_bots, 'bots in train set that appear', count, 'times.'

count = 0
bots_in_test = set([])
for row in DictReader(open(test_path)):
	if row['device_ip'] in bots:
		bots_in_test.add(row['device_ip'])
		count += 1

print len(bots_in_test), 'appear in test set', count, 'times.'

print '\nElapsed time:', str(datetime.now() - start)
