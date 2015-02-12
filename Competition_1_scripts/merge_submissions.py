from csv import reader
from itertools import izip
from os import chdir

chdir('/mnt/new')

model = 'app'

hashes = ['2007672721904746826', '-6553929249067973233', '5449845956042923340', '-8435269146325907594', '9011457927234643708', '-6992349277918252699', '4449861956167923545', '3889030619434681955', '1446084750891026729', '-3111004380614318185', '-3996849117578628345', '9011466927363644012', '-6553920248927972780', '1449844956139923532', '920087729268517', '-8553932248936972878', '2449852956164923435', '-3996851117566628300', '-7553931249059973227', '8011452927237643717', '446077750878026852', '889026619423681890', '-8553939249068973074', '4007687721936746842', '-3111005380620318178', '1889011619308681698', '-1435253146514908250', '-3996855117566628120', '5734247508079759663', '5734247508079759663', '-7553930248944972891', '3007684722044747039', '-3111004380613318064', '446086750866026737', '1889014619415682002', '1889008619292681693', '-8553937249069973228', '1006914087610268371', '3572248178088395063', '1889009619302681703', '3007678721920746832', '4449870956315923762', '-992316277860252773', '5007697722070747045', '8892793824618578559', '2007676721926746859', '8572390178771395975', '-7435274146444908003', '3006908087624268531', '6892787824603578383', '-1110980380464317895', '-6992350277925252818', '-5435278146558908268', '-1110984380589318235', '2449855956277923734', '2449856956283923756', '7690722158747361', '-4996856117572628292', '4007683721916746826', '3007684722042747036', '-2435260146532908266', '-6992335277900252892', '4889035619464681919', '-1110994380615318224', '6892785824598578506', '2449843956141923582', '-110977380473317801', '-7553929248947972773', '2007678722031747163', '889026619423681890', '-6992348277913252806', '-5553945249189973309', '7892787824609578427', '8572395178776395946', '-5553938249053973087', '446077750883026745', '1007690722041747052', '9011467927373644021', '8892797824619578396', '-6435264146450907959', '-2435248146396907828', '-8553938249068973080', '-5427790821877604708', '6007708722083747142', '-4435263146434907992', '6572384178762395948', '6572377178754396120', '8572395178765396036', '8572393178763396053']

files = []
for h in hashes:
	files.append(open('submission_' + h + '.csv', 'r'))

readers = map(reader, files)
with open('merged_submission_' + model + '.csv', 'w') as out:
    for lines in izip(*readers):
        sample = [str(lines[0][1])]
        sample += [str(l[0]) for l in lines]
        out.write('%s\n' % ','.join(sample))

for f in files:
    f.close()