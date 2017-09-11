import re
import glob, os

goodFiles = []

with open("training.xml", "r") as ins:
    for line in ins:
        line = line.strip('\n')

        tokens = re.findall('image file="([^"]*)"', line)

        if len(tokens) != 1:
            continue

        fileName = tokens[0]
        #print(fileName)
        goodFiles.append(fileName)

with open("testing.xml", "r") as ins:
    for line in ins:
        line = line.strip('\n')

        tokens = re.findall('image file="([^"]*)"', line)

        if len(tokens) != 1:
            continue

        fileName = tokens[0]
        #print(fileName)
        goodFiles.append(fileName)

print('----------')
test = './images/*'
r = glob.glob(test)
for i in r:
    if not i in goodFiles:
        print(i)
        os.remove(i)

print(len(goodFiles))