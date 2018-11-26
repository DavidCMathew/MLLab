traindata=[]
with open('data/enjoysport.csv') as f:
    for line in f:
        traindata.append(line.strip().split(','))

rows = len(traindata)
cols = len(traindata[0])

h = ['0'] * cols

for i in range(1, rows):
    t = traindata[i]
    if t[cols-1] == '0':
        print('Negative example ignored')
        continue
    for y in range(0, cols-1):
        if h[y] == t[y]:
            continue
        h[y] = t[y] if h[y] == '0' else '?'
    print(h)

print('The maximally specific set is ')
print(str(h[:-1]).replace('[', '<').replace(']', '>'))
