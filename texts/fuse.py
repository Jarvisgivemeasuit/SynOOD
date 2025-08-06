files = ['neg_labels_for_gener_ood_part1.txt',
         'neg_labels_for_gener_ood_part2.txt',]

res = []
for file in files:
    with open(file, 'r') as f:
        for line in f:
            res.append(line)

res = list(set(res))
with open('neg_labels_for_gener_ood_all.txt', 'w') as f:
    for line in res:
        f.write(line)
