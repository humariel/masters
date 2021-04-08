import os
from shutil import copyfile
###########################################################################
########## MELANOMAS ARE 1, NON-MELANOMAS ARE 0 ###########################
###########################################################################

path = os.getcwd()
print(path)
DATASET_DIR = path + '/Datasets/Main'
TRAIN_DIR = DATASET_DIR + '/train'
TEST_DIR = DATASET_DIR + '/test'
GROUND_TRUTH = 'cond_pgan/4k_melanoma_4k_non_melanoma_csv/8526.csv'
SOURCE_DIR = 'cond_pgan/4k_melanoma_4k_non_melanoma/'

melanomas = []
non_melanomas = []
train_ammount = 0.8

with open(GROUND_TRUTH, 'r') as _file:
    # ignore header
    _file.readline()
    for line in _file:
        split = line.split(',')
        name = split[0]
        mel = int(split[1])
        if mel != 0:
            melanomas.append(name)
        elif mel == 0:
            non_melanomas.append(name)

    _file.close()

train_limit_melanoma = int(len(melanomas)*train_ammount)
train_limit_non_melanoma = int(len(non_melanomas)*train_ammount)

#copy melanomas
counter = 0
for name in melanomas:
    #copy to train folder
    if counter < train_limit_melanoma:
        copyfile(SOURCE_DIR+'/'+name, TRAIN_DIR+'/1/'+name)
    #copy to test folder
    else:
        copyfile(SOURCE_DIR+'/'+name, TEST_DIR+'/1/'+name)
    counter += 1

#copy non-melanomas
counter = 0
for name in non_melanomas:
    #copy to train folder
    if counter < train_limit_non_melanoma:
        copyfile(SOURCE_DIR+'/'+name, TRAIN_DIR+'/0/'+name)
    #copy to test folder
    else:
        copyfile(SOURCE_DIR+'/'+name, TEST_DIR+'/0/'+name)
    counter += 1

