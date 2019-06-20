from os import listdir, mkdir
from os.path import isfile, isdir, join
from shutil import rmtree,copy2
import random, config

def cleanup(dir):
    rmtree(dir)
    mkdir(dir)

def split_data_set():
    print("Splitting Data")
    categories = [ file for file in listdir(config.dataFile) if isdir(join(config.dataFile,file))]
    cleanup(config.trainFile)
    cleanup(config.validationFile)
    for category in categories:
        mkdir(join(config.trainFile, category))
        mkdir(join(config.validationFile, category))
        images = [ file for file in listdir(join(config.dataFile, category)) if isfile(join(join(config.dataFile, category),file))]
        # print(images)
        random.shuffle(images)
        # print("after shuffle: ",images)
        noOfImages = len(images)
        trainDistr = int(noOfImages*config.splitRatio)
        trainImages = images[:trainDistr]
        validationImages = images[trainDistr:]
        for im in trainImages:
            copy2(join(join(config.dataFile,category),im), join(config.trainFile,category))
        for im in validationImages:
            copy2(join(join(config.dataFile,category),im), join(config.validationFile,category))

if __name__=='__main__':
    split_data_set()