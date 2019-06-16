import classification
from os import listdir, mkdir
from os.path import isfile, isdir, join, abspath
from prediction import predict 
import config
import json

def get_confusion_matrix():
    categories = [ file for file in listdir(config.validationFile) if isdir(join(config.validationFile,file))]
    # confMatRow = {cat,0 for cat in categories}
    confMat = {cat:{categ:0 for categ in categories} for cat in categories}
    cm_file = open(config.confMatFile,"w")
    cm_file.write(",".join(["    "]+categories))
    cm_file.write('\n')
    for cat in categories:
        catFile = join(config.validationFile, cat)
        images = [ abspath(join(catFile,file)) for file in listdir(catFile) if isfile(join(catFile,file))]
        for image in images:
            label = predict(image)
            confMat[cat][label] += 1
        print("ConfMat: ",confMat[cat])
        cm_file.write(",".join([str(nm) for nm in list(confMat[cat].values())]))
        cm_file.write('\n')

    # json.dump(confMat,open(config.confMatFile,"w"))

if __name__=='__main__':
    get_confusion_matrix()