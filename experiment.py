import splitData            ############
import classification
import confusionMatrix
import indexing
import search

if __name__=='__main__':
    splitData.split_data_set()           #############
    classification.train_model()
    # confusionMatrix.get_confusion_matrix()
    # indexing.index_files()
    # search.search_sample()