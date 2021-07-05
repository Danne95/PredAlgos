                    #########################################################################
     ##   ##        #                                                                       #     ##   ##
    #   #   #       #       Final project of introduction to data mining course.            #    #   #   #
    #       #       #                                                                       #    #       #
     #     #        #       Names: Dan Nereusov, 316176866                                  #     #     #
       # #          #              Alex Lazarovich, 317508315                               #       # #
        #           #                                                                       #        #
                    #########################################################################

import sys
import Funcs
import pandas as pd
from datetime import datetime

def main():
    
    mainPath = sys.argv[0].replace('CLI.py','')
    args = sys.argv[1:]
    if len(args) == 0:
        noCommandFault()
    else:
        menu(args,mainPath)        

def menu(args,mainPath):
    args = list(map(lambda x:x.lower(),args))
    commandList = ['naive_bayes','id3','knn','kmeans','model']
    fileDict = {}
    if args[0] == 'help':
        if len(args) == 1:
            print('\nPossible commands are:\n')
            for command in commandList:
                print('---',command)
            print('\nEnter command name followed by help for possible commands for that command.')
        else:
            helpFault()
    else:
        if args[0] in commandList:
            if args[1] == 'help':
                helpMenu(args)
            elif len(args)>2:
                results = {}
                Tresults = {}
                model = None
                if args[1]!='help':
                    if args[0] in commandList and args[0] != 'model':
                        openFiles(args,fileDict)
                        if args[4] == 'our':
                            preprocessFiles(fileDict,mainPath)
                        else:
                            preprocessFiles(fileDict,mainPath,True)
                        model = modelAndClassify(args,fileDict,results,Tresults)
                    else:
                        Funcs.install_and_import('dill')
                        if 'plot' in args:
                            models = args[2:-1]
                        else:
                            models = args[2:]
                        for model in models:
                            openFiles(args,fileDict,False)
                            model = Funcs.dill.load(open(model, 'rb'))
                            fileDict['attributesDict'] = model['attrDict']
                            if model['implementation'] == 'sklearn':
                                preprocessFiles(fileDict,mainPath,True,True)
                            elif model['implementation'] == 'our': 
                                preprocessFiles(fileDict,mainPath,False,True)
                            outerModelClassify(args,fileDict,model,results,Tresults)
                    if 'save' in args and 'model' not in args:
                        model['attrDict'] = fileDict['attributesDict']
                        if 'our' in args:
                            model_save(model,mainPath,'our_'+args[0])
                        else:
                            model_save(model,mainPath,'sklearn_'+args[0])
                    if 'plot' in args:
                        if 'model' in args:
                            Funcs.comparing_plotter(results,results)
                        else:
                            Funcs.comparing_plotter(results,Tresults)
                for i in list(results.keys()):
                    print("Algorithem test: ",i," : ")
                    print(results[i])
                if Tresults != {}:
                    for i in list(Tresults.keys()):
                        print("Algorithem train: ",i," : ")
                        print(Tresults[i])
        else:
            print('\nUnknown command, enter help for possible commands')

    


def helpFault():
    print('\nHelp can\'t be followed by a command.')

def helpMenu(args):
    helpDict = {}
    helpDict['naive_bayes'] = '\n\nNaive bayes classifier format: naive_bayes testFilePath trainFilePath structFilePath sklearn/our save/plot**\n\n--- testFilePath - Enter the path of the file you want to test.\n--- trainFilePath - Enter the path of the file you want to build the model with.\n--- structFilePath - Enter the path of the file explaining the structure of the DB.\n--- sklearn/our - Enter one of these to choose which implementation is used for model building and classifying.\n--- save - optional entry, Enter save to save the model to a file to be used later.\n--- plot - optional entry, Enter plot to create a plot comparing the classification of the test file against the classification of the train file.\n\n** list of optional entries.'
    helpDict['id3'] = '\n\nId3 classifier format: id3 testFilePath trainFilePath structFilePath sklearn/our save/plot**\n\n--- testFilePath - Enter the path of the file you want to test.\n--- trainFilePath - Enter the path of the file you want to build the model with.\n--- structFilePath - Enter the path of the file explaining the structure of the DB.\n--- sklearn/our - Enter one of these to choose which implementation is used for model building and classifying.\n--- save - optional entry, Enter save to save the model to a file to be used later.\n--- plot - optional entry, Enter plot to create a plot comparing the classification of the test file against the classification of the train file.\n\n** list of optional entries.'
    helpDict['knn'] = '\n\nKNN classifier format: knn testFilePath trainFilePath structFilePath save/plot**\n\n--- testFilePath - Enter the path of the file you want to test.\n--- trainFilePath - Enter the path of the file you want to build the model with.\n--- structFilePath - Enter the path of the file explaining the structure of the DB.\n--- save - optional entry, Enter save to save the model to a file to be used later.\n--- plot - optional entry, Enter plot to create a plot comparing the classification of the test file against the classification of the train file.\n\n** list of optional entries.'
    helpDict['kmeans'] = '\n\nK-Means clustering format: kmeans testFilePath trainFilePath structFilePath save/plot**\n\n--- testFilePath - Enter the path of the file you want to test.\n--- trainFilePath - Enter the path of the file you want to build the model with.\n--- structFilePath - Enter the path of the file explaining the structure of the DB.\n--- save - optional entry, Enter save to save the model to a file to be used later.\n--- plot - optional entry, Enter plot to create a plot comparing the classification of the test file against the classification of the train file.\n\n** list of optional entries.'
    helpDict['model'] = '\n\nClassify/Cluster by outer model format: model testFilePath modelPath** plot**\n\n--- testFilePath - Enter the path of the file you want to test.\n--- modelPath - Enter the path of the model you want to classify by, could be more than 1 model for comparison.\n--- plot - optional entry, Enter plot to create a plot comparing the classification of the test file against other models, each plot shows classification by 1 model.'
    if len(args) == 1:
        noCommandFault()
    elif len(args) > 2:
        helpFault()
    else:
        print(helpDict[args[0]])

def noCommandFault():
    print('\nYou haven\'t entered any commands, enter help for possible commands.')

def modelAndClassify(args,fileDict,results,Tresults):
    classifyDict = {}
    classifyDict['naive_bayes'] = {}
    classifyDict['naive_bayes']['our'] = Funcs.Naive_bayes_classify
    classifyDict['naive_bayes']['sklearn'] = Funcs.sklearn_NB_classify
    classifyDict['id3'] = {}
    classifyDict['id3']['our'] = Funcs.ID3_classify
    classifyDict['id3']['sklearn'] = Funcs.sklearn_ID3_classify
    classifyDict['knn'] = Funcs.KNN_classify
    classifyDict['kmeans'] = Funcs.KMeans_classify
    model = classifyDict[args[0]]
    if not callable(model):
        model = model[args[4]]
    m = model(results,fileDict)
    if 'plot' in args:
        model(Tresults,fileDict,None,fileDict['trainFile'])
    return m

def outerModelClassify(args,fileDict,oM,results,Tresults):
    classifyDict = {}
    classifyDict['naive_bayes'] = {}
    classifyDict['naive_bayes']['our'] = Funcs.Naive_bayes_classify
    classifyDict['naive_bayes']['sklearn'] = Funcs.sklearn_NB_classify
    classifyDict['id3'] = {}
    classifyDict['id3']['our'] = Funcs.ID3_classify
    classifyDict['id3']['sklearn'] = Funcs.sklearn_ID3_classify
    classifyDict['knn'] = Funcs.KNN_classify
    classifyDict['kmeans'] = Funcs.KMeans_classify
    model = classifyDict[oM['type']]
    if not callable(model):
        model = model[oM['implementation']]
    m = model(results,fileDict,oM)
    """if 'plot' in args:
        model(Tresults,fileDict,oM,fileDict['trainFile'])"""
        

def openFiles(args,fileDict,newModel = True):
    try:
        fileDict['testFile'] = pd.read_csv(args[1])
    except:
        print('\nTest file couldn\'t be found.')
        sys.exit()
    if newModel:
        try:
            fileDict['trainFile'] = pd.read_csv(args[2])
        except:
            print('\nTrain file couldn\'t be found.')
            sys.exit()
        try:
            fileDict['structFile'] = open(args[3])
        except:
            print('\nStruct file couldn\'t be found.')
            sys.exit()
        fileDict['attributesDict'] = Funcs.InputToDict(fileDict['structFile'])

def preprocessFiles(fileDict,mainPath,sklearn = None,model = None):
    if sklearn:
        fileDict['testFile'] = Funcs.sklearn_preprocessing(fileDict['testFile'].copy(deep=True),fileDict['attributesDict'])
        if model == None:
            fileDict['trainFile'] = Funcs.sklearn_preprocessing(fileDict['trainFile'].copy(deep=True),fileDict['attributesDict'])
    else:
        fileDict['testFile'] = Funcs.our_preprocessing(fileDict['testFile'].copy(deep=True),fileDict['attributesDict'])
        #fileDict['testFile'] = Funcs.numericBinning(fileDict['testFile'].copy(deep=True),fileDict['attributesDict'])
        if model == None:
            fileDict['trainFile'] = Funcs.our_preprocessing(fileDict['trainFile'].copy(deep=True),fileDict['attributesDict'])
            fileDict['trainFile'] = Funcs.numericBinning(fileDict['trainFile'].copy(deep=True),fileDict['attributesDict'])
    date_time = datetime.now().strftime("_%H%M%S_%m%d%y")
    fileDict['testFile'].to_csv(mainPath+'test_clean'+date_time+'.csv', encoding='utf-8')
    if model == None:
        fileDict['trainFile'].to_csv(mainPath+'train_clean'+date_time+'.csv', encoding='utf-8')

def model_save(model,mainPath,mType):
    if model != None:
        date_time = datetime.now().strftime("_%H%M%S_%m%d%y")
        filename = mainPath+mType+'_model'+date_time+'.sav'
        Funcs.install_and_import('dill')
        Funcs.dill.dump(model , open(filename, 'wb'))
        print('\nModel saved to: '+filename+'\n')

main()
