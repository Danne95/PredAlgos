                    #########################################################################
     ##   ##        #                                                                       #     ##   ##
    #   #   #       #       Final project of introduction to data mining course.            #    #   #   #
    #       #       #                                                                       #    #       #
     #     #        #       Names: Dan Nereusov, 316176866                                  #     #     #
       # #          #              Alex Lazarovich, 317508315                               #       # #
        #           #                                                                       #        #
                    #########################################################################


def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        if hasattr(pip, 'main'):
            pip.main(['install', package])
        else:
            pip._internal.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

# Imports
import math
install_and_import('pandas')
import pandas as pd
install_and_import('numpy')
import numpy as np
install_and_import('sklearn')
install_and_import('info_gain')
from info_gain import info_gain
install_and_import('pyitlib')
from pyitlib import discrete_random_variable as drv
from sklearn.metrics import confusion_matrix
    
def InputToDict(data):
    #function creates a dictionary of attributes from structure file
    attributes = {}
    for i in data:
        attr = i.split()[1]
        x = i.split()[2]
        if i.split()[2] == 'NUMERIC':
            field = x
        else:
            field = x.replace('{','').replace('}','').split(',')
        attributes[attr]=field
    return attributes


# In[54]:


def numericBinning(data,attributes,NumOfBins = 22):
    #function descretizies values if necessary
    if NumOfBins is None:
        NumOfBins = (int)(math.log(data.shape[0],5))
    for i in data:
        if attributes[i] == 'NUMERIC':
            data[i] = pd.cut(data[i],NumOfBins)
            attributes[i] = data[i].unique().tolist()
            minAttr = min(attributes[i])
            maxAttr = max(attributes[i])
            newMinAttr = pd.Interval(-float('inf'),min(attributes[i]).right)
            newMaxAttr = pd.Interval(max(attributes[i]).left,float('inf'))
            attributes[i].remove(minAttr)
            attributes[i].remove(maxAttr)
            attributes[i].append(newMinAttr)
            attributes[i].append(newMaxAttr)
            data[i] = data[i].replace(to_replace=minAttr,value=newMinAttr)
            data[i] = data[i].replace(to_replace=maxAttr,value=newMaxAttr)
    return data


# In[55]:


def probTable(trainFile,attributes):
    #function creates a table with all attributes and all class values
    classVals = dictHelper(trainFile,attributes,'class')
    allprobs = {}
    for x in attributes:
        if(x != 'class'):
            allprobs[x] = probsCalc(trainFile,attributes,x,classVals)
    allprobs['class'] = {}
    for val in classVals:
        allprobs['class'][val] = classVals[val]
            
    return allprobs


# In[56]:


def probsCalc(data,attributes,x,classVals):
    #function creates a dictionary with 2 probabilities (yes and no), for each value of an attribute, 
    xVals = dictHelper(data,attributes,x)
    subDict = {}
    for i in xVals:
        subDict[i] = {}
        for j in classVals:
            (subDict[i])[j] = len(data[x].loc[(data[x] == i) & (data['class'] == j)])/len(data[x].loc[data['class'] == j]) #|Sunny|/|yes|,|yes/Sunny|/|Sunny|      
    return subDict


# In[57]:


def dictHelper(data,attributes,x):
    #function creates a dictionary with all all possible values as keys and counts the number of each value apearances
    valuesDict = {}
    for val in attributes[x]:
        valuesDict[val] = len(data[x].loc[data[x] == val])#.size     
    return valuesDict


# In[58]:


def bestIGattr(data,attributes,toSplit=False):
    #function finds the atrribute with the best Info Gain 
    classEntropy = drv.entropy(data['class']).item(0)
    attrsIG = {}
    for attr in attributes:
        if toSplit:
            attrsIG[attr] = (info_gain.info_gain(data[attr],data['class']))/splitInfo(data[attr],attr)
        else:
            attrsIG[attr] = (info_gain.info_gain(data[attr],data['class']))
    maxGain = max(attrsIG.values())
    if maxGain == 0:
        return 'IG0'
    for attr in attrsIG:
        if attrsIG[attr] == maxGain:
            return attr

# In[59]:


def splitInfo(data,attrName):
    #??????????????
    uniqueValues = data.unique().tolist()
    tempDF = pd.DataFrame(data)
    i=1
    for val in uniqueValues:
        tempDF = tempDF.replace(to_replace=val,value=i)
        i+=1  
    return drv.entropy(tempDF[attrName].to_list())


# In[60]:


def ID3_tree(classDict, data, attrDict, attributes, toSplit = False,numNodes = 100):
    #function creates a decision tree recursivly with dictionaries
    if 'class' in attributes:
        attributes.remove('class')
    if(len(data['class'])<=numNodes and len(data['class'])>0):
        return data['class'].mode().iloc[0]
    else:
        bestOp = bestIGattr(data,attributes,toSplit)
        if bestOp == 'IG0':
            return data['class'].mode().iloc[0]
        classDict[bestOp]={}
        for val in attrDict[bestOp]:
            if(len(data.loc[data[bestOp] == val]) > 0 and len(attributes) > 0 ):
                newAttrs = attributes.copy()
                newAttrs.remove(bestOp)
                classDict[bestOp][val] = ID3_tree({},data.loc[data[bestOp] == val],attrDict,newAttrs)
        return classDict


# In[61]:

def our_preprocessing(data,attributes):
    data = data.dropna(subset=['class'])
    for i in data:
        if attributes[i] == 'NUMERIC':
            data[i] = data[i].fillna(data[i].mean())
        elif i != 'class':
            data[i] = data[i].fillna(data[i].mode().iloc[0])           
    return data

def sklearn_preprocessing(data,attributes):
    #function preprocesses the data to fit sklearn functions
        #for numeric normalize all values to a min-max normalization
        #for categorial values convert values to numbers (1..2..3..4..)
    data = data.dropna(subset=['class'])
    for i in data:
        if attributes[i] == 'NUMERIC':
            data[i] = data[i].fillna(data[i].mean())
            minimum = min(data[i])
            maximum = max(data[i])
            data[i] = (data[i]-minimum)/(maximum-minimum)
            
        elif i != 'class':
            data[i] = data[i].fillna(data[i].mode().iloc[0])
            if len(attributes[i]) == 2:
                data[i] = data[i].apply(lambda x:attributes[i].index(x))
            #categorial
            else:
                #data = data.drop(i,axis=1)
                data[i] = data[i].apply(lambda x:attributes[i].index(x.lower()))            
    return data


# In[62]:


def Naive_bayes_classify(results,fileDict,model = None,data = None):
    if data is None:
        data = fileDict['testFile']
    if model is None:
        attributes = fileDict['attributesDict']
        probDict = probTable(fileDict['trainFile'], fileDict['attributesDict'])
        structFile = fileDict['structFile']
    else:
        attributes = model['attributes']
        probDict = model['probDict']
        structFile = model['structFile']
    #naive bayes calculation
    TP=TN=FP=FN=0.001
    for i,row in data.iterrows():
        yesV = 1
        noV = 1
        for x in attributes:
            if x != 'class': 
                if row[x] in attributes[x]:
                    yesV *=probDict[x][row[x]]['yes']
                    noV  *=probDict[x][row[x]]['no']
                else:
                    for bins in attributes[x]: 
                        if row[x] in bins:
                            yesV *=probDict[x][bins]['yes']
                            noV  *=probDict[x][bins]['no']

        yesV *=probDict['class']['yes']
        noV *=probDict['class']['no']
        if yesV >= noV :
            if row['class'] == 'yes':
                TP+=1
            else:
                FP+=1    
        else:
            if row['class'] == 'no':
                TN+=1
            else:
                FN+=1
    resultsDict(results,'our_Naive_bayes',TP,FN,FP,TN)
    return {'probDict' :probDict, 'attributes': attributes, 'structFile':structFile,'type':'naive_bayes','implementation':'our'}


# In[63]:


def sklearn_NB_classify(results,fileDict,model=None,data = None):
    if data is None:
        data = fileDict['testFile']
    from sklearn.naive_bayes import GaussianNB
    #naive bayes using sklearn
    if model is None:
        trainData = fileDict['trainFile']
        k = (int)(math.log(data.shape[0],4))
        model = GaussianNB()
        model.fit(trainData.drop('class',axis=1),trainData['class'])
    else:
        model = model['model']
    ans = model.predict(data.drop('class',axis=1))
    TN, FP, FN, TP =(confusion_matrix(data['class'], ans)).ravel()
    resultsDict(results,'sklearn_Naive_bayes',TP,FN,FP,TN)
    return {'model':model,'type':'naive_bayes','implementation':'sklearn'}


# In[64]:


def ID3_classify(results,fileDict,model=None,data = None):
    if data is None:
        data = fileDict['testFile']
    if model is None:
        treeDict = ID3_tree({},fileDict['trainFile'],fileDict['attributesDict'],list(fileDict['attributesDict'].keys()))
    else:
        treeDict = model['treeDict']
    #ID3 decision tree calculation
    TP=TN=FP=FN=0.001
    for index,row in data.iterrows():
        currentDict = treeDict.copy()
        while type(currentDict)==dict:
            currentRoot = list(currentDict.keys())[0]
            if row[currentRoot] in currentDict[currentRoot].keys():
                val = row[currentRoot]
            else:
                for i in currentDict[currentRoot]:
                    if row[currentRoot] in i:
                        val = i
                        break
            try:
                currentDict = currentDict[currentRoot][val]
            except:
                try:
                    dis=float("inf")
                    for i in currentDict[currentRoot]:
                        disToMin = abs(i.left - row[currentRoot])
                        disToMax = abs(i.right - row[currentRoot])
                        if(min(disToMin,disToMax)<dis):
                            dis = min(disToMin,disToMax)
                            minInterval = i
                    currentDict = currentDict[currentRoot][minInterval]
                except:
                    currentDict = currentDict[currentRoot][list(currentDict[currentRoot].keys())[0]]
        if row['class'] == 'yes':
            if currentDict == 'yes':
                TP+=1
            else:
                FN+=1
        else:
            if currentDict == 'yes':
                FP+=1
            else:
                TN+=1
    resultsDict(results,'our_ID3',TP,FN,FP,TN)
    return {'treeDict':treeDict,'type':'id3','implementation':'our'}


# In[65]:


def sklearn_ID3_classify(results,fileDict,model = None,data = None,depth=4,nodes=700):    #it is possible to display the tree ! if we want.
    if data is None:
        data = fileDict['testFile']
    from sklearn.tree import DecisionTreeClassifier
    #ID3 decision tree using sklearn
    if model is None:
        trainData = fileDict['trainFile']
        k = (int)(math.log(data.shape[0],5))
        clf = DecisionTreeClassifier(criterion="entropy",max_depth=depth,min_samples_split=nodes)
        """,min_samples_split=nodes"""
        clf = clf.fit(trainData.drop('class',axis=1),trainData['class'])
    else:
        clf = model['model']
    ans = clf.predict(data.drop('class',axis=1))
    TN, FP, FN, TP =(confusion_matrix(data['class'], ans)).ravel()
    resultsDict(results,'sklearn_ID3',TP,FN,FP,TN)
    return {'model':clf,'type':'id3','implementation':'sklearn'}


# In[66]:


def KNN_classify(results,fileDict,model = None,data = None,k=1):
    if data is None:
        data = fileDict['testFile']
    from sklearn.neighbors import KNeighborsClassifier
    #KNN using sklearn
    if model is None:
        trainData = fileDict['trainFile']
        #k= (int)(math.log(data.shape[0],5))
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(trainData.drop('class',axis=1),trainData['class'])
    else:
        knn = model['model']
    ans = knn.predict(data.drop('class',axis=1))
    TN, FP, FN, TP =(confusion_matrix(data['class'], ans)).ravel()
    resultsDict(results,'sklearn_KNN',TP,FN,FP,TN)
    return {'model':knn,'type':'knn','implementation':'sklearn'}


# In[67]:


def KMeans_classify(results,fileDict,model = None,data = None,k=20):
    if data is None:
        data = fileDict['testFile']
    from sklearn.cluster import KMeans
    #KMeans using sklearn
    #k= (int)(math.log(data.shape[0],4))
    TP=TN=FP=FN=0.001
    if model is None:
        trainData = fileDict['trainFile']
        kmeans = KMeans(n_clusters=k,random_state=0)
        trainPredict = kmeans.fit_predict(trainData.drop('class',axis=1))
        trainPredict = pd.DataFrame(trainPredict)
        cluster_majors = {}
        for i in range(k):
            cluster_majors[i] = trainData.loc[trainPredict.loc[trainPredict[0]==i].index,'class'].mode().iloc[0]
    else:
        kmeans = model['model']
        cluster_majors = model['clusters']
    ans = kmeans.predict(data.drop('class',axis=1))
    for index,row in data.iterrows():
        if row['class'] == 'yes':
            if cluster_majors[ans[index]] == 'yes':
                TP+=1
            else:
                FN+=1
        else:
            if cluster_majors[ans[index]] == 'yes':
                FP+=1
            else:
                TN+=1
    resultsDict(results,'sklearn_KMeans',TP,FN,FP,TN)
    return {'model':kmeans,'clusters':cluster_majors,'type':'kmeans','implementation':'sklearn'}


# In[68]:

def resultsDict(results,funcName,TP,FN,FP,TN):
    results[funcName] = {}
    results[funcName]['accuracy'] = round((TP+TN)/(TP+TN+FP+FN)*100,2)
    results[funcName]['precision'] = round((TP)/(TP+FP)*100,2)
    results[funcName]['recall'] = round((TP)/(TP+FN)*100,2)
    results[funcName]['FMeasure'] = round((2*TP)/(2*TP+FP+FN)*100,2)


def comparing_plotter(onTest,onTrain):
    install_and_import('bokeh')
    from bokeh.io import output_file, show
    from bokeh.models import ColumnDataSource, FactorRange
    from bokeh.plotting import figure
    from bokeh.transform import factor_cmap
    from bokeh.palettes import Spectral6
    from bokeh.resources import CDN
    from bokeh.embed import file_html
    from bokeh.layouts import column
    #generates a plot comparing the two runs on testfile and trainfile
    plots = []
    for name in onTest.keys():
        x = [(val,algo) for val in list(onTest[name].keys()) for algo in ["data= test","data= train"]]
        counts = sum(zip(list(onTest[name].values()),
                         list(onTrain[name].values())),())
        source = ColumnDataSource(data=dict(x=x, counts=counts))
        p = figure(x_range=FactorRange(*x), plot_height=404, title=name+" test file vs train file",
                   toolbar_location=None, tools="")
        p.vbar(x='x', top='counts', width=0.9, source=source,
               fill_color=factor_cmap('x', palette=Spectral6, factors=list(["data= test","data= train"]), start=1, end=2))
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1
        p.axis.minor_tick_line_color = None
        p.yaxis.axis_label = "Percentages"
        p.outline_line_color = None
        p.xgrid.grid_line_color = None
        plots.append(p)
    r = column(plots)
    show(r)
    
    return

"""

#--------------------------------------
# accuracy = how good is the algorithem at predicting positive and negative answers
# precision = Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
# recall = Recall is the ratio of correctly predicted positive observations to the all observations in actual class.
# F=Measure = weighted average of Precision and Recall.
    #especially if you have an uneven class distribution.
    """


