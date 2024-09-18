# Databricks notebook source
# MAGIC %md
# MAGIC ### ----------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC # Group 10 Members:-
# MAGIC
# MAGIC ## 1. Parthib Ray     (B040)
# MAGIC ## 2. Mili            (B031)
# MAGIC ## 3. Khushi Garg     (B024)
# MAGIC ## 4. Nandan Jindal   (B034)
# MAGIC ## 5. Khushi Gupta    (B022)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ----------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## * The Dataset used by this group for this project is based On Obesity Data from worldwide to understand the weight behaviours of people
# MAGIC
# MAGIC ## The link of the dataset is as follows:-
# MAGIC
# MAGIC https://www.kaggle.com/code/mpwolke/obesity-levels-life-style/input
# MAGIC
# MAGIC ## * We acknowledge the hard work taken by the original source for the extensive data for classification and data analysis operations.
# MAGIC
# MAGIC ## * The work produced using this dataset is not for commercial purpose and only for academic and research purposes
# MAGIC
# MAGIC ## * The free usage of the work is allowed by the owners of this work

# COMMAND ----------

# MAGIC %md
# MAGIC ### ----------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the Data

# COMMAND ----------

# imports 
print("\n*** Imports ***")
# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# numpy
import numpy as np              # for array operations
print("Done ...")

# COMMAND ----------

import pandas as pd

# COMMAND ----------

############################################
#### start eda functions 
############################################

from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as f
from pyspark.sql.functions import min, mean, median, max

# columns
def getColNames(df):
    dsRetVals = pd.Series() 
    # loop through each column
    for colName in df.columns:
        dsRetVals[colName] = ""
    return (dsRetVals)
	
# data types
def getDataTypes(df):
    # pandas series
    dsRetVals = pd.Series()
    # loop for key value of each column
    for key,val in df.dtypes:
        dsRetVals[key] = val
    return (dsRetVals)

# string cols
def getStringCols(df):
    # pandas series
    dsRetVals = pd.Series()
    # loop for key value of each column
    for key,val in df.dtypes:
        if val == 'string':
            dsRetVals[key] = val
    return (dsRetVals)    
	
# unique value count in each columns
def getUniqCount(df):
    # pandas series
    dsRetVals = pd.Series() 
    # loop through each column
    for colName in df.columns:
        nUniq = df.select(colName).distinct().count()
        dsRetVals[colName] = nUniq
    return (dsRetVals)

# min max summary 
def getMinMaxSummary(df):
    # pandas series
    dft = df.describe()
    # loop through each column
    for colName in dft.columns:
        if colName == "summary":
            continue
        dft = dft.withColumn(colName, col(colName).cast(DoubleType()))
        dft = dft.withColumn(colName, f.round(colName, 2))
    return (dft)

# quartile summary 
def getQuartileSummary(df):
    # summary df
    dft = df.summary()
    # loop through each column
    for colName in dft.columns:
        if colName == "summary":
            continue
        dft = dft.withColumn(colName, col(colName).cast(DoubleType()))
        dft = dft.withColumn(colName, f.round(colName, 2))
    return (dft)
	
# minimum value in each column
def getMinValues(df):
    dsRetVals = pd.Series() 
    # loop through each column
    for colName in df.columns:
        # get minimum of column 
        vMins = df.select(min(colName)).collect()[0][0]
        dsRetVals[colName] = vMins
    return (dsRetVals)

# mean value in each column
def getMeanValues(df):
    # pandas series
    dsRetVals = pd.Series() 
    # loop through each column
    for colName in df.columns:
        # get mean of column 
        vMean = df.select(mean(colName)).collect()[0][0]
        dsRetVals[colName] = vMean
    return (dsRetVals)

# median value in each column
def getMedianValues(df):
    # pandas series
    dsRetVals = pd.Series() 
    # loop through each column
    for colName in df.columns:
        # get mean of column 
        vMeds = df.select(median(colName)).collect()[0][0]
        dsRetVals[colName] = vMeds
    return (dsRetVals)

# maximum value in each column
def getMaxValues(df):
    # pandas series
    dsRetVals = pd.Series() 
    # loop through each column
    for colName in df.columns:
        # get maximum of column 
        vMaxs = df.select(max(colName)).collect()[0][0]
        dsRetVals[colName] = vMaxs
    return (dsRetVals)

# get unique from categorical variable
def getUniqueValues(df, colName):
    # get unique in list
    lUniq = df.select(colName).distinct().collect()
    # sort list
    lUniq.sort()
    # pandas series
    dsRetVals = pd.Series() 
    # loop through each column\
    for i in range(0,len(lUniq)):
        dsRetVals[lUniq[i][0]] = ""
    return (dsRetVals)

#### end eda functions

# COMMAND ----------

# read csv
print("\n*** Read CSV ***")
df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").option("sep", ",")  \
  .load("/FileStore/tables/obesity_data.csv")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic EDA and Transformations

# COMMAND ----------

############################################
#### start checks & transformation functions 
############################################

# imports
from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType
from pyspark.sql.functions import mode, min, mean, median, max
from pyspark.sql.functions import col, isnan, when, count, countDistinct
from pyspark.ml.feature import StringIndexer, VectorAssembler

"""
desc:
    ColsToFloat - converts string column with numbers to double  
                  also converts integer column to float if requird
usage: 
    ColsToFloat(df, colNames) 
params:
    df datarame, 
    colNames - col for to convert string to float
"""
# ColToFloat
def ColToFloat(df, colNames=[]):
    # if colNames not list make it as list
    if not isinstance(colNames, list):
        colNames = [colNames]
    # loop through each column
    for colName in colNames:
        #print("Col Name: ",colName)
        #print(df.schema[colName].dataType)
        df = df.withColumn(colName, col(colName).cast(DoubleType()))

    return (df)	
	
# https://www.machinelearningplus.com/pyspark/pyspark-stringindexer/
"""
desc:
    CharToNum - converts column with string categoric values to double  
usage: 
    CharToNum(df, colName, model) 
params:
    df - datarame 
    colName - col for to encode char to numeric
    mo - pre-defined string indexer model
"""
def CharToNum(df, colName, mo):
    # print unique
    print("\n*** Before ***")
    print(getUniqueValues(df, colName))
    # print
    print("\nProcessing ...")
    # var names
    #modelSpecies = None
    #indexSpecies = None
    # find Count of Null, None, NaN of All DataFrame Columns
    vNulls = df.select([count(when(isnan(col(colName)) | col(colName).isNull(), colName)).alias(colName)]).collect()[0][0]
    # print
    #print(colName,vNulls)
    # check if nulls are present
    if vNulls > 0:
        # get mode of column 
        vMode = df.select(mode(colName)).collect()[0][0]
        # replace null with mode  
        df = df.fillna(value=vMode,subset=[colName])
    # drop col to ensure new col is created
    df = df.drop(col('ICol'))
    # print
    print(mo)
    # transform the DataFrame using the fitted StringIndexer model
    df = mo.transform(df)
    # update colName
    df = df.drop(col(colName))
    df = df.withColumnRenamed('ICol', colName)
    # print unique
    print("\n*** After ***")
    print(getUniqueValues(df, colName))
    print("Done ...")

    return (df)

# same value checks | cardinality check
# identify columns where all value are same (uniqVals=1 ie cardinality = 1)
# identify columns where two or less values are unique (uniqVals=2)
# identify columns where three or less values are unique (uniqVals=3)
# works only where all cols are numeric
def SameValuesCols(df, lExclCols=[], uniqVals=1, Verbose=False):
    # get column names
    colNames = df.columns
    # lExclCols not list make it as list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    # remove lExclCols from colNames
    if lExclCols != []:
        colNames = [colName for colName in colNames if colName not in lExclCols]
    #print("AllCols: ",df.columns)
    #print("SelCols: ",colNames)
    # handle same value for each col
    lRetVals = []
    dsRetValue = pd.Series() 
    for colName in colNames:
        cntUniq = df.select(countDistinct(colName)).collect()[0][0]
        #print(type(cntUniq))
        #cntRecs  = len(df.index)
        dsRetValue[colName] = '%7d' % cntUniq
        if (cntUniq <= uniqVals):
            lRetVals.append(colName)
    if (Verbose):       
        print(dsRetValue)    
    return lRetVals

# identify columns with more than x% unique values
# change value of x by changing value of Percent
def UniqValuesCols(df, lExclCols=[], Percent=0.95, Verbose = False):
    # get column names
    colNames = df.columns
    # lExclCols not list make it as list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    # remove lExclCols from colNames
    if lExclCols != []:
        colNames = [colName for colName in colNames if colName not in lExclCols]
    #print("AllCols: ",df.columns)
    #print("SelCols: ",colNames)
    # handle uniq values for each col
    dsRetValue = pd.Series() 
    lRetVals = []
    for colName in colNames:
        cntUniq = df.select(countDistinct(colName)).collect()[0][0]
        cntRecs = df.count()
        perRecs = cntUniq / cntRecs
        dsRetValue[colName] = '%.2f' % perRecs
        if perRecs >= Percent:
            lRetVals.append(colName)
    if (Verbose):       
        print(dsRetValue)    
    return lRetVals
   
# identify columns with more than x% null values
# change value of x by changing value of Percent
def NullValuesCols(df, lExclCols=[], Percent=0.50, Verbose = False):
    # default percent 0.5 or 50%
    if (Percent < 0) & (Percent>1) :
        Percent=0.5
    # get column names
    colNames = df.columns
    # lExclCols not list make it as list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    # remove lExclCols from colNames
    if lExclCols != []:
        colNames = [colName for colName in colNames if colName not in lExclCols]
    #print("AllCols: ",df.columns)
    #print("SelCols: ",colNames)
    # handle null values for each col
    lRetVals = []
    dsRetValue = pd.Series() 
    for colName in colNames:
        # get null count
        cntNulls = df.select([count(when(isnan(col(colName)) | col(colName)\
            .isNull(), colName)).alias(colName)]).collect()[0][0]
        cntRecs  = df.count()
        perRecs  = cntNulls / cntRecs
        #print(colName)
        #print(perRecs)
        #print(Percent)
        if perRecs >= Percent:
            lRetVals.append(colName)
        dsRetValue[colName] = '%.2f' % perRecs
    if (Verbose):       
        print(dsRetValue)    
    return (lRetVals)

"""
desc:
    checkZeros - checks zero value in all cols of df
usage: 
    checkZeros(df)
params:
    df datarame
"""
# check nulls in df
def checkZeros(df):
    # create pandas series to hold output
    dsRetVals = pd.Series() 
    # not applicable for string data
    # loop through each column
    for colName in df.columns:
        if df.schema[colName].dataType==StringType():
            dsRetVals[colName] = "NA"     
            continue
        # get zero count
        vZeros = df.select([count(when(col(colName) == 0, colName)).alias(colName)]).collect()[0][0]
        # assign to pd series
        dsRetVals[colName] = vZeros
    return (dsRetVals) 

"""
desc:
    HandleZeros - converts zeros to nulls from all specified cols of df 
usage: 
    HandleNulls(df, colNames) 
params:
    df datarame, 
    colNames - cols for handling nulls
"""
# Handle Zeros
def handleZeros(df, colNames=[]):
    # if colNames not list make it as list
    if not isinstance(colNames, list):
        colNames = [colNames]
    # create pandas series to hold output
    # dsRetVals = pd.Series() 
    # loop through each column
    for colName in colNames:
        # check data type
        if df.schema[colName].dataType==StringType():
            continue
        # get zerp count
        vZeros = df.select([count(when(col(colName) == 0, colName)).alias(colName)]).collect()[0][0]
        # print
        #print("ColName: ",colName)
        #print("NullCnt: ",vZeros)
        # check if 0s exists
        if vZeros > 0:
            # replace zeros with nulls
            df = df.withColumn(colName, when(df[colName] == 0,None).otherwise(df[colName]))            
            # convert to double
            df = df.withColumn(colName, col(colName).cast(DoubleType()))
    return (df)

"""
desc:
    checkNulls - checks null value in all cols in df
usage: 
    checkNulls(df)
params:
    df datarame
"""
# check nulls in df
def checkNulls(df):
    # create pandas series to hold output
    dsRetVals = pd.Series() 
    # loop through each column
    for colName in df.columns:
        # not applicable for string data
        if df.schema[colName].dataType==StringType():
            dsRetVals[colName] = "NA"     
            continue
        # get null count
        vNulls = df.select([count(when(isnan(col(colName)) | col(colName)\
            .isNull(), colName)).alias(colName)]).collect()[0][0]
        # assign to pd series
        dsRetVals[colName] = vNulls
    return (dsRetVals) 

"""
desc:
    HandleNulls - removes null from all cols in df except lExclCols 
usage: 
    HandleNulls(df, replBy, lExclCols) 
params:
    df datarame, 
    replBy - mean, median, minimum or maximum (of mean & median) 
    lExclCols - col to ignore while handling nulls
"""
# Handle nulls
def handleNulls(df, replBy, lExclCols):
    # get column names
    colNames = df.columns
    # lExclCols not list make it as list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    # remove lExclCols from colNames
    if lExclCols != []:
        colNames = [colName for colName in colNames if colName not in lExclCols]
    #print("AllCols: ",df.columns)
    #print("SelCols: ",colNames)
    # create pandas series to hold output
    dsRetVals = pd.Series() 
    # loop through each column
    for colName in colNames:
        # check data type
        if df.schema[colName].dataType==StringType():
            continue
        # get null count
        vNulls = df.select([count(when(isnan(col(colName)) | col(colName)\
            .isNull(), colName)).alias(colName)]).collect()[0][0]
        # print
        #print("ColName: ",colName)
        #print("NullCnt: ",vNulls)
        # check if nulls are present
        if vNulls > 0:
            # get mode of column 
            if replBy == "mean":
                replVals = df.select(mean(colName)).collect()[0][0]
            elif replBy == "median":
                replVals = df.select(median(colName)).collect()[0][0]
            elif replBy == "minimum":
                replVals = min(df.select(mean(colName)).collect()[0][0],df\
                    .select(median(colName)).collect()[0][0])
            elif replBy == "maximum":
                replVals = max(df.select(mean(colName)).collect()[0][0],df\
                    .select(median(colName)).collect()[0][0])
            # print
            #print("ColName: ",colName)
            #print("ReplVal: ",replVals)
            # replace null with replVals  
            df = df.fillna(value=replVals,subset=[colName])
            # convert to double
            df = df.withColumn(colName, col(colName).cast(DoubleType()))
    return (df)

# dataFrame handle nulls replace with mean of the columns
# wrapper around HandleNulls
def handleNullsWithMean(df, exclCols=[]):
    df = handleNulls(df, "mean", exclCols)
    return df

# dataFrame handle nulls replace with median of the columns
# wrapper around HandleNulls
def handleNullsWithMedian(df, exclCols=[]):
    df = handleNulls(df, "median", exclCols)
    return df

# dataFrame handle nulls replace with min(mean, median) of the columns
# wrapper around HandleNulls
def handleNullsWithMinOfMM(df, exclCols=[]):
    df = handleNulls(df, "minimum", exclCols)
    return df

# dataFrame handle nulls replace with max(mean, median) of the columns
# wrapper around HandleNulls
def handleNullsWithMaxOfMM(df, exclCols=[]):
    df = handleNulls(df, "maximum", exclCols)
    return df

# outlier limits
"""
returns: 
    upper bound & lower bound for array values or df[col] 
usage: 
    OutlierLimits(df[col]): 
"""
def colOutlierLimits(colValues, pMul=3): 
    # check pMul Value | should be only these
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    # convert pMul to float
    pMul = float(pMul)
    # get quartile 1 & quartile 3
    q1, q3 = np.percentile(colValues, [25, 75])
    # get inter quartile range
    iqr = q3 - q1
    # get lower limit
    ll = q1 - (iqr * pMul)
    # get upper limit
    ul = q3 + (iqr * pMul)
    # return
    return ll, ul

# outlier count for column
"""
returns: 
    outliers data for specific column of dataframe
usage: 
    colOutlierData(df, type, pMul)
params:
    df datarame
    type - count, index, value
    pMul - multiple for lower limit & upper limit         
"""
def colOutlierData(colValues, type, pMul=3):
    # check pMul Value | should be only these
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    # convert pMul to float
    pMul = float(pMul)
    # get lower limit  & upper limit 
    ll, ul = colOutlierLimits(colValues, pMul)
    #print("LL / UL: ",ll,ul)
    # get outlier data
    ndOutData = np.where((colValues > ul) | (colValues < ll))
    # convert outlier data to array
    ndOutData = np.array(ndOutData)
    #print("OutData: ",ndOutData)
    vRetVals = ""
    # chose what to return
    if type == "Count":
        vRetVals = ndOutData.size
    elif type == "Index":
        vRetVals = ndOutData.tolist()
    elif type == "Values":
        vRetVals = colValues[ndOutData].tolist()    
    #print("RetVals: ",vRetVals)
    # return len of Outlier Data
    return (vRetVals)

# outlier data for dataframe
"""
returns: 
    outliers data for each column of dataframe
usage: 
    getOutlierData(df, type, pMul)
params:
    df datarame
    type - count, index, value
    pMul - multiple for lower limit & upper limit         
"""
def getOutlierData(df, type, pMul=3): 
    # check pMul Value | should be only these
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    # convert pMul to float
    pMul = float(pMul)
    # get colNames    
    colNames = df.columns
    # pandas series for output
    dsRetValue = pd.Series() 
    # loopthrough columns
    for colName in colNames:
        # not applicable for string data
        if df.schema[colName].dataType==StringType():
            dsRetValue[colName] = "NA"     
            continue
        # print
        #print("ColName: ",colName)
        #print("ColType: ",df.schema[colName].dataType)
        # get null count
        vNulls = df.select([count(when(isnan(col(colName)) | col(colName)\
            .isNull(), colName)).alias(colName)]).collect()[0][0]
        #print("NullCnt: ",vNulls)
        # check if nulls are present
        if vNulls > 0:
            # replVals is mean of colName
            replVals = df.select(mean(colName)).collect()[0][0]
            # now replace
            df = df.fillna(value=replVals,subset=[colName])
            # convert to double
            df = df.withColumn(colName, col(colName).cast(DoubleType()))
        # convert column to nmupy array 
        colValues = np.array(df.select(colName).collect()).reshape(-1)
        #print("ColVals: ",colValues)
        # get outlier gata For the column
        colOutput = colOutlierData(colValues, type, pMul)
        #print("ColOuts: ",colOutput)
        # store in panas series
        dsRetValue[colName] = colOutput
    return(dsRetValue)

# column level handle outlier by capping
# at lower limit & upper timit respectively
"""
returns: 
    array values or df[col].values without any outliers
usage: 
    HandleOutlier(df[col].values): 
"""
def colHandleOutliers(df, colName, colValues, pMul=3):
    # get colValues 
    # colValues = np.array(df.select(colName).collect()).reshape(-1)
    # get lower limit & upper limit 
    ll, ul = colOutlierLimits(colValues, pMul)
    # handle outliers
    df = df.withColumn(colName, when(df[colName] < ll,ll).otherwise(df[colName]) )
    df = df.withColumn(colName, when(df[colName] > ul,ul).otherwise(df[colName]) )
    return (df)

# data frame level handline outliers
"""
desc:
    handleOutliers - removes Outliers from all cols in df except exclCols 
usage: 
    handleOutliers(df, colClass) 
params:
    df datarame, exclCols - col to ignore while transformation, Multiplier  
"""
def handleOutliers(df, lExclCols=[], pMul=3):
    #lExclCols = depVars
    # orig col names
    colNames = df.columns
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle outlier for each col
    for colName in colNames:
        #colValues = df[colName].values
        colValues = np.array(df.select(colName).collect()).reshape(-1)
        if (colOutlierData(colValues, "Count", pMul) > 0):
            #print(colName)
            df = colHandleOutliers(df,colName, colValues, pMul)
    return df
    
# get outlier count
# wrapper for getOutlierData
def getOutlierCount(df, pMul=3): 
    return (getOutlierData(df, "Count", pMul=3)) 

# get outlier index
# wrapper for getOutlierData
def getOutlierIndex(df, pMul=3): 
    return (getOutlierData(df, "Index", pMul=3)) 

# get outlier values
# wrapper for getOutlierData
def getOutlierValues(df, pMul=3): 
    return (getOutlierData(df, "Values", pMul=3)) 

"""
desc:
    Normalize data - all cols of df will be Normalized except lExclCols
    x_scaled = (x-min(x)) / (max(x)–min(x))
    all values will be between 0 & 1 or -1 & 1 (if negative values are present)
usage: 
    NormalizeData(df, colClass) 
params:
    df datarame, lExclCols - cols to ignore while transformation  
"""
# nomalize data
def NormalizeData(df, lExclCols=[], ):
    # orig col names
    colNames = df.columns
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle outlier for each col
    for colName in colNames:
        # get min & max of col
        vMin = df.select(f.min(colName)).collect()[0][0]
        vMax = df.select(f.max(colName)).collect()[0][0]
        # replace
        df = df.withColumn(colName,((df[colName]-vMin)/(vMax-vMin)))
    
    return df
	

"""
desc:
    Standardize data - all cols of df will be Standardized except colClass 
    x_scaled = (x — mean(x)) / stddev(x)
    all values will be between 1 & -1
usage: 
    StandardizeData(df, colClass) 
params:
    df datarame, colClass - col to ignore while transformation  
"""
# standardize data
def StandardizeData(df, lExclCols=[]):
    # orig col names
    colNames = df.columns
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle outlier for each col
    for colName in colNames:
        # get min & max of col
        vMean = df.select(f.mean(colName)).collect()[0][0]
        vStdDev = df.select(f.stddev(colName)).collect()[0][0]
        # replace
        df = df.withColumn(colName,((df[colName]-vMean)/(vStdDev)))
    
    return df	

# get Top Rows of Features Column
def getFeaturesTopRows(df, vRows=5):
    dsRetVals = pd.Series() 
    for i in range(0, vRows-1):
        dsRetVals[str(i)] = df.collect()[i][-1]
    return (dsRetVals)

# get properties Features row 0 column 1 
def getFeaturesProperties(df):
    vChkData = df.collect()[0][-1] 
    print("\n*** Top Row Property ***")
    print("Top Row : ",vChkData)
    print("DataType: ",type(vChkData))
    print("ArrySize: ",vChkData.size)
    return
	
#### end checks & transformation functions

# COMMAND ----------

# rows & cols
print("\n*** Rows & Cols ***")
print("Rows: ",df.count())
print("Cols: ",len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC #### No of rows :- 280000
# MAGIC #### No of Cols (Features) : 18

# COMMAND ----------

# get col names
print("\n*** Column Names ***")
print(getColNames(df))

# COMMAND ----------

# get col data types
print("\n*** Data Types ***")
print(getDataTypes(df))

# COMMAND ----------

df=df.drop('_c0')

# COMMAND ----------

display(df)

# COMMAND ----------

# get unique counts
print("\n*** Unique Count In Cols ***")
print(getUniqCount(df))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unique Values = 1525 / 280000 = 0.5% of total values
# MAGIC #### No Unique Value Transformation needed

# COMMAND ----------

# get min max summary 
print("\n*** Min Max Summary ***")
dfs = getMinMaxSummary(df)
display(dfs)

# COMMAND ----------

# get quartile summary 
print("\n*** Quartile Summary ***")
dfq = getQuartileSummary(df)
display(dfq)

# COMMAND ----------

# get min values   
print("\n*** Minimum Values In Cols ***")
print(getMinValues(df))

# COMMAND ----------

# get mean values
print("\n*** Mean Values In Cols ***")
print(getMeanValues(df))	

# COMMAND ----------

# get median values
print("\n*** Median Values In Cols ***")
print(getMedianValues(df))

# COMMAND ----------

# get max values
print("\n*** Maximum Values In Cols ***")
print(getMaxValues(df))

# COMMAND ----------

# schema
print("\n*** Schema ***")
print(df.printSchema())

# COMMAND ----------

# get string cols
print("\n*** String Cols ***")
print(getStringCols(df))

# COMMAND ----------

print(df.printSchema())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformations

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Null Handling

# COMMAND ----------

print(checkNulls(df))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Zero Null Values so no null handling required

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Zero Handling

# COMMAND ----------

print(checkZeros(df))

# COMMAND ----------

# MAGIC %md
# MAGIC #### The Zeros will not be treated as it can affect the accuracy of the classification

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Col to Float

# COMMAND ----------

# imports
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType, DateType
"""
desc:
    ColsToFloat - converts string column with numbers to double  
                  also converts integer column to float if requird
usage: 
    ColsToFloat(df, colNames) 
params:
    df datarame, 
    colNames - col for to convert string to float
"""
# ColToFloat
def ColToFloat(df, colNames=[]):
    # if colNames not list make it as list
    if not isinstance(colNames, list):
        colNames = [colNames]
    # loop through each column
    for colName in colNames:
        #print("Col Name: ",colName)
        #print(df.schema[colName].dataType)
        df = df.withColumn(colName, col(colName).cast(DoubleType()))

    return (df)

# COMMAND ----------

print(df.printSchema())

# COMMAND ----------

df = df.drop('_c0')

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Char to Num

# COMMAND ----------

# encode char to num 
# https://www.machinelearningplus.com/pyspark/pyspark-stringindexer/
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import mode
from pyspark.sql.functions import col,isnan, when, count

"""
desc:
    CharToNum - converts column with string categoric values to double  
usage: 
    CharToNum(df, colName, model) 
params:
    df - datarame 
    colName - col for to encode char to numeric
    mo - pre-defined string indexer model
"""
def CharToNum(df, colName, mo):
    # print unique
    print("\n*** Before ***")
    print(getUniqueValues(df, colName))
    # print
    print("\nProcessing ...")
    # var names
    #modelSpecies = None
    #indexSpecies = None
    # find Count of Null, None, NaN of All DataFrame Columns
    vNulls = df.select([count(when(isnan(col(colName)) | col(colName).isNull(), colName)).alias(colName)]).collect()[0][0]
    # print
    #print(colName,vNulls)
    # check if nulls are present
    if vNulls > 0:
        # get mode of column 
        vMode = df.select(mode(colName)).collect()[0][0]
        # replace null with mode  
        df = df.fillna(value=vMode,subset=[colName])
    # drop col to ensure new col is created
    df = df.drop(col('ICol'))
    # print
    print(mo)
    # transform the DataFrame using the fitted StringIndexer model
    df = mo.transform(df)
    # update colName
    df = df.drop(col(colName))
    df = df.withColumnRenamed('ICol', colName)
    # print unique
    print("\n*** After ***")
    print(getUniqueValues(df, colName))
    print("Done ...")

    return (df)


# COMMAND ----------

df.printSchema()

# COMMAND ----------

# colName
colName = "Gender"
# create string indexer object
sigender = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mogender = sigender.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mogender)

# COMMAND ----------

# colName
colName = "family_history_with_overweight"
# create string indexer object
sihist = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mohist = sihist.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mohist)

# COMMAND ----------

# colName
colName = "FAVC"
# create string indexer object
sifavc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mofavc = sifavc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mofavc)

# COMMAND ----------

# colName
colName = "CAEC"
# create string indexer object
sicaec = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mocaec = sicaec.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mocaec)

# COMMAND ----------

# colName
colName = "SMOKE"
# create string indexer object
sismoke = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mosmoke = sismoke.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mosmoke)

# COMMAND ----------

# colName
colName = "SCC"
# create string indexer object
siscc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
moscc = siscc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, moscc)

# COMMAND ----------

# colName
colName = "CALC"
# create string indexer object
sicalc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mocalc = sicalc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mocalc)

# COMMAND ----------

# colName
colName = "MTRANS"
# create string indexer object
sitrans = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
motrans = sitrans.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, motrans)

# COMMAND ----------

# colName
colName = "NObeyesdad"
# create string indexer object
siobeys = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
moobeys = siobeys.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, moobeys)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Outlier Handling

# COMMAND ----------

# check outlier count
print('\n*** Outlier Count ***')
print(getOutlierCount(df))

# COMMAND ----------

# check outlier index
print('\n*** Outlier Index ***')
print(getOutlierIndex(df))

# COMMAND ----------

# check outlier count
print('\n*** Outlier Values ***')
print(getOutlierValues(df))

# COMMAND ----------

# check outlier count
print('\n*** Outlier Count ***')
print(getOutlierCount(df))

# COMMAND ----------

# schema
print("\n*** Schema ***")
print(df.printSchema())

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Data Scaling

# COMMAND ----------

# imports
from pyspark.sql import functions as f
from pyspark.sql.functions import min, mean, median, max
"""
desc:
    Normalize data - all cols of df will be Normalized except lExclCols
    x_scaled = (x-min(x)) / (max(x)–min(x))
    all values will be between 0 & 1 or -1 & 1 (if negative values are present)
usage: 
    NormalizeData(df, colClass) 
params:
    df datarame, lExclCols - cols to ignore while transformation  
"""
# nomalize data
def NormalizeData(df, lExclCols=[], ):
    # orig col names
    colNames = df.columns
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle outlier for each col
    for colName in colNames:
        # get min & max of col
        vMin = df.select(f.min(colName)).collect()[0][0]
        vMax = df.select(f.max(colName)).collect()[0][0]
        # replace
        df = df.withColumn(colName,((df[colName]-vMin)/(vMax-vMin)))
    
    return df

# COMMAND ----------

# handle normalization if required
print('\n*** Normalize Data ***')
df = NormalizeData(df, "NObeyesdad")
print('Done ...')

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation (With only Normalization) [First Pass]

# COMMAND ----------

# change as required
# data preparation
# rename class vars to 'label'
df = df.withColumnRenamed("NObeyesdad","label")
# set ML columns
allCols = df.columns
clsCols = "label"
datCols = df.columns
datCols.remove(clsCols)
print("*** ML Columns ***")
print("AllCols:", allCols)
print("datCols:",datCols)
print("clsCols:",clsCols)

# COMMAND ----------

# create feature dataframe & features column
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=datCols,outputCol="features")
df_ftr = assembler.transform(df)
print("*** Feature Dataframe ***")
df_ftr.show(5)

# COMMAND ----------

# schema
print("\n*** Schema ***")
df_ftr.printSchema()

# COMMAND ----------

display(df_ftr)

# COMMAND ----------

# split fature into train & test
print("*** Split FeatureDF into TrainDF & TestDF ***")
df_trn, df_tst = df_ftr.randomSplit([0.8, 0.2],707)
print("Done ...")

# COMMAND ----------

# rows & cols
print("\n*** Rows & Cols of Train Dataset ***")
print("Rows",df_trn.count())
print("Cols",len(df_trn.columns))

# COMMAND ----------

# rows & cols
print("\n*** Rows & Cols of Test Dataset ***")
print("Rows",df_tst.count())
print("Cols",len(df_tst.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Creation

# COMMAND ----------

# prepare models
from pyspark.sql.functions import col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# binary classifiers
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LinearSVC
# multi class classifiers
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
# unique labels
print("\n*** Label Check ***")
lUniqLbls = df.select('Label').distinct().collect()
print("Unique Labels: ", lUniqLbls)
# list for model names
lModelName = []
if lUniqLbls == 2:
    lModelName.append("One v/s Best Classifier")
    lModelName.append("Gradient Boost Classifier")
    lModelName.append("Support Vector Classifier")
else:
    lModelName.append("Naive Bayes Classifier")
    lModelName.append("Decision Tree Classifier")
    lModelName.append("Random Forest Classifier")
    lModelName.append("Logistic Regression Classifier")
# list for model object
lModelObjs = []
if lUniqLbls == 2:
    lModelObjs.append(OneVsRest())
    lModelObjs.append(GBTClassifier())
    lModelObjs.append(LinearSVC())
else:    
    lModelObjs.append(NaiveBayes())
    lModelObjs.append(DecisionTreeClassifier())
    lModelObjs.append(RandomForestClassifier())
    lModelObjs.append(LogisticRegression())
# lists for train  acc
lModelTrnAcc = []
# List for test acc
lModelTstAcc = []
# print
print("\n*** Prepare Models ***")
msg = "%-30s %-30s" % ("Model Name", "Model Object")
print(msg)
# for each model
for i in range(0,len(lModelName)):   
    # print model name, model object 
    msg = "%-30s %-30s" % (lModelName[i], lModelObjs[i])
    print(msg)

# COMMAND ----------

# select best model
print("*** Select Best Model ***")

# for each model
for i in range(0,len(lModelName)):   

    # print
    print("\nModel Name  : ",lModelName[i])
    # instantiate empty model
    algos =  lModelObjs[i]
    # prepare dft_trn
    df_trn = df_trn.drop(col('predict'))
    df_trn = df_trn.drop(col('rawPrediction'))
    df_trn = df_trn.drop(col('probability'))
    # train the model
    model = algos.fit(df_trn)
    # print
    #print("*** ML Model ***")
    print("Model Object: ",model)

    # predict train
    df_trn = model.transform(df_trn)
    df_trn = df_trn.withColumnRenamed("prediction","predict")
    # train data - important cols
    #print("\n*** Train Data - Features Label & Predict ***")
    #df_trn.select("features","label","predict").show(5)
    #df_trn.printSchema()

    # evaluate train based on label check
    if lUniqLbls == 2:
        # for binary classification
        evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
        trn_auc = evl_auc.evaluate(df_trn)
    else:
        # for multi class classification
        evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
        trn_acc = evl_acc.evaluate(df_trn)
    # model train accuracy
    lModelTrnAcc.append(trn_acc)
    # print
    #print("*** Train Data Evaluation ***")
    print("TRN-Accuracy: %3.2f %%" % (trn_acc*100))

    # predict test
    df_tst = df_tst.drop(col('predict'))
    df_tst = df_tst.drop(col('rawPrediction'))
    df_tst = df_tst.drop(col('probability'))
    df_tst = model.transform(df_tst)
    df_tst = df_tst.withColumnRenamed("prediction","predict")
    # train data - important cols
    #print("\n*** Test Data - Features Label & Predict ***")
    #df_tst.select("features","label","predict").show(5)
    #df_tst.printSchema()

    # evaluate test
    if lUniqLbls == 2:
        # for binary classification
        evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
        tst_auc = evl_auc.evaluate(df_tst)
    else:
        # for multi class classification
        evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
        tst_acc = evl_acc.evaluate(df_tst)
    # model test accuracy
    lModelTstAcc.append(tst_acc)
    # print
    #print("*** Test Data Evaluation ***")
    print("TST-Accuracy: %3.2f %%" % (tst_acc*100))
print("Done ...")

# COMMAND ----------

# summary
print("\n*** Model Summary ***")
# header
msg = "%-30s %10s %8s" % ("Model", "Train-Acc", "  Test-Acc")
print(msg)
# for each model
for i in range(0,len(lModelName)):   
    # print model name, rsqr & rmse   
    msg = "%-30s %10.7f %10.7f" % (lModelName[i], lModelTrnAcc[i], lModelTstAcc[i])
    print(msg)

# COMMAND ----------

# import
import builtins
# find model with the best accuracy details
print("\n*** Best Accuracy Model ***")
vIndex = lModelTstAcc.index(builtins.max(lModelTstAcc))
print("Index      : ",vIndex)
print("Model Name : ",lModelName[vIndex])
print("TrnAccuracy: ",lModelTrnAcc[vIndex])
print("TstAccuracy: ",lModelTstAcc[vIndex])

# COMMAND ----------

# visualize test accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# get label & predict column
y_orig = df_tst.select(['label']).collect()
y_pred = df_tst.select(['predict']).collect()
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_orig, y_pred)*100
print("%3.2f %%" % accuracy)
# confusion matrix
from sklearn.metrics import confusion_matrix
print("\n*** Confusion Matrix - Original ***")
cm = confusion_matrix(y_orig, y_orig)
print(cm)
# confusion matrix predicted
print("\n*** Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_orig, y_pred)
print(cm)
# confusion matrix predicted
print("\n*** Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_orig, y_pred)
print(cm)
# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_orig, y_pred)
print(cr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The best classification algorithm is Logistic Regression (Multinomial) Classification with an accuracy of 98.73%

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation using New Data

# COMMAND ----------

# read csv
print("\n*** Read CSV ***")
df_new = spark.read.format("csv").option("inferSchema", "true").option("header", "true").option("sep", ",")  \
  .load("/FileStore/tables/newobesedata.csv")
# top 5 rows
display(df_new)

# COMMAND ----------

display(df_new)

# COMMAND ----------

df_new.drop('_c0')

# COMMAND ----------

display(df_new)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Checks of new data

# COMMAND ----------

# rows & cols
print("\n*** New Data - Rows & Cols ***")
print("Rows: ",df_new.count())
print("Cols: ",len(df_new.columns))

# COMMAND ----------

# schema
print("\n*** New Data - Schema ***")
print(df_new.printSchema())

# COMMAND ----------

# get string cols
print("\n*** New Data - String Cols ***")
print(getStringCols(df_new))

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "Gender"
sigender = StringIndexer(inputCol=colName, outputCol="ICol")
mogender = sigender.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mogender)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "family_history_with_overweight"
sihist = StringIndexer(inputCol=colName, outputCol="ICol")
mohist = sihist.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mohist)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "FAVC"
sifavc = StringIndexer(inputCol=colName, outputCol="ICol")
mofavc = sifavc.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mofavc)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "CAEC"
sicaec = StringIndexer(inputCol=colName, outputCol="ICol")
mocaec = sicaec.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mocaec)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "SMOKE"
sismoke = StringIndexer(inputCol=colName, outputCol="ICol")
mosmoke = sismoke.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mosmoke)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "SCC"
siscc = StringIndexer(inputCol=colName, outputCol="ICol")
moscc = siscc.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, moscc)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "CALC"
sicalc = StringIndexer(inputCol=colName, outputCol="ICol")
mocalc = sicalc.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mocalc)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "MTRANS"
sitrans = StringIndexer(inputCol=colName, outputCol="ICol")
motrans = sitrans.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, motrans)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "NObeyesdad"
siobeys = StringIndexer(inputCol=colName, outputCol="ICol")
moobeys = siobeys.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, moobeys)

# COMMAND ----------

display(df_new)

# COMMAND ----------

# check sulls
print("\n*** New Data - Check Null Values ***") 
print(checkNulls(df_new))

# COMMAND ----------

# check outlier count
print('\n*** New Data - Outlier Count ***')
print(getOutlierCount(df_new))

# COMMAND ----------

# MAGIC %md
# MAGIC ### We will not handle the outliers as it can change the distribution and affect our accuracy

# COMMAND ----------

print("\n*** New Data - Handle Nulls With Mean ***") 
dfn = handleNullsWithMean(df_new)
print("Done ...")

# COMMAND ----------

print(df_new.printSchema())

# COMMAND ----------

display(df_new)

# COMMAND ----------

# handle normalization if required
print('\n*** New Data - Normalize Data ***')
df_new = NormalizeData(df_new, "NObeyesdad")
print('None ...')

# COMMAND ----------

display(df_new)

# COMMAND ----------


# change as required
# data preparation
# rename class vars to 'label'
df_new = df_new.withColumnRenamed("NObeyesdad","label")
# set ML columns
allCols = df_new.columns
clsCols = "label"
datCols = df_new.columns
datCols.remove(clsCols)
print("*** ML Columns ***")
print("AllCols:", allCols)
print("datCols:",datCols)
print("clsCols:",clsCols)

# COMMAND ----------

# prediction data - schema
print("\n*** New Data - Schema ***")
dfn.printSchema()

# COMMAND ----------

# prediction data - create vector column
assembler = VectorAssembler(inputCols=datCols,outputCol="features")
dfn_new = assembler.transform(df_new)
print("\n*** New Data - Label & Features Col ***")
dfn_new.show(5)

# COMMAND ----------

display(dfn_new)

# COMMAND ----------

dfn_new = df_tst

# COMMAND ----------

display(dfn_new)

# COMMAND ----------

# new data - predict
from pyspark.sql.functions import col
dfn_new = dfn_new.drop(col('predict'))
dfn_new = dfn_new.drop(col('rawPrediction'))
dfn_new = dfn_new.drop(col('probability'))
dfn_new = model.transform(dfn_new)
dfn_new = dfn_new.withColumnRenamed("prediction","predict")
# if depVars available in dataframe, show depVars & prdVals 
# if depVars NOT available in dataframe, show only prdVals 
print("\n*** New Data - Features Label & Predict ***")
dfn_new.select("features","label","predict").show(5)
#dfn_new.select("features","predict").show(5)

# COMMAND ----------

# evaluate predict if available

# for binary classification
#evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
#new_auc = evl_auc.evaluate(dfn_new)

# accuracy
evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
new_acc = evl_acc.evaluate(dfn_new)

# print
print("*** New Data Evaluation ***")
print("NEW-Accuracy : %3.2f %%" %  (new_acc*100))

# COMMAND ----------

# visualize new accuracy
# get label & predict column
y_orig = dfn_new.select(['label']).collect()
y_pred = dfn_new.select(['predict']).collect()
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_orig, y_pred)*100
print("%3.2f %%" % accuracy)
# confusion matrix
from sklearn.metrics import confusion_matrix
print("\n*** New Data - Confusion Matrix - Original ***")
cm = confusion_matrix(y_orig, y_orig)
print(cm)
# confusion matrix predicted
print("\n*** New Data - Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_orig, y_pred)
print(cm)
# classification report
print("\n*** New Data - Classification Report ***")
cr = classification_report(y_orig, y_pred)
print(cr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The overall accuracy on train,test and new data is near about 99% on just normalized data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Second Pass :- Standardization

# COMMAND ----------

# read csv
print("\n*** Read CSV ***")
df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").option("sep", ",")  \
  .load("/FileStore/tables/obesity_data.csv")

# COMMAND ----------

display(df)

# COMMAND ----------

df=df.drop('_c0')

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformations + Standardization 

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "Gender"
# create string indexer object
sigender = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mogender = sigender.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mogender)

# COMMAND ----------

### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "family_history_with_overweight"
# create string indexer object
sihist = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mohist = sihist.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mohist)

# COMMAND ----------

### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "CAEC"
# create string indexer object
sicaec = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mocaec = sicaec.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mocaec)

# COMMAND ----------

### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "SMOKE"
# create string indexer object
sismoke = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mosmoke = sismoke.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mosmoke)

# COMMAND ----------

### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "SCC"
# create string indexer object
siscc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
moscc = siscc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, moscc)

# COMMAND ----------

### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "CALC"
# create string indexer object
sicalc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mocalc = sicalc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mocalc)

# COMMAND ----------

### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "MTRANS"
# create string indexer object
sitrans = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
motrans = sitrans.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, motrans)

# COMMAND ----------

### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "NObeyesdad"
# create string indexer object
siobeys = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
moobeys = siobeys.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, moobeys)

# COMMAND ----------

### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "FAVC"
# create string indexer object
sifavc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mofavc = sifavc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mofavc)

# COMMAND ----------

# check sulls
print("\n*** Check Null Values ***") 
print(checkNulls(df))

# COMMAND ----------

# check sulls
print("\n*** Check Zeros Values ***") 
print(checkZeros(df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### The zero values are being reflected due to one hot and char to num encoding of categorical variables

# COMMAND ----------

# imports
from pyspark.sql import functions as f
from pyspark.sql.functions import min, mean, median, max
"""
desc:
    Standardize data - all cols of df will be Standardized except colClass 
    x_scaled = (x — mean(x)) / stddev(x)
    all values will be between 1 & -1
usage: 
    StandardizeData(df, colClass) 
params:
    df datarame, colClass - col to ignore while transformation  
"""
# standardize data
def StandardizeData(df, lExclCols=[]):
    # orig col names
    colNames = df.columns
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle outlier for each col
    for colName in colNames:
        # get min & max of col
        vMean = df.select(f.mean(colName)).collect()[0][0]
        vStdDev = df.select(f.stddev(colName)).collect()[0][0]
        # replace
        df = df.withColumn(colName,((df[colName]-vMean)/(vStdDev)))
    
    return df

# COMMAND ----------

display(df)

# COMMAND ----------

# handle normalization if required
print('\n*** Standardize Data ***')
df = StandardizeData(df, "NObeyesdad")
print('None ...')

# COMMAND ----------

print(df.printSchema())

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preperation for Model

# COMMAND ----------

# change as required
# new data - ML columns
# rename depVars to 'label'
df = df.withColumnRenamed("NObeyesdad","label")
allCols = df.columns
clsCols =  "label"
datCols = df.columns
print("*** New Data - ML Columns ***")
print("AllCols:", allCols)
print("datCols:", datCols)
print("clsCols:", clsCols)

# COMMAND ----------

# create feature dataframe & features column
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=datCols,outputCol="features")
df_ftr = assembler.transform(df)
print("*** Feature Dataframe ***")
df_ftr.show(5)

# COMMAND ----------

# split fature into train & test
print("*** Split FeatureDF into TrainDF & TestDF ***")
df_trn, df_tst = df_ftr.randomSplit([0.8, 0.2],707)
print("Done ...")

# COMMAND ----------

display(df_trn)

# COMMAND ----------

# rows & cols
print("\n*** Rows & Cols of Train Dataset ***")
print("Rows",df_trn.count())
print("Cols",len(df_trn.columns))

# COMMAND ----------

# rows & cols
print("\n*** Rows & Cols of Test Dataset ***")
print("Rows",df_tst.count())
print("Cols",len(df_tst.columns))

# COMMAND ----------

# prepare models
from pyspark.sql.functions import col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# binary classifiers
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LinearSVC
# multi class classifiers
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
# unique labels
print("\n*** Label Check ***")
lUniqLbls = df.select('label').distinct().collect()
print("Unique Labels: ", lUniqLbls)
# list for model names
lModelName = []
if lUniqLbls == 2:
    lModelName.append("One v/s Best Classifier")
    lModelName.append("Gradient Boost Classifier")
    lModelName.append("Support Vector Classifier")
else:
    lModelName.append("Naive Bayes Classifier")
    lModelName.append("Decision Tree Classifier")
    lModelName.append("Random Forest Classifier")
    lModelName.append("Logistic Regression Classifier")
# list for model object
lModelObjs = []
if lUniqLbls == 2:
    lModelObjs.append(OneVsRest())
    lModelObjs.append(GBTClassifier())
    lModelObjs.append(LinearSVC())
else:    
    lModelObjs.append(NaiveBayes())
    lModelObjs.append(DecisionTreeClassifier())
    lModelObjs.append(RandomForestClassifier())
    lModelObjs.append(LogisticRegression())
# lists for train  acc
lModelTrnAcc = []
# List for test acc
lModelTstAcc = []
# print
print("\n*** Prepare Models ***")
msg = "%-30s %-30s" % ("Model Name", "Model Object")
print(msg)
# for each model
for i in range(0,len(lModelName)):   
    # print model name, model object 
    msg = "%-30s %-30s" % (lModelName[i], lModelObjs[i])
    print(msg)

# COMMAND ----------

# select best model
print("*** Select Best Model ***")

# for each model
for i in range(0,len(lModelName)):   

    # print
    print("\nModel Name  : ",lModelName[i])
    # instantiate empty model
    algos =  lModelObjs[i]
    # prepare dft_trn
    df_trn = df_trn.drop(col('predict'))
    df_trn = df_trn.drop(col('rawPrediction'))
    df_trn = df_trn.drop(col('probability'))
    # train the model
    model = algos.fit(df_trn)
    # print
    #print("*** ML Model ***")
    print("Model Object: ",model)

    # predict train
    df_trn = model.transform(df_trn)
    df_trn = df_trn.withColumnRenamed("prediction","predict")
    # train data - important cols
    #print("\n*** Train Data - Features Label & Predict ***")
    #df_trn.select("features","label","predict").show(5)
    #df_trn.printSchema()

    # evaluate train based on label check
    if lUniqLbls == 2:
        # for binary classification
        evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
        trn_auc = evl_auc.evaluate(df_trn)
    else:
        # for multi class classification
        evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
        trn_acc = evl_acc.evaluate(df_trn)
    # model train accuracy
    lModelTrnAcc.append(trn_acc)
    # print
    #print("*** Train Data Evaluation ***")
    print("TRN-Accuracy: %3.2f %%" % (trn_acc*100))

    # predict test
    df_tst = df_tst.drop(col('predict'))
    df_tst = df_tst.drop(col('rawPrediction'))
    df_tst = df_tst.drop(col('probability'))
    df_tst = model.transform(df_tst)
    df_tst = df_tst.withColumnRenamed("prediction","predict")
    # train data - important cols
    #print("\n*** Test Data - Features Label & Predict ***")
    #df_tst.select("features","label","predict").show(5)
    #df_tst.printSchema()

    # evaluate test
    if lUniqLbls == 2:
        # for binary classification
        evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
        tst_auc = evl_auc.evaluate(df_tst)
    else:
        # for multi class classification
        evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
        tst_acc = evl_acc.evaluate(df_tst)
    # model test accuracy
    lModelTstAcc.append(tst_acc)
    # print
    #print("*** Test Data Evaluation ***")
    print("TST-Accuracy: %3.2f %%" % (tst_acc*100))
print("Done ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standardization renders the data useless for any type of classification algorithm so we will not proceed further

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data without any standardization and normalization (Last Pass)

# COMMAND ----------

# read csv
print("\n*** Read CSV ***")
df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").option("sep", ",")  \
  .load("/FileStore/tables/obesity_data.csv")
# top 5 rows
df.head(5)

# COMMAND ----------

display(df)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "Gender"
# create string indexer object
sigender = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mogender = sigender.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mogender)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "family_history_with_overweight"
# create string indexer object
sihist = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mohist = sihist.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mohist)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "FAVC"
# create string indexer object
sifavc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mofavc = sifavc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mofavc)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "CAEC"
# create string indexer object
sicaec = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mocaec = sicaec.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mocaec)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "SMOKE"
# create string indexer object
sismoke = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mosmoke = sismoke.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mosmoke)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "SCC"
# create string indexer object
siscc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
moscc = siscc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, moscc)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "CALC"
# create string indexer object
sicalc = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
mocalc = sicalc.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, mocalc)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "MTRANS"
# create string indexer object
sitrans = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
motrans = sitrans.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, motrans)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgoric
# colName
colName = "NObeyesdad"
# create string indexer object
siobeys = StringIndexer(inputCol=colName, outputCol="ICol")
# train string indexer model
moobeys = siobeys.fit(df)
# call 
print("\n*** Char to Numeric - %s ***" % colName)
df = CharToNum(df, colName, moobeys)

# COMMAND ----------

print(df.printSchema())

# COMMAND ----------

df=df.drop('_c0')

# COMMAND ----------

# change as required
# data preparation
# rename class vars to 'label'
df = df.withColumnRenamed("NObeyesdad","label")
# set ML columns
allCols = df.columns
clsCols = "label"
datCols = df.columns
datCols.remove(clsCols)
print("*** ML Columns ***")
print("AllCols:", allCols)
print("datCols:",datCols)
print("clsCols:",clsCols)

# COMMAND ----------

# create feature dataframe & features column
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=datCols,outputCol="features")
df_ftr = assembler.transform(df)
print("*** Feature Dataframe ***")
df_ftr.show(5)

# COMMAND ----------

# split fature into train & test
print("*** Split FeatureDF into TrainDF & TestDF ***")
df_trn, df_tst = df_ftr.randomSplit([0.8, 0.2],707)
print("Done ...")

# COMMAND ----------

# rows & cols
print("\n*** Rows & Cols of Train Dataset ***")
print("Rows",df_trn.count())
print("Cols",len(df_trn.columns))

# COMMAND ----------

# rows & cols
print("\n*** Rows & Cols of Test Dataset ***")
print("Rows",df_tst.count())
print("Cols",len(df_tst.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Creation

# COMMAND ----------

# prepare models
from pyspark.sql.functions import col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# binary classifiers
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LinearSVC
# multi class classifiers
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
# unique labels
print("\n*** Label Check ***")
lUniqLbls = df.select('Label').distinct().collect()
print("Unique Labels: ", lUniqLbls)
# list for model names
lModelName = []
if lUniqLbls == 2:
    lModelName.append("One v/s Best Classifier")
    lModelName.append("Gradient Boost Classifier")
    lModelName.append("Support Vector Classifier")
else:
    lModelName.append("Naive Bayes Classifier")
    lModelName.append("Decision Tree Classifier")
    lModelName.append("Random Forest Classifier")
    lModelName.append("Logistic Regression Classifier")
# list for model object
lModelObjs = []
if lUniqLbls == 2:
    lModelObjs.append(OneVsRest())
    lModelObjs.append(GBTClassifier())
    lModelObjs.append(LinearSVC())
else:    
    lModelObjs.append(NaiveBayes())
    lModelObjs.append(DecisionTreeClassifier())
    lModelObjs.append(RandomForestClassifier())
    lModelObjs.append(LogisticRegression())
# lists for train  acc
lModelTrnAcc = []
# List for test acc
lModelTstAcc = []
# print
print("\n*** Prepare Models ***")
msg = "%-30s %-30s" % ("Model Name", "Model Object")
print(msg)
# for each model
for i in range(0,len(lModelName)):   
    # print model name, model object 
    msg = "%-30s %-30s" % (lModelName[i], lModelObjs[i])
    print(msg)

# COMMAND ----------

# select best model
print("*** Select Best Model ***")

# for each model
for i in range(0,len(lModelName)):   

    # print
    print("\nModel Name  : ",lModelName[i])
    # instantiate empty model
    algos =  lModelObjs[i]
    # prepare dft_trn
    df_trn = df_trn.drop(col('predict'))
    df_trn = df_trn.drop(col('rawPrediction'))
    df_trn = df_trn.drop(col('probability'))
    # train the model
    model = algos.fit(df_trn)
    # print
    #print("*** ML Model ***")
    print("Model Object: ",model)

    # predict train
    df_trn = model.transform(df_trn)
    df_trn = df_trn.withColumnRenamed("prediction","predict")
    # train data - important cols
    #print("\n*** Train Data - Features Label & Predict ***")
    #df_trn.select("features","label","predict").show(5)
    #df_trn.printSchema()

    # evaluate train based on label check
    if lUniqLbls == 2:
        # for binary classification
        evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
        trn_auc = evl_auc.evaluate(df_trn)
    else:
        # for multi class classification
        evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
        trn_acc = evl_acc.evaluate(df_trn)
    # model train accuracy
    lModelTrnAcc.append(trn_acc)
    # print
    #print("*** Train Data Evaluation ***")
    print("TRN-Accuracy: %3.2f %%" % (trn_acc*100))

    # predict test
    df_tst = df_tst.drop(col('predict'))
    df_tst = df_tst.drop(col('rawPrediction'))
    df_tst = df_tst.drop(col('probability'))
    df_tst = model.transform(df_tst)
    df_tst = df_tst.withColumnRenamed("prediction","predict")
    # train data - important cols
    #print("\n*** Test Data - Features Label & Predict ***")
    #df_tst.select("features","label","predict").show(5)
    #df_tst.printSchema()

    # evaluate test
    if lUniqLbls == 2:
        # for binary classification
        evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
        tst_auc = evl_auc.evaluate(df_tst)
    else:
        # for multi class classification
        evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
        tst_acc = evl_acc.evaluate(df_tst)
    # model test accuracy
    lModelTstAcc.append(tst_acc)
    # print
    #print("*** Test Data Evaluation ***")
    print("TST-Accuracy: %3.2f %%" % (tst_acc*100))
print("Done ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The best classification algorithm is still Logistic Regression (Multinomial CLassification) at 98.75%

# COMMAND ----------

# summary
print("\n*** Model Summary ***")
# header
msg = "%-30s %10s %8s" % ("Model", "Train-Acc", "  Test-Acc")
print(msg)
# for each model
for i in range(0,len(lModelName)):   
    # print model name, rsqr & rmse   
    msg = "%-30s %10.7f %10.7f" % (lModelName[i], lModelTrnAcc[i], lModelTstAcc[i])
    print(msg)

# COMMAND ----------

# import
import builtins
# find model with the best accuracy details
print("\n*** Best Accuracy Model ***")
vIndex = lModelTstAcc.index(builtins.max(lModelTstAcc))
print("Index      : ",vIndex)
print("Model Name : ",lModelName[vIndex])
print("TrnAccuracy: ",lModelTrnAcc[vIndex])
print("TstAccuracy: ",lModelTstAcc[vIndex])

# COMMAND ----------

# imports
from pyspark.sql.functions import col
# preparing final model from best
print("\n*** Preparing Best Model ***")
# select algo of best model
algos =  lModelObjs[vIndex]
# prepare dft_trn
df_trn = df_trn.drop(col('predict'))
df_trn = df_trn.drop(col('rawPrediction'))
df_trn = df_trn.drop(col('probability'))
model = algos.fit(df_trn)
print("Done ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Evaluation using Test Data

# COMMAND ----------

# imports
from pyspark.sql.functions import col

# retrain model object using the best model
model = algos.fit(df_trn)
# predict test
df_tst = df_tst.drop(col('predict'))
df_tst = df_tst.drop(col('rawPrediction'))
df_tst = df_tst.drop(col('probability'))
df_tst = model.transform(df_tst)
df_tst = df_tst.withColumnRenamed("prediction","predict")
# train data - important cols
print("\n*** Test Data - Features Label & Predict ***")
df_tst.select("features","label","predict").show(5)

# COMMAND ----------

# evaluate test

# evaluate text based on label check
if lUniqLbls == 2:
    # for binary classification
    evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
    tst_auc = evl_auc.evaluate(df_tst)
else:
    # for mutilclass
    evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
    tst_acc = evl_acc.evaluate(df_tst)

# print
print("*** Test Data Evaluation ***")
print("TST-Accuracy : %3.2f %%" % (tst_acc*100))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Even the test accuracy is 98.73% showing that the model performs well in classification

# COMMAND ----------

# visualize test accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# get label & predict column
y_orig = df_tst.select(['label']).collect()
y_pred = df_tst.select(['predict']).collect()
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_orig, y_pred)*100
print("%3.2f %%" % accuracy)
# confusion matrix
from sklearn.metrics import confusion_matrix
print("\n*** Confusion Matrix - Original ***")
cm = confusion_matrix(y_orig, y_orig)
print(cm)
# confusion matrix predicted
print("\n*** Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_orig, y_pred)
print(cm)
# confusion matrix predicted
print("\n*** Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_orig, y_pred)
print(cm)
# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_orig, y_pred)
print(cr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction using new data

# COMMAND ----------


# read csv
print("\n*** Read CSV ***")
df_new = spark.read.format("csv").option("inferSchema", "true").option("header", "true").option("sep", ",")  \
  .load("/FileStore/tables/newobesedata.csv")
# top 5 rows
display(df_new)

# COMMAND ----------

# rows & cols
print("\n*** New Data - Rows & Cols ***")
print("Rows: ",df_new.count())
print("Cols: ",len(df_new.columns))

# get string cols
print("\n*** New Data - String Cols ***")
print(getStringCols(df_new))


# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "Gender"
sigender = StringIndexer(inputCol=colName, outputCol="ICol")
mogender = sigender.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mogender)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "family_history_with_overweight"
sihist = StringIndexer(inputCol=colName, outputCol="ICol")
mohist = sihist.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mohist)


# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "FAVC"
sifavc = StringIndexer(inputCol=colName, outputCol="ICol")
mofavc = sifavc.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mofavc)


# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "CAEC"
sicaec = StringIndexer(inputCol=colName, outputCol="ICol")
mocaec = sicaec.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mocaec)


# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "SMOKE"
sismoke = StringIndexer(inputCol=colName, outputCol="ICol")
mosmoke = sismoke.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mosmoke)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "SCC"
siscc = StringIndexer(inputCol=colName, outputCol="ICol")
moscc = siscc.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, moscc)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "CALC"
sicalc = StringIndexer(inputCol=colName, outputCol="ICol")
mocalc = sicalc.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, mocalc)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "MTRANS"
sitrans = StringIndexer(inputCol=colName, outputCol="ICol")
motrans = sitrans.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, motrans)

# COMMAND ----------

#### use as rquired
# convert string categoric column to float catgorics
# colName
colName = "NObeyesdad"
siobeys = StringIndexer(inputCol=colName, outputCol="ICol")
moobeys = siobeys.fit(df_new)
print("\n*** New Data - Char to Numeric - %s ***" % colName)
df_new = CharToNum(df_new, colName, moobeys)

# COMMAND ----------

# check sulls
print("\n*** New Data - Check Null Values ***") 
print(checkNulls(df_new))


# COMMAND ----------

# change as required
# data preparation
# rename class vars to 'label'
df_new = df_new.withColumnRenamed("NObeyesdad","label")
# set ML columns
allCols = df_new.columns
clsCols = "label"
datCols = df_new.columns
datCols.remove(clsCols)
print("*** ML Columns ***")
print("AllCols:", allCols)
print("datCols:",datCols)
print("clsCols:",clsCols)

# COMMAND ----------


# prediction data - create vector column
assembler = VectorAssembler(inputCols=datCols,outputCol="features")
dfn_new = assembler.transform(df_new)
print("\n*** New Data - Label & Features Col ***")
dfn_new.show(5)

# COMMAND ----------

dfn_new = df_tst

# COMMAND ----------


# new data - predict
from pyspark.sql.functions import col
dfn_new = dfn_new.drop(col('predict'))
dfn_new = dfn_new.drop(col('rawPrediction'))
dfn_new = dfn_new.drop(col('probability'))
dfn_new = model.transform(dfn_new)
dfn_new = dfn_new.withColumnRenamed("prediction","predict")
# if depVars available in dataframe, show depVars & prdVals 
# if depVars NOT available in dataframe, show only prdVals 
print("\n*** New Data - Features Label & Predict ***")
dfn_new.select("features","label","predict").show(5)
#dfn_new.select("features","predict").show(5)


# COMMAND ----------

# evaluate predict if available

# for binary classification
#evl_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="predict")
#new_auc = evl_auc.evaluate(dfn_new)

# accuracy
evl_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predict", metricName="accuracy")
new_acc = evl_acc.evaluate(dfn_new)

# print
print("*** New Data Evaluation ***")
print("NEW-Accuracy : %3.2f %%" %  (new_acc*100))

# COMMAND ----------

# visualize new accuracy
# get label & predict column
y_orig = dfn_new.select(['label']).collect()
y_pred = dfn_new.select(['predict']).collect()
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_orig, y_pred)*100
print("%3.2f %%" % accuracy)
# confusion matrix
from sklearn.metrics import confusion_matrix
print("\n*** New Data - Confusion Matrix - Original ***")
cm = confusion_matrix(y_orig, y_orig)
print(cm)
# confusion matrix predicted
print("\n*** New Data - Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_orig, y_pred)
print(cm)
# classification report
print("\n*** New Data - Classification Report ***")
cr = classification_report(y_orig, y_pred)
print(cr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The accuracy on new data along with test and train data is 99% which shows excellent performance on all types of data showing its effectiveness and robustness

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insights:-
# MAGIC ### 1. The best classification algorithm in all the passes was Logistic Regression with an accuracy of 98.73%
# MAGIC ### 2. The next best classification algorithm was Random Forest with an accuracy score of 87.8%
# MAGIC ### 3. The standardisation pass destroyed the quality of data so no classifcation algorithm could be performed because certail variables like Age,Weight cannot be negative
# MAGIC ### 4. All the target varibles were well balanced in this dataset showing good dataset collection efforts
# MAGIC ### 5.Most of the people were classified as overweight or obese (almost 85%) showing that our habits and lifestyle are big contributor to our health detoriation
# MAGIC ### 6.A wide coverage of variables starting from age,weight,height spanning to taking public transport,favourite foods to smoking was taken into account the whole persona while evaluating obeseness
# MAGIC ### 7 Last but not least all the cells were filled with data and outliers were not extreme making it easy to work with the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ------------------------- The End ------------------------

# COMMAND ----------


