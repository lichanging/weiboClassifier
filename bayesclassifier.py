import random
import json
import math
import numpy
import time

def splitDataSet(dataset , splitRatio):
	trainSize = int (len(dataset)*splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet)<trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet , copy]

def loadDataSet(dataFile):
	weiboset = []
	file = open(dataFile,'r')
	for line in file.readlines():
		data = json.loads(line)
		weiboset.append(data)
	return weiboset

def calParameter(dataset,featureName,parDict):
	numpyArray = numpy.array(dataset)
	numpyMean = numpyArray.mean()
	parDict[featureName+'Mean'] = numpyMean
	numpyStd = numpyArray.std()
	parDict[featureName+'Std'] = numpyStd
	return parDict



def transTime(s):
    month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    result = []
    result.append(s[-4:])
    result.append(str(month_dict[s[4:7]]))
    result.append(s[8:10])
    result.append(s[11:13])
    result.append(s[14:16])
    result.append(s[17:19])
    s = ':'.join(result)
    timeArray = time.strptime(s,"%Y:%m:%d:%H:%M:%S")
    return time.mktime(timeArray)/3600

def featureSelection(dataset,totalSampleSize):
	parameter = {}
	feature_followerRatio = []
	feature_bifollowerRatio = []
	feature_createfrequence =[]
	sampleSize = len(dataset)
	ratio = float(sampleSize)/float(totalSampleSize)
	parameter['ratio'] = ratio
	for data in dataset:
		feature_followerRatio.append (float(data['user_info']['followers_count'])/(float(data['user_info']['followers_count']+data['user_info']['friends_count'])))
		feature_bifollowerRatio.append (float(data['user_info']['bi_followers_count'])/float(min(data['user_info']['followers_count'],data['user_info']['friends_count'])+1))
		try:
			feature_createfrequence.append (float(data['user_info']['statuses_count'])/(transTime(data['weibo_info'][0]['created_at'])-transTime(data['user_info']['created_at'])))
		except IndexError:
			feature_createfrequence.append (float(0))
	calParameter(feature_followerRatio,'followerRatio',parameter)
	calParameter(feature_bifollowerRatio,'bifollowerRatio',parameter)
	calParameter(feature_createfrequence,'createfrequence',parameter)
	return parameter


def classifier(trainSet):
	negative=[]
	positive=[]
	negParameter = {}
	posParameter = {}
	for data in trainSet:
		if data['label'] == 0:
			negative.append(data)
		else:
			positive.append(data)
	totalSampleSize = len(trainSet)
	negParameter = featureSelection(negative,totalSampleSize)
	posParameter = featureSelection(positive,totalSampleSize)
	print (negParameter)
	print (posParameter)
	return [negParameter,posParameter]

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def testClassifier(data,negParameter,posParameter):
	negCalResult = []
	posCalResult = []
	followerRatio = float(data['user_info']['followers_count'])/(float(data['user_info']['followers_count']+data['user_info']['friends_count']))
	bifollowerRatio = float(data['user_info']['bi_followers_count'])/float(min(data['user_info']['followers_count'],data['user_info']['friends_count'])+1)
	try:
		createfrequence = float(data['user_info']['statuses_count'])/(transTime(data['weibo_info'][0]['created_at'])-transTime(data['user_info']['created_at']))
	except IndexError:
		createfrequence = 0
	negCalResult.append(calculateProbability(followerRatio,negParameter['followerRatioMean'],negParameter['followerRatioStd']))
	negCalResult.append(calculateProbability(followerRatio,negParameter['bifollowerRatioMean'],negParameter['bifollowerRatioStd']))
	negCalResult.append(calculateProbability(followerRatio,negParameter['createfrequenceMean'],negParameter['createfrequenceStd']))
	posCalResult.append(calculateProbability(followerRatio,posParameter['followerRatioMean'],posParameter['followerRatioStd']))
	posCalResult.append(calculateProbability(followerRatio,posParameter['bifollowerRatioMean'],posParameter['bifollowerRatioStd']))
	posCalResult.append(calculateProbability(followerRatio,posParameter['createfrequenceMean'],posParameter['createfrequenceStd']))
	classifierLabel = 0
	posValue = 1 
	negValue = 1

	for i in negCalResult:
		negValue *= i

	for j in posCalResult:
		posValue *= j
   
	if negValue > posValue:
		classifierLabel = 0
	else:
		classifierLabel = 1

	if data['label']==classifierLabel:
		print ('bingo!'+'this weibo\'label is:'+str(classifierLabel))
	else:
		print ('error!'+'calssiferLable is:'+str(classifierLabel))
	print negCalResult
	print posCalResult

	


def main():
	negParameter = {}
	posParameter = {}
	weiboDataSet = loadDataSet('weibo_users.json')
	testData = loadDataSet('bayes.json')
	splitRatio = 0.67
	trainSet , testSet = splitDataSet(weiboDataSet,splitRatio)
	negParameter,posParameter = classifier(trainSet)
	
	testClassifier(testData[0],negParameter,posParameter)




if __name__ == "__main__":
	main()



