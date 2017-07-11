#-*- coding:utf-8 -*-
import random
import json
import math
import numpy
import time
#随机划分训练集和测试集
def splitDataSet(dataset , splitRatio):
	trainSize = int (len(dataset)*splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet)<trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet , copy]

#加载json格式的数据集
def loadDataSet(dataFile):
	weiboset = []
	file = open(dataFile,'r')
	for line in file.readlines():
		data = json.loads(line)
		weiboset.append(data)
	return weiboset

#计算特征为连续数值的特征的均值和标准差
def calParameter(dataset,featureName,parDict):
	numpyArray = numpy.array(dataset)
	numpyMean = numpyArray.mean()
	parDict[featureName+'Mean'] = numpyMean
	numpyStd = numpyArray.std()
	parDict[featureName+'Std'] = numpyStd
	return parDict

#将数据中的时间字符串转换为数值类型
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

#在训练集中提取特征
def featureSelection(dataset,totalSampleSize):
	parameter = {}
	feature_followerRatio = []
	feature_bifollowerRatio = []
	feature_createfrequence = []
	feature_favouritesCount = []
	allow_all_act_msg = 0
	description = 0
	geo_enabled = 0
	#该类型占总训练数目的概率
	sampleSize = len(dataset)
	ratio = float(sampleSize)/float(totalSampleSize)
	parameter['ratio'] = ratio
	#逐条数据进行指定特征的抽取和处理
	for data in dataset:
		feature_followerRatio.append (float(data['user_info']['followers_count'])/(float(data['user_info']['followers_count']+data['user_info']['friends_count'])))
		feature_bifollowerRatio.append (float(data['user_info']['bi_followers_count'])/float(min(data['user_info']['followers_count'],data['user_info']['friends_count'])+1))
		try:
			feature_createfrequence.append (float(data['user_info']['statuses_count'])/(transTime(data['weibo_info'][0]['created_at'])-transTime(data['user_info']['created_at'])))
		except IndexError:
			feature_createfrequence.append (float(0))
		feature_favouritesCount.append (float(data['user_info']['favourites_count']))
		if data['user_info']['allow_all_act_msg'] == False :
			allow_all_act_msg += 1
		if data['user_info']['geo_enabled'] == False :
			geo_enabled += 1
		if data['user_info']['description'] is None :
			description += 1

	#计算数值连续型特征服从高斯分布的参数
	calParameter(feature_followerRatio,'followerRatio',parameter)
	calParameter(feature_bifollowerRatio,'bifollowerRatio',parameter)
	calParameter(feature_createfrequence,'createfrequence',parameter)
	calParameter(feature_favouritesCount,'favouritesCount',parameter)
	#计算数值离散型特征的概率
	parameter['allow_all_act_msg_False'] = float(allow_all_act_msg+1)/float(len(dataset)+2)
	parameter['allow_all_act_msg_True'] = 1 - float(allow_all_act_msg+1)/float(len(dataset)+2)
	parameter['geo_enabled_False'] = float(geo_enabled+1)/float(len(dataset)+2)
	parameter['geo_enabled_True'] = 1 - float(geo_enabled+1)/float(len(dataset)+2)
	parameter['description_False'] = float(description+1)/float(len(dataset)+2)
	parameter['description_True'] = 1 - float(description+1)/float(len(dataset)+2)

	#有训练集得到的朴素贝叶斯算法的参数集
	return parameter

#朴素贝叶斯分类器建模
def classifier(trainSet):
	negative=[]
	positive=[]
	negParameter = {}
	posParameter = {}
	#将训练集根据人工标注分类，分为垃圾账户和正常账户两类
	for data in trainSet:
		if data['label'] == 0:
			negative.append(data)
		else:
			positive.append(data)
	totalSampleSize = len(trainSet)
	#分别对两个类型提取特征并得到参数集
	negParameter = featureSelection(negative,totalSampleSize)
	posParameter = featureSelection(positive,totalSampleSize)
	print (negParameter)
	print (posParameter)
	return [negParameter,posParameter]

#高斯分布计算函数
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#单挑数据测试用例
def testClassifier(data,negParameter,posParameter):
	negCalResult = []
	posCalResult = []
	testParameterCon = {}
	testParameterDis = {}
	#处理数值连续的特征
	testParameterCon['followerRatio'] = float(data['user_info']['followers_count'])/(float(data['user_info']['followers_count']+data['user_info']['friends_count']))
	testParameterCon['bifollowerRatio'] = float(data['user_info']['bi_followers_count'])/float(min(data['user_info']['followers_count'],data['user_info']['friends_count'])+1)
	try:
		testParameterCon['createfrequence'] = float(data['user_info']['statuses_count'])/(transTime(data['weibo_info'][0]['created_at'])-transTime(data['user_info']['created_at']))
	except IndexError:
		testParameterCon['createfrequence'] = 0
	testParameterCon['favouritesCount'] = float((data['user_info']['favourites_count']))

	for key,value in testParameterCon.items():
		negCalResult.append(calculateProbability(value,negParameter[key+'Mean'],negParameter[key+'Std']))
		posCalResult.append(calculateProbability(value,posParameter[key+'Mean'],posParameter[key+'Std']))

	#处理数值离散的特征
	testParameterDis['allow_all_act_msg'] = data['user_info']['allow_all_act_msg']
	if data['user_info']['description'] is None :
		testParameterDis['description'] = False
	else:
		testParameterDis['description'] = True
	testParameterDis['geo_enabled'] = data['user_info']['geo_enabled']

	for key,value in testParameterDis.items():
		negCalResult.append(negParameter[key+'_'+str(value)])
		posCalResult.append(posParameter[key+'_'+str(value)])

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
		return 1
	else:
		print ('error!'+'this weibo\'label is:'+str(data['label']))
		return 0

def main():
	negParameter = {}
	posParameter = {}
	trueNum = 0
	weiboDataSet = loadDataSet('weibo_users.json')
	testData = loadDataSet('bayes.json')
	splitRatio = 0.67
	trainSet , testSet = splitDataSet(weiboDataSet,splitRatio)
	negParameter,posParameter = classifier(trainSet)
	for data in testSet:
		if testClassifier(data,negParameter,posParameter) == 1:
			trueNum += 1
	print ('识别正确率为:{0}%').format(float(trueNum)/len(testSet)*100)

if __name__ == "__main__":
	main()



