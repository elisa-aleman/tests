#-*- coding: utf-8 -*-
#!python3 

import sqlite3
import os.path
import sys
PythonPath = os.path.join(os.path.expanduser('~'), 'Master','python','libraries')
sys.path.append(os.path.abspath(PythonPath))
from ProjectMethods import *
from StanfordNLP import *
import sys

version = sys.version

# sentence = "但 购物 一般 四 之 谷 附近 有 个 超市 赤坂 见 附 附近 只有 吃饭 的 地方 和 一"
# sentence_list = sentence.split()
# text_err = tagger.tag(sentence_list)

#Has some errors
#[('', u'\u4f46#AD'), ('', u'\u8d2d\u7269#NN'), ('', u'\u4e00\u822c#AD'), ('', u'\u56db#CD'), ('', u'\u4e4b#DEG'), ('', u'\u8c37#NN'), ('', u'\u9644\u8fd1#NN'), ('', u'\u6709#VE'), ('', u'\u4e2a#M'), ('', u'\u8d85\u5e02#NN'), ('', u'\u8d64\u5742#NN'), ('', u'\u89c1#VV'), ('', u'\u9644#NN'), ('', u'\u9644\u8fd1#NN'), ('', u'\u53ea\u6709#AD'), ('', u'\u5403\u996d#VV'), ('', u'\u7684#DEC'), ('', u'\u5730\u65b9#NN'), ('', u'\u548c#CC'), ('', u'\u4e00#CD')]
# 但#AD ... tag is stuck to text in second part of tuple

# text = [tuple(i[1].split('#')) for i in text_err]
# print(text)

#[(u'\u4f46', u'AD'), (u'\u8d2d\u7269', u'NN'), (u'\u4e00\u822c', u'AD'), (u'\u56db', u'CD'), (u'\u4e4b', u'DEG'), (u'\u8c37', u'NN'), (u'\u9644\u8fd1', u'NN'), (u'\u6709', u'VE'), (u'\u4e2a', u'M'), (u'\u8d85\u5e02', u'NN'), (u'\u8d64\u5742', u'NN'), (u'\u89c1', u'VV'), (u'\u9644', u'NN'), (u'\u9644\u8fd1', u'NN'), (u'\u53ea\u6709', u'AD'), (u'\u5403\u996d', u'VV'), (u'\u7684', u'DEC'), (u'\u5730\u65b9', u'NN'), (u'\u548c', u'CC'), (u'\u4e00', u'CD')]

#JJ is adjective
#VA is verb-adj usually followed by teki kanji

###


sqlite_file1 = 'SVM_training.sqlite'
sqlite_file2 = 'ctrip_db.sqlite'
log_file_name = 'POSTagged_AdjectivesSearch_log.txt'
adj_file_name = 'POSTagged_Adjectives.txt'
log_file_name2 = 'POSTagged_AdjectivesSearch_log_full_corpus.txt'
adj_file_name2 = 'POSTagged_Adjectives_full_corpus.txt'

#########################################
############## SQL GETs #################
#########################################

def GetSentences(c):
	# minnum = 1
	# c.execute("SELECT max(RID) from Sentences")
	# maxnum = c.fetchone()[0]
	sentences = []
	c.execute("SELECT Sentence FROM 'Sentences'")
	raw = c.fetchall()
	sentences = [i[0][2:-1] for i in raw]
	return sentences

def SearchAdjectives(tagged = True):
	tagger = getTagger()
	if tagged:
		conn, c = Connect(sqlite_file1)
	else:
		conn, c = Connect(sqlite_file2)
	if tagged:
		log_file = MakeLogFile(log_file_name)
		adj_file = MakeDictPath(adj_file_name)
		pos_log = MakeLogFile('POS_Tags_log.txt')
	else:
		log_file = MakeLogFile(log_file_name2)
		adj_file = MakeDictPath(adj_file_name2)
		pos_log = MakeLogFile('POS_Tags_log_full_corpus.txt')
	sentences = GetSentences(c)
	print("Sentences Collected...")
	conn.close()
	adjs = set()
	adjcounts = dict()
	counts = dict()
	ind = 1
	for sentence in sentences:
		post = POSTag(tagger,sentence)
		with open(pos_log, 'a') as logf:
			strlog = str(post)
			strlog+= "\n"
			logf.write(strlog)
		for item in post:
			if item[1] == u'JJ' or item[1] == u'VA':
				adjs.add(item[0])
				if adjcounts.get(item[0]):
					adjcounts[item[0]] += 1
				else:
					adjcounts[item[0]] = 1
			if counts.get(item[0]):
				counts[item[0]] += 1
			else:
				counts[item[0]] = 1
		print("POS Tagging sentence {} of {}         ".format(ind, len(sentences)))
		ind += 1
		up()
	down()
	print("Done Tagging")
	print("")
	adjs_list = list(adjs)
	for adj in adjs_list:
		if version.startswith('2'):
			word = adj.encode('utf-8')
		else:
			word = adj
		adjcount = adjcounts[adj]
		totalcount = counts[adj]
		adjprob = adjcount/(totalcount*1.0)
		strlog = "Word : {}".format(word)
		printSTDlog(strlog, log_file)
		strlog = "Counts as Adjective : {}".format(adjcount)
		printSTDlog(strlog, log_file)
		strlog = "Total Counts : {}".format(totalcount)
		printSTDlog(strlog, log_file)
		strlog = "Probability as Adjective : {0:.2f}".format(adjprob)
		printSTDlog(strlog, log_file)
		strlog = ""
		printSTDlog(strlog, log_file)

if __name__ == "__main__":
	#SearchAdjectives(tagged = False)]
	#test("但 酒店 周边 服务 设施 不 多 有 诸多 不便")