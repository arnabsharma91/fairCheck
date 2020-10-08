#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import csv as cv
import numpy as np
import random as rd
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
import re
import torch
import os, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import trainDecTree, tree2Logic, ReadZ3Output, processCandCex, util, assume2logic, assert2logic
from joblib import dump, load
import time
import PytorchDNNStruct


class generateData:

    def __init__(self, feNameArr, feTypeArr, minValArr, maxValArr):
        self.nameArr = feNameArr
        self.typeArr = feTypeArr
        self.minArr = minValArr
        self.maxArr = maxValArr

    # Function to generate a new sample
    def funcGenData(self):
        tempData = np.zeros((1, len(self.nameArr)), dtype=object)
        f = open('MUTWeight.txt', 'r')
        weight_content = f.readline()
        for k in range(0, len(self.nameArr)):
            fe_type = ''
            fe_type = self.typeArr[k]

            if 'int' in fe_type:
                if weight_content == 'False':
                    tempData[0][k] = rd.randint(self.minArr[k], self.maxArr[k])
                else:
                    tempData[0][k] = rd.randint(-99999999999, 9999999999999999)
            else:
                if weight_content == 'False':
                    tempData[0][k] = round(rd.uniform(0, self.maxArr[k]), 1)
                else:
                    tempData[0][k] = round(rd.uniform(-99999999999, 9999999999999), 3)

        return tempData

    # Function to check whether a newly generated sample already exists in the list of samples
    def funcCheckUniq(self, matrix, row):
        row_temp = row.tolist()
        matrix_new = matrix.tolist()
        if row_temp in matrix_new:
            return True
        else:
            return False

    # Function to combine several steps
    def funcGenerateTestData(self):
        tst_pm = 5000
        testMatrix = np.zeros(((tst_pm + 1), len(self.nameArr)), dtype=object)
        feature_track = []
        flg = False

        i = 0
        while i <= tst_pm:
            # Generating a test sample
            temp = self.funcGenData()
            # Checking whether that sample already in the test dataset
            flg = self.funcCheckUniq(testMatrix, temp)
            if not flg:
                for j in range(0, len(self.nameArr)):
                    testMatrix[i][j] = temp[0][j]
                i = i + 1

        with open('TestingData.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(self.nameArr)
            writer.writerows(testMatrix)


class dataFrameCreate(NodeVisitor):

    def __init__(self):
        self.feName = None
        self.feType = None
        self.feMinVal = -99999
        self.feMaxVal = 0

    def generic_visit(self, node, children):
        pass

    def visit_feName(self, node, children):
        self.feName = node.text

    def visit_feType(self, node, children):
        self.feType = node.text

    def visit_minimum(self, node, children):
        digit = float(re.search(r'\d+', node.text).group(0))
        self.feMinVal = digit

    def visit_maximum(self, node, children):
        digit = float(re.search(r'\d+', node.text).group(0))
        self.feMaxVal = digit


class readXmlFile:

    def __init__(self, fileName):
        self.fileName = fileName

    def funcReadXml(self):
        grammar = Grammar(
            r"""
    
            expr             = name / type / minimum / maximum / xmlStartDoc / xmlStartInps / xmlEndInps / xmlStartInp /
                                                                        xmlEndInp / xmlStartValTag /xmlEndValTag
            name             = xmlStartNameTag feName xmlEndNameTag
            type             = xmlStartTypeTag feType xmlEndTypeTag
            minimum          = xmlStartMinTag number xmlEndMinTag
            maximum          = xmlStartMaxTag number xmlEndMaxTag
            xmlStartDoc      = '<?xml version="1.0" encoding="UTF-8"?>'
            xmlStartInps     = "<Inputs>"
            xmlEndInps       = "<\Inputs>"
            xmlStartInp      = "<Input>"
            xmlEndInp        = "<\Input>"
            xmlStartNameTag  = "<Feature-name>"
            xmlEndNameTag    = "<\Feature-name>"
            xmlStartTypeTag  = "<Feature-type>"
            xmlEndTypeTag    = "<\Feature-type>"
            xmlStartValTag   = "<Value>"
            xmlEndValTag     = "<\Value>"
            xmlStartMinTag   = "<minVal>"
            xmlEndMinTag     = "<\minVal>"
            xmlStartMaxTag   = "<maxVal>"
            xmlEndMaxTag     = "<\maxVal>"
            feName           = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
            feType           = ~"[A-Z 0-9]*"i
            number           = ~"[+-]?([0-9]*[.])?[0-9]+"
            """
        )

        with open(self.fileName) as f1:
            file_content = f1.readlines()
        file_content = [x.strip() for x in file_content]
        feNameArr = []
        feTypeArr = []
        minValArr = []
        maxValArr = []
        feName_type = {}
        fe_type = ''
        for lines in file_content:
            tree = grammar.parse(lines)
            dfObj = dataFrameCreate()
            dfObj.visit(tree)

            if dfObj.feName is not None:
                feNameArr.append(dfObj.feName)
                fe_name = dfObj.feName
            if dfObj.feType is not None:
                feTypeArr.append(dfObj.feType)
                fe_type = dfObj.feType
                feName_type[fe_name] = fe_type
            if dfObj.feMinVal != -99999:
                if 'int' in fe_type:
                    minValArr.append(int(dfObj.feMinVal))
                else:
                    minValArr.append(dfObj.feMinVal)
            if dfObj.feMaxVal != 0:
                if 'int' in fe_type:
                    maxValArr.append(int(dfObj.feMaxVal))
                else:
                    maxValArr.append(dfObj.feMaxVal)
        try:
            with open('feNameType.csv', 'w') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in feName_type.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

        genDataObj = generateData(feNameArr, feTypeArr, minValArr, maxValArr)
        genDataObj.funcGenerateTestData()


class makeOracleData:

    def __init__(self, model, train_data, train_data_loc):
        self.model = model
        self.train_data = train_data
        self.train_data_loc = train_data_loc
        if self.train_data:
            if self.train_data_loc == '':
                raise Exception('Please provide the location of the train data')
                sys.exit(1)
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

    def funcGenOracle(self):
        if not self.train_data:
            dfTest = pd.read_csv('TestingData.csv')
        else:
            dfTest = pd.read_csv(self.train_data_loc)
            dfTest.to_csv('TestingData.csv', index=False, header=True)

        dataTest = dfTest.values
        predict_list = np.zeros((1, dfTest.shape[0]))
        X = dataTest[:, :-1]

        if 'numpy.ndarray' in str(type(self.model)):
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                dfTest.loc[i, 'Class'] = int(predict_list[0][i])

        else:
            if self.paramDict['model_type'] == 'Pytorch':
                X = torch.tensor(X, dtype=torch.float32)
                predict_class = []
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = predict_class[i]
            else:
                data_new = np.zeros((dataTest.shape[0], dataTest.shape[1]), dtype=object)
                predict_class = self.model.predict(X)
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = int(predict_class[i])
        dfTest.to_csv('OracleData.csv', index=False, header=True)


class propCheck:

    def __init__(self, max_samples=None, deadline=None, model=None, no_of_params=None, xml_file='',
                 mul_cex=False, model_with_weight=False, train_data_available=False,
                 train_data_loc='',
                 model_type=None, model_path=''):

        self.paramDict = {}
        if max_samples is None:
            self.max_samples = 1000
        else:
            self.max_samples = max_samples
        self.paramDict['max_samples'] = self.max_samples

        if deadline is None:
            self.deadline = 500000
        else:
            self.deadline = deadline
        self.paramDict['deadlines'] = self.deadline
        self.paramDict['white_box_model'] = 'Decision tree'

        if (no_of_params is None) or (no_of_params > 3):
            raise Exception("Please provide a value for no_of_params or the value of it is too big")
        else:
            self.no_of_params = no_of_params
        self.paramDict['no_of_params'] = self.no_of_params
        self.paramDict['mul_cex_opt'] = mul_cex
        self.paramDict['multi_label'] = False

        if xml_file == '':
            raise Exception("Please provide a file name")
        else:
            try:
                self.xml_file = xml_file
            except Exception as e:
                raise Exception("File does not exist")

        f = open('MUTWeight.txt', 'w')
        if not model_with_weight:
            f.write(str(False))
            if model_type == 'sklearn':
                if model is None:
                    if model_path == '':
                        raise Exception("Please provide a classifier to check")
                    else:
                        self.model = load(model_path)
                        self.paramDict['model_path'] = model_path
                        self.paramDict['model_type'] = 'sklearn'

                else:
                    self.paramDict['model_type'] = 'sklearn'
                    self.model = model
                    dump(self.model, 'Model/MUT.joblib')

            elif model_type == 'Pytorch':
                self.paramDict['model_type'] = 'Pytorch'
                self.paramDict['model_path'] = model_path
                self.model = PytorchDNNStruct.Net()
                self.model = torch.load(model_path)
                self.model.eval()
            else:
                raise Exception("Please provide the type of the model (Pytorch/sklearn)")

        # Adjusting for fairness aware test cases
        else:
            dfWeight = pd.read_csv('MUTWeight.csv')
            pred_weight = dfWeight.values
            pred_weight = pred_weight[:, :-1]
            self.model = pred_weight
            f.write(str(True))

        f.close()

        try:
            with open('param_dict.csv', 'w') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in self.paramDict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

        if not train_data_available:
            genData = readXmlFile(self.xml_file)
            genData.funcReadXml()
        gen_oracle = makeOracleData(self.model, train_data_available, train_data_loc)
        gen_oracle.funcGenOracle()


class runChecker:

    def __init__(self):
        self.df = pd.read_csv('OracleData.csv')
        f = open('MUTWeight.txt', 'r')
        self.MUTcontent = f.readline()
        f.close()
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

        if self.MUTcontent == 'False':
            self.model_type = self.paramDict['model_type']
            if 'model_path' in self.paramDict:
                model_path = self.paramDict['model_path']
                if self.model_type == 'Pytorch':
                    self.model = PytorchDNNStruct.Net()
                    self.model = torch.load(model_path)
                    self.model.eval()
                else:
                    self.model = load(model_path)
            else:
                self.model = load('Model/MUT.joblib')
        else:
            dfWeight = pd.read_csv('MUTWeight.csv')
            pred_weight = dfWeight.values
            pred_weight = pred_weight[:, :-1]
            self.model = pred_weight
        with open('TestSet.csv', 'w', newline='') as csvfile:
            fieldnames = self.df.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
        with open('CexSet.csv', 'w', newline='') as csvfile:
            fieldnames = self.df.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])

    def funcCreateOracle(self):
        dfTest = pd.read_csv('TestingData.csv')
        data = dfTest.values
        X = data[:, :-1]
        if self.MUTcontent == 'False':
            if self.paramDict['model_type'] == 'Pytorch':
                X = torch.tensor(X, dtype=torch.float32)
                predict_class = []
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = predict_class[i]
            else:
                predict_class = self.model.predict(X)
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = predict_class[i]
            dfTest.to_csv('OracleData.csv', index=False, header=True)
        else:
            predict_list = np.zeros((1, dfTest.shape[0]))
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                dfTest.loc[i, 'Class'] = int(predict_list[0][i])
            dfTest.to_csv('OracleData.csv', index=False, header=True)

    def funcPrediction(self, X, dfCand, testIndx):
        if self.MUTcontent == 'False':
            if self.model_type == 'Pytorch':
                X_pred = torch.tensor(X[testIndx], dtype=torch.float32)
                predict_prob = self.model(X_pred.view(-1, X.shape[1]))
                return int(torch.argmax(predict_prob))
            else:
                if self.MUTcontent == 'False':
                    return self.model.predict(
                        util.convDataInst(X, dfCand, testIndx))
        else:
            temp_class = np.sign(np.dot(self.model, X[testIndx]))
            if temp_class < 0:
                return 0
            else:
                return temp_class

    def addModelPred(self):
        dfCexSet = pd.read_csv('CexSet.csv')
        dataCex = dfCexSet.values

        if self.MUTcontent == 'False':
            if self.model_type == 'Pytorch':
                X = dataCex[:, :-1]
                X = torch.tensor(X, dtype=torch.float32)
                predict_class = []
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
            else:
                predict_class = self.model.predict(dataCex[:, :-1])
            for i in range(0, dfCexSet.shape[0]):
                dfCexSet.loc[i, 'Class'] = predict_class[i]
        else:
            X = dataCex[:, :-1]
            predict_list = np.zeros((1, dfCexSet.shape[0]))
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                if predict_list[0][i] < 0:
                    predict_list[0][i] = 0
                dfCexSet.loc[i, 'Class'] = predict_list[0][i]
        dfCexSet.to_csv('CexSet.csv', index=False, header=True)

    def runPropCheck(self):
        retrain_flag = False
        MAX_CAND_ZERO = 5
        count_cand_zero = 0
        count = 0
        satFlag = False
        start_time = time.time()

        while count < self.max_samples:
            print('count is:', count)
            tree = trainDecTree.functrainDecTree()
            tree2Logic.functree2LogicMain(tree, self.no_of_params)
            util.storeAssumeAssert('DecSmt.smt2')
            util.addSatOpt('DecSmt.smt2')
            os.system(r"z3 DecSmt.smt2 > FinalOutput.txt")
            satFlag = ReadZ3Output.funcConvZ3OutToData(self.df)
            if not satFlag:
                if count == 0:
                    print('No CEX is found by the checker at the first trial')
                    return 0
                elif (count != 0) and (self.mul_cex == 'True'):
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if round(dfCexSet.shape[0] / self.no_of_params) == 0:
                        print('No CEX is found')
                        return 0
                    print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                    self.addModelPred()
                    return round(dfCexSet.shape[0] / self.no_of_params)
                elif (count != 0) and (self.mul_cex == 'False'):
                    print('No Cex is found after ' + str(count) + ' no. of trials')
                    return 0
            else:
                processCandCex.funcAddCex2CandidateSet()
                processCandCex.funcAddCexPruneCandidateSet(tree)
                processCandCex.funcCheckCex()
                # Increase the count if no further candidate cex has been found
                dfCand = pd.read_csv('Cand-set.csv')
                if round(dfCand.shape[0] / self.no_of_params) == 0:
                    count_cand_zero += 1
                    if count_cand_zero == MAX_CAND_ZERO:
                        if self.mul_cex == 'True':
                            dfCexSet = pd.read_csv('CexSet.csv')
                            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                            if round(dfCexSet.shape[0] / self.no_of_params) > 0:
                                self.addModelPred()
                            return round(dfCexSet.shape[0] / self.no_of_params) + 1
                        else:
                            print('No CEX is found by the checker')
                            return 0
                else:
                    count = count + round(dfCand.shape[0] / self.no_of_params)
                data = dfCand.values
                X = data[:, :-1]
                y = data[:, -1]
                if dfCand.shape[0] % self.no_of_params == 0:
                    arr_length = dfCand.shape[0]
                else:
                    arr_length = dfCand.shape[0] - 1
                testIndx = 0
                while testIndx < arr_length:
                    temp_count = 0
                    temp_store = []
                    temp_add_oracle = []
                    for i in range(0, self.no_of_params):
                        if self.funcPrediction(X, dfCand, testIndx) == y[testIndx]:
                            temp_store.append(X[testIndx])
                            temp_count += 1
                            testIndx += 1
                        else:
                            retrain_flag = True
                            temp_add_oracle.append(X[testIndx])
                            testIndx += 1
                    if temp_count == self.no_of_params:
                        if self.mul_cex == 'True':
                            with open('CexSet.csv', 'a', newline='') as csvfile:
                                writer = cv.writer(csvfile)
                                writer.writerows(temp_store)
                        else:
                            print('A counter example is found, check it in CexSet.csv file: ', temp_store)
                            with open('CexSet.csv', 'a', newline='') as csvfile:
                                writer = cv.writer(csvfile)
                                writer.writerows(temp_store)
                            self.addModelPred()
                            return 1
                    else:
                        util.funcAdd2Oracle(temp_add_oracle)

                if retrain_flag == True:
                    self.funcCreateOracle()

                if (time.time() - start_time) > self.deadline:
                    print("Time out")
                    break

        dfCexSet = pd.read_csv('CexSet.csv')
        if (round(dfCexSet.shape[0] / self.no_of_params) > 0) and (count >= self.max_samples):
            self.addModelPred()
            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
            print('No. of Samples looked for counter example has exceeded the max_samples limit')
        else:
            print('No counter example has been found')


def Assume(*args):
    grammar = Grammar(
        r"""
    
        expr        = expr1 / expr2 / expr3 /expr4 /expr5 / expr6 /expr7
        expr1       = expr_dist1 logic_op num_log
        expr2       = expr_dist2 logic_op num_log
        expr3       = classVar ws logic_op ws value
        expr4       = classVarArr ws logic_op ws value
        expr5       = classVar ws logic_op ws classVar
        expr6       = classVarArr ws logic_op ws classVarArr
        expr7       = "True"
        expr_dist1  = op_beg?abs?para_open classVar ws arith_op ws classVar para_close op_end?
        expr_dist2  = op_beg?abs?para_open classVarArr ws arith_op ws classVarArr para_close op_end?
        classVar    = variable brack_open number brack_close
        classVarArr = variable brack_open variable brack_close
        para_open   = "("
        para_close  = ")"
        brack_open  = "["
        brack_close = "]"
        variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        logic_op    = ws (geq / leq / eq / neq / and / lt / gt) ws
        op_beg      = number arith_op
        op_end      = arith_op number
        arith_op    = (add/sub/div/mul)
        abs         = "abs"
        add         = "+"
        sub         = "-"
        div         = "/"
        mul         = "*"
        lt          = "<"
        gt          = ">"
        geq         = ">="
        leq         = "<="
        eq          = "="
        neq         = "!="
        and         = "&"
        ws          = ~"\s*"
        value       = ~"\d+"
        num_log     = ~"[+-]?([0-9]*[.])?[0-9]+"
        number      = ~"[+-]?([0-9]*[.])?[0-9]+"
        """
    )

    tree = grammar.parse(args[0])
    assumeVisitObj = assume2logic.AssumptionVisitor()
    if len(args) == 3:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.storeArr(args[2])
        assumeVisitObj.visit(tree)
    elif len(args) == 2:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.visit(tree)
    elif len(args) == 1:
        assumeVisitObj.visit(tree)


def Assert(*args):
    grammar = Grammar(
        r"""
    expr        = expr1 / expr2/ expr3
    expr1       = classVar ws operator ws number
    expr2       = classVar ws operator ws classVar
    expr3       = classVar mul_cl_var ws operator ws neg? classVar mul_cl_var
    classVar    = class_pred brack_open variable brack_close
    model_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    class_pred  = model_name classSymbol
    classSymbol = ~".predict"
    brack_open  = "("
    brack_close = ")"
    variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    brack3open  = "["
    brack3close = "]"
    class_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    mul_cl_var  = brack3open class_name brack3close
    operator    = ws (gt/ lt/ geq / leq / eq / neq / and/ implies) ws
    lt          = "<"
    gt          = ">"
    geq         = ">="
    implies     = "=>"
    neg         = "~"
    leq         = "<="
    eq          = "=="
    neq         = "!="
    and         = "&"
    ws          = ~"\s*"
    number      = ~"[+-]?([0-9]*[.])?[0-9]+"
    """
    )

    tree = grammar.parse(args[0])
    assert_visit_obj = assert2logic.AssertionVisitor()
    assert_visit_obj.visit(tree)

    obj_faircheck = runChecker()
    start_time = time.time()
    obj_faircheck.runPropCheck()
    print('time required is', time.time() - start_time)
    os.remove('assumeStmnt.txt')
    os.remove('assertStmnt.txt')
    if os.path.exists('Cand-set.csv'):
        os.remove('Cand-set.csv')
    if os.path.exists('CandidateSet.csv'):
        os.remove('CandidateSet.csv')
    if os.path.exists('CandidateSetInst.csv'):
        os.remove('CandidateSetInst.csv')
    if os.path.exists('CandidateSetBranch.csv'):
        os.remove('CandidateSetBranch.csv')
    os.remove('OracleData.csv')
    os.remove('dict.csv')
    os.remove('param_dict.csv')
    os.remove('TestDataSMT.csv')
    if os.path.exists('TestDataSMTMain.csv'):
        os.remove('TestDataSMTMain.csv')
    #os.remove('feNameType.csv')
    os.remove('TestingData.csv')
    os.remove('TestSet.csv')
    os.remove('DecSmt.smt2')
    if os.path.exists('ToggleBranchSmt.smt2'):
        os.remove('ToggleBranchSmt.smt2')
    if os.path.exists('ToggleFeatureSmt.smt2'):
        os.remove('ToggleFeatureSmt.smt2')
    os.remove('TreeOutput.txt')
    if os.path.exists('SampleFile.txt'):
        os.remove('SampleFile.txt')
    os.remove('FinalOutput.txt')
    os.remove('MUTWeight.txt')
    if os.path.exists('ConditionFile.txt'):
        os.remove('ConditionFile.txt')

