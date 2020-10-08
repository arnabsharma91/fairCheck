#!/usr/bin/env python

import pandas as pd
import csv as cv
from parsimonious.nodes import NodeVisitor
import re
from utils import util
import sys

class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[self.size()-1]

     def size(self):
         return len(self.items)

class InfixConverter:
    
    def __init__(self):
        self.stack = Stack()
        self.precedence = {'+':1, '-':1, 'abs':1, '*':2, '/':2, '^':3}

    def hasLessOrEqualPriority(self, a, b):
        if a not in self.precedence:
            return False
        if b not in self.precedence:
            return False
        return self.precedence[a] <= self.precedence[b]

    def isOperand(self, ch):
        return ch.isalpha() or ch.isdigit() or ch in '.' or ch in '_'

    def isOperator(self, x):
        ops = ['+', '-', 'abs','/', '*']
        return x in ops
    
    def isOpenParenthesis(self, ch):
        return ch == '('

    def isCloseParenthesis(self, ch):
        return ch == ')'

    def toPostfix(self, expr):
        expr = expr.replace(" ", "")
        self.stack = Stack()
        output = ' '

        for c in expr:
            if self.isOperand(c):
                output += c
            else:
                if self.isOpenParenthesis(c):
                    output += " "
                    output += ')'
                    
                    self.stack.push(c)
                elif self.isCloseParenthesis(c):
                    operator = self.stack.pop()
                    output += " "
                    output += '('
                    
                    while not self.isOpenParenthesis(operator):
                        output += " "
                        output += operator
                        operator = self.stack.pop() 
                else:
                    while (not self.stack.isEmpty()) and self.hasLessOrEqualPriority(c,self.stack.peek()):
                        output += " "
                        output += self.stack.pop()

                    self.stack.push(c)
                output += " "
        while not self.stack.isEmpty():
            output += self.stack.pop()

        return output
    
    '''
     1. Reverse expression string
     2. Replace open paren with close paren and vice versa
     3. Get Postfix and reverse it
    '''
    def toPrefix(self, expr):
        reverse_expr =''
        for c in expr[::-1]:
            if c == '(':
                reverse_expr += ") "
            elif c == ')':
                reverse_expr += "( "
            else:
                reverse_expr += c+" "

        reverse_postfix = self.toPostfix(reverse_expr)

        return reverse_postfix[::-1]

    def convert(self, expr):
        try:
            result = eval(expr);
        except:
            result = expr
        return self.toPrefix(expr)


class AssumptionVisitor(NodeVisitor):
    
    def __init__(self):
        self.varList = []
        self.classVarList = []
        self.arithOperator = []
        self.logicOperator = ""
        self.numList = []
        self.numEnd = ""
        self.feIndex = 99999
        self.feValue = 0
        self.count = 0
        self.df = pd.read_csv('OracleData.csv')
        self.feArr = []
        self.noOfAttr = self.df.shape[1]
        self.varMapDict = {}
        self.prefix_list = []
        self.varInd = False
        self.arrFlag = False
    
    def generic_visit(self, node, children):
        pass
        
    def visit_arith_op(self, node, children):
        self.arithOperator.append(node.text)
        
    def visit_logic_op(self, node, children):
        if '!=' in node.text:
            self.logicOperator = 'not(='
        else:
            self.logicOperator = node.text
        
    def visit_number(self, node, children):
        self.numList.append(node.text)
    
    def visit_classVar(self, node, children):
        if self.varInd:
            raise Exception("Feature indexes given twice")
            sys.exit(1)
        self.classVarList.append(node.text)
        self.checkIndexConstncy()
        self.feIndex = int(re.search(r'\d+', node.text).group(0))
        self.checkValFeIndex()
        
    def visit_classVarArr(self, node, children): 
        self.classVarList.append(node.text)
                
    def visit_num_log(self, node, children):
        self.numEnd = node.text
    
    def visit_value(self, node, children):
        self.feValue = float(node.text)
    
    def visit_expr_dist1(self, node, children):
        expr_dist1 = str(node.text)
        self.getPrefixExp(expr_dist1)
    
    def visit_expr1(self, node, children):
        self.expr2logic(self.prefix_list)
        
    def visit_expr_dist2(self, node, children):
        expr_dist2 = str(node.text)
        self.getPrefixExp(expr_dist2)
    
    def visit_expr2(self, node, children):
        self.expr2logic(self.prefix_list)
    
    def visit_expr3(self, node, children):
        temp_expr = node.text
        self.replaceIndex(temp_expr)
        f = open('assumeStmnt.txt', 'a')
        f.write('\n')
        f.write("(assert ("+self.logicOperator+" "+str(self.df.columns.values[self.feIndex]+str(0))+
                " "+str(self.feValue)+"))")
        
        if self.logicOperator == 'not(=':
            f.write(')')
        f.write('\n')
        f.close()
    
    def visit_expr4(self, node, children):
        temp_expr = node.text
        self.replaceIndex(temp_expr)
        f = open('assumeStmnt.txt', 'a')
        f.write('\n')
        f.write("(assert ("+self.logicOperator+" "+str(self.df.columns.values[self.feIndex]+str(0))+
                " "+str(self.feValue)+"))")
        if self.logicOperator == 'not(=':
            f.write(')')
        f.write('\n')
        f.close()
        
    def visit_expr5(self, node, children):
        temp_expr = node.text
        self.replaceIndex(temp_expr)
        f = open('assumeStmnt.txt', 'a')
        f.write('\n')
        f.write("(assert ("+self.logicOperator+" "+str(self.df.columns.values[self.feIndex]+str(0))+" "+
                str(self.df.columns.values[self.feIndex]+str(1))+"))")
        if self.logicOperator == 'not(=':
            f.write(')')
        f.write('\n')
        f.close()
    
    def visit_expr6(self, node, children):
        if self.arrFlag:
            pass
        else:    
            temp_expr = node.text
            self.replaceIndex(temp_expr)
            f = open('assumeStmnt.txt', 'a')
            f.write('\n')
            f.write("(assert ("+self.logicOperator+" "+str(self.df.columns.values[self.feIndex]+str(0))+" "+
                str(self.df.columns.values[self.feIndex]+str(1))+"))")
            if self.logicOperator == 'not(=':
                f.write(')')
            f.write('\n')    
            f.close()

    def visit_expr7(self, node, children):
        self.varMapDict['no_assumption'] = 'True'
        self.storeMapping()


    def replaceIndex(self, temp_expr):
        for i in range(0, len(self.classVarList)):
            self.varList.append(self.classVarList[i].split('[', 1)[0])
        
        for i in range(0, len(self.classVarList)):
            if self.classVarList[i] in temp_expr:
                temp_expr=temp_expr.replace(str(self.classVarList[i]), str(self.df.columns.values[self.feIndex]
                                                                             +str(i)))
                for j in range(0, len(self.varList)):
                    if self.varList[j] in self.classVarList[i]:
                        self.varMapDict[self.varList[j]] = str(self.df.columns.values[self.feIndex]+str(i))
                        self.feArr.append(self.df.columns.values[self.feIndex]+str(i))
        self.varMapDict['no_assumption'] = False                
        self.storeMapping()

    def getMapping(self):
        return self.varMapDict
    
    def storeInd(self, index):
        self.varInd = True
        self.feIndex = index
        self.checkValFeIndex()
        
    def storeArr(self, arr):
        self.valueArr = arr
        self.arrFlag = True
    
    def getPrefixExp(self, temp_expr1):
        for i in range(0, len(self.classVarList)):
            self.varList.append(self.classVarList[i].split('[', 1)[0])
        
        for i in range(0, len(self.classVarList)):
            if self.classVarList[i] in temp_expr1:
                temp_expr1=temp_expr1.replace(str(self.classVarList[i]), str(self.df.columns.values[self.feIndex]
                                                                             +str(i)))
                for j in range(0, len(self.varList)):
                    if self.varList[j] in self.classVarList[i]:
                        self.varMapDict[self.varList[j]] = str(self.df.columns.values[self.feIndex]+str(i))
                        self.feArr.append(self.df.columns.values[self.feIndex]+str(i))
        
        self.varMapDict['no_assumption'] = False
        self.storeMapping()
        prefix_obj = InfixConverter()
        prefix_expr = prefix_obj.convert(temp_expr1)
        self.prefix_list = util.String2List(prefix_expr)

    def storeMapping(self):
        
        if self.arrFlag:
            self.varMapDict['no_mapping'] = 'True'
            self.varMapDict['no_assumption'] = 'False'
        else:
            self.varMapDict['no_mapping'] = 'False'
        try:
            with open('dict.csv', 'w') as csv_file:  
                writer = cv.writer(csv_file)
                for key, value in self.varMapDict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")
        
    def expr2logic(self, prefix_list):
        abs_flag = False
        f = open('assumeStmnt.txt', 'a')
        f.write('\n \n')
        f.write("(assert ("+self.logicOperator+" ")
        count_par = 2
        if self.logicOperator == 'not(=':
            count_par += 1
        
        for el in prefix_list:
            for op in self.arithOperator:
                if el == 'abs':
                    abs_flag = True
                    for i in range(0, self.df.shape[1]):
                        temp = str(self.df.columns.values[i])
                        if temp in self.feArr[0]:
                            feature = self.df.columns.values[i]
                            break       
                    if 'float' in str(self.df.dtypes[feature]):
                        f.write("("+'absoluteReal'+" ")
                    else:
                        f.write("("+'absoluteInt'+" ")
                    count_par += 1
                    break
                if op in el:
                    f.write("("+op+" ")
                    count_par += 1
            for op in self.numList:
                if op in el:
                    f.write(op+" ")
            for op in self.feArr:
                if op in el:
                    f.write(op+" ")
            if el == ')':
                if abs_flag:
                    count_par -= 2
                    f.write('))')    
                else:
                    count_par -= 1
                    f.write(')')
        if len(self.arithOperator) > 1:
            f.write(") ")
            count_par -= 1
        f.write(self.numEnd) 
        while count_par >= 1:
            f.write(")")
            count_par -=1
            
        f.write('\n')            
        f.close()        
                
    def checkValFeIndex(self):    
        if self.feIndex > self.noOfAttr:
            raise Exception("Feature Index exceed maximum no. Of features in the data")
            sys.exit(1)
    
    def checkIndexConstncy(self):
        digit1 = int(re.search(r'\d+', self.classVarList[0]).group(0))
        if len(self.classVarList) > 1:
            digit2 = int(re.search(r'\d+', self.classVarList[1]).group(0))
            if digit1 != digit2:
                raise Exception("Feature Indexes don't match")
                sys.exit(1)

