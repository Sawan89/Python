# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:04:16 2016

@author: Sawan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binned_statistic

#Regression output
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

os.chdir('C:\\Users\\Sawan\\Documents\\R Python course\\OBJ-Oriented Prog App\\content')
os.getcwd()

hospital_data = pd.read_table('salian_sawan_export', sep='\t')
hospital_data.columns
hospital_data.dtypes


hospital_data2 = hospital_data.ix[:,['Teaching' ,'DonorType' , 'Gender', 'PositionTitle', 'Compensation', 'TypeControl','OperRev','OperInc','AvlBeds', 'NetPatRev','InOperExp','OutOperExp']]
hospital_data2.dropna()
hospital_data['Teaching'].max()

hospital_data2.Compensation.unique()

hospital_data3 = hospital_data.ix[:,['OperRev','OperInc']]

#Create Dummy Variables.
teaching_dummy = pd.get_dummies(hospital_data2['Teaching'])
teaching_dummy.head()
hospital_data3 = hospital_data3.join(teaching_dummy)

donorType_dummy = pd.get_dummies(hospital_data2['DonorType'])
donorType_dummy.head()
hospital_data3 = hospital_data3.join(donorType_dummy)

gender_dummy = pd.get_dummies(hospital_data2['Gender'])
gender_dummy.head()
hospital_data3 = hospital_data3.join(gender_dummy)

positionTitle_dummy = pd.get_dummies(hospital_data2['PositionTitle'])
positionTitle_dummy.head()
hospital_data3 = hospital_data3.join(positionTitle_dummy)

compensation_dummy = pd.get_dummies(hospital_data2['Compensation'])
compensation_dummy.head()
hospital_data3 = hospital_data3.join(compensation_dummy)

typeControl_dummy = pd.get_dummies(hospital_data2['TypeControl'])
typeControl_dummy.head()
hospital_data3 = hospital_data3.join(typeControl_dummy)

hospital_data3.dtypes

#Binning AvlBeds
hospital_data2['AvlBeds'].max()
hospital_data2['AvlBeds'].min()


bin_interval = [12, 42, 72, 102, 430, 909]
bin_counts,bin_edges,binnum = binned_statistic(hospital_data2['AvlBeds'],hospital_data2['AvlBeds'],statistic='count',bins=bin_interval)


binlabels = ['AvlBeds_12_42', 'AvlBeds_42_72', 'AvlBeds_72_102', 'AvlBeds_102_430', 'AvlBeds_430_909']
AvlBeds_categ = pd.cut(hospital_data2['AvlBeds'], bin_interval, right=False, retbins=False, labels=binlabels)
AvlBeds_categ.name = 'AvlBeds_categ'
hospital_data3 = hospital_data3.join(pd.DataFrame(AvlBeds_categ))

#Converting to Dummy variable.
AvlBeds_dummy = pd.get_dummies(hospital_data3['AvlBeds_categ'])
AvlBeds_dummy.head()
hospital_data3 = hospital_data3.join(AvlBeds_dummy)

#Renaming
hospital_data3.columns = ['OperRev', 'OperInc', 'Teaching_Small_Rural', 'Teaching_Teaching', 'DonorType_Alumni', 'DonorType_Charity', 'Gender_F', 'Gender_M', 'PositionTitle_ActingDirector', 'PositionTitle_RegionalRepresentative', 'PositionTitle_SafetyInspectionMember', 'PositionTitle_StateBoardRepresentative','Compensation_23987', 'Compensation_46978', 'Compensation_89473', 'Compensation_248904', 'TypeControl_CityCounty', 'TypeControl_District', 'TypeControl_Investor', 'TypeControl_NonProfit', 'AvlBeds_categ', 'AvlBeds_12_42', 'AvlBeds_42_72', 'AvlBeds_72_102', 'AvlBeds_102_430', 'AvlBeds_430_909']

#Regression model 1
reg1 =smf.ols('OperRev ~ Teaching_Small_Rural + Teaching_Teaching + DonorType_Alumni+ DonorType_Charity+ Gender_F+ Gender_M+ PositionTitle_ActingDirector+ PositionTitle_RegionalRepresentative+ PositionTitle_SafetyInspectionMember+ PositionTitle_StateBoardRepresentative+Compensation_23987+ Compensation_46978+ Compensation_89473+ Compensation_248904+ TypeControl_CityCounty+ TypeControl_District+ TypeControl_Investor+ TypeControl_NonProfit+ AvlBeds_categ+ AvlBeds_12_42+ AvlBeds_42_72+ AvlBeds_72_102+ AvlBeds_102_430+ AvlBeds_430_909', hospital_data3).fit()
reg1.summary()

#Regression model 2
reg2 =smf.ols('OperInc ~ Teaching_Small_Rural + Teaching_Teaching + DonorType_Alumni+ DonorType_Charity+ Gender_F+ Gender_M+ PositionTitle_ActingDirector+ PositionTitle_RegionalRepresentative+ PositionTitle_SafetyInspectionMember+ PositionTitle_StateBoardRepresentative+Compensation_23987+ Compensation_46978+ Compensation_89473+ Compensation_248904+ TypeControl_CityCounty+ TypeControl_District+ TypeControl_Investor+ TypeControl_NonProfit+ AvlBeds_categ+ AvlBeds_12_42+ AvlBeds_42_72+ AvlBeds_72_102+ AvlBeds_102_430+ AvlBeds_430_909', hospital_data3).fit()
reg2.summary()



