#数据清洗与因素分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = pd.read_csv('ch17_cs_training.csv')  # 假设数据存储在data.csv文件中
print('原始数据概括：')
data.info()

from sklearn.ensemble import RandomForestRegressor
def set_missing(df):
    print('随机森林回归填充0值：')
    process_df = df.iloc[:,[5,0,1,2,3,4,6,7,8,9]]
    known = process_df[process_df['MonthlyIncome'] != 0].values
    unknown = process_df[process_df['MonthlyIncome'] == 0].values
    X = known[:, 1:]
    y = known[:, 0]
    rfr = RandomForestRegressor(random_state=0,n_estimators=200, max_depth=3,n_jobs=-1)
    rfr.fit(X,y)
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    df.loc[df['MonthlyIncome'] == 0, 'MonthlyIncome'] = predicted
    return df

def outlier_processing(df,cname):
    s=df[cname]
    oneQuater=s.quantile(0.25)
    threeQuater=s.quantile(0.75)
    IQR=threeQuater-oneQuater
    lower_limit=oneQuater-1.5*IQR
    upper_limit=threeQuater+1.5*IQR
    df=df[(s>lower_limit) & (s<upper_limit)]
    return df

print('Monthly_Income属性离群点原始分布：')
data[['MonthlyIncome']].boxplot()
plt.savefig('ch17_cs01.png',dpi=300,bbox_inches='tight')
plt.show()
print('删除离群点、填充缺失数据：')
data = outlier_processing(data,'MonthlyIncome')
data = set_missing(data)
print('处理Monthly_Income后数据概况：')
data.info()
data[['MonthlyIncome']].boxplot()
plt.savefig('ch17_cs02.png',dpi=300,bbox_inches='tight')
plt.show()

data = outlier_processing(data,'DebtRatio')
data = outlier_processing(data,'NumberOfOpenCreditLinesAndLoans')
data = outlier_processing(data,'NumberRealEstateLoansOrLines')
data = outlier_processing(data,'NumberOfDependents')
data = outlier_processing(data,'RevolvingUtilizationOfUnsecuredLines')
data = outlier_processing(data,'age')

Features = ['NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate']
Feature_labels = ['30-59Days','60-89Days','90+Days']
print('NumberOfTime30-59DaysPastDueNotWorse,\
      NumberOfTime60-89DaysPastDueNotWorse,\
      NumberOfTimes90DaysLate 原始分布：')
data[Features].boxplot()
plt.savefig('ch17_cs03.png',dpi=300,bbox_inches='tight')
plt.xticks([1,2,3],Feature_labels)
plt.show()
print('删除离群点后：')
data =data[(data['NumberOfTime30-59DaysPastDueNotWorse']<90) 
           & (data['NumberOfTime60-89DaysPastDueNotWorse']<90) 
           & (data['NumberOfTimes90DaysLate']<90)]
data[Features].boxplot()
plt.savefig('ch17_cs04.png',dpi=300,bbox_inches='tight')
plt.xticks([1,2,3],Feature_labels)
plt.show()
print('处理离群点后数据概况：')
data.info()

from sklearn.model_selection import train_test_split
data['SeriousDlqin2yrs'] = 1 - data['SeriousDlqin2yrs']
Y = data['SeriousDlqin2yrs']
X = data.iloc[:,1:]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
train = pd.concat([X_train,Y_train],axis=1)
test = pd.concat([X_test,Y_test],axis=1)
clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
print('训练集样本分布：')
print(train.shape)
print('测试集数据分布：')
print(test.shape)

def mono_bin(res,feat,n=10):
    good=res.sum()
    bad=res.count()-good
    d1 = pd.DataFrame({'feat':feat,'res':res,'Bucket':pd.qcut(feat,n)})
    d2 = d1.groupby('Bucket',as_index=True)
    d3 = pd.DataFrame(d2['feat'].min(),columns=['min'])
    d3['max'] = d2['feat'].max()
    d3['min'] =d2['feat'].min()
    d3['sum'] = d2['res'].sum()
    d3['total'] = d2['res'].count()
    d3['rate'] = d2.mean().res
    d3['woe'] = np.log((d3['rate']/good)/(1-d3['rate'])/bad)
    d3['goodattribute']= d3['sum']/good
    d3['badattribute'] = (d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_values(by='min'))
    cut=[]
    cut.append(float('-inf'))
    for i in range(1,n):
        qua=feat.quantile(i/n)
        cut.append(round(qua,4))
        cut.append(float('inf'))
        woe = list(d4['woe'].round(3))
        return d4,cut,woe,iv

def self_bin(res,feat,cat):
    good=res.sum()
    bad=res.count()-good
    d1 = pd.DataFrame({'feat':feat,'res':res,'Bucket':pd.cut(feat,cat)})
    d2 = d1.groupby('Bucket',as_index=True)
    d3 = pd.DataFrame(d2['feat'].min(),columns=['min'])
    d3['max'] = d2['feat'].max()
    d3['min'] =d2['feat'].min()
    d3['sum'] = d2['res'].sum()
    d3['total'] = d2['res'].count()
    d3['rate'] = d2.mean().res
    d3['woe'] = np.log((d3['rate']/good)/(1-d3['rate'])/bad)
    d3['goodattribute']= d3['sum']/good
    d3['badattribute'] = (d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_values(by='min'))
    woe = list(d4['woe'].round(3))
    return d4,iv,woe

pinf = float('inf')
ninf = float('-inf')
dfx1,ivx1,cutx1,woex1 = mono_bin(train['SeriousDlqin2yrs'],
                                 train['RevolvingUtilizationOfUnsecuredLines'],n=10)
print("="*60)
print('RevolvingUtilizationOfUnsecuredLines分箱结果,WOE信息：')
print(dfx1)
dfx2,ivx2,cutx2,woex2 = mono_bin(train['SeriousDlqin2yrs'],train['age'],n=10)
dfx4,ivx4,cutx4,woex4 = mono_bin(train['SeriousDlqin2yrs'],train['DebtRatio'],n=10)
dfx5,ivx5,cutx5,woex5 = mono_bin(train['SeriousDlqin2yrs'],train['MonthlyIncome'],n=10)

cutx3 = [ninf,0,1,3,5,pinf]
cut6 = [ninf,1,2,3,5,pinf]
cutx7 = [ninf,0,1,3,5,pinf]
cutx8 = [ninf,0,1,2,3,pinf]
cutx9 = [ninf,0,1,3,pinf]
cutx10 = [ninf,0,1,2,3,5,pinf]
dfx3,ivx3,woex3 = self_bin(train['SeriousDlqin2yrs'],
                           train['NumberOfTime30-59DaysPastDueNotWorse'],cutx3)
print("="*60)
print('NumberOfOpenCreditLinesAndLoans分箱结果,WOE信息：')
print(dfx3)
dfx6,ivx6,woex6 = self_bin(train['SeriousDlqin2yrs'],
                           train['NumberOfOpenCreditLinesAndLoans'],cut6)
dfx7,ivx7,woex7 = self_bin(train['SeriousDlqin2yrs'],
                           train['NumberOfTimes90DaysLate'],cutx7)
dfx8,ivx8,woex8 = self_bin(train['SeriousDlqin2yrs'],
                           train['NumberRealEstateLoansOrLines'],cutx8)
dfx9,ivx9,woex9 = self_bin(train['SeriousDlqin2yrs'], 
                           train['NumberOfTime60-89DaysPastDueNotWorse'],cutx9)
dfx10,ivx10,woex10 = self_bin(train['SeriousDlqin2yrs'], 
                          train['NumberOfDependents'],cutx10)

ivlist = [ivx1,ivx2,ivx3,ivx4,ivx5,ivx6,ivx7,ivx8,ivx9,ivx10]
index = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'] #补齐10个标签
fig1 = plt.figure(1) # 修正p1t → plt
ax1 = fig1.add_subplot(1,1,1)
x = np.arange(len(index))+1
ax1.bar(x,ivlist,width=0.4)
ax1.set_xticks(x)
ax1.set_xticklabels(index,rotation=0,fontsize=12)
ax1.set_ylabel('IV(Information Value)',fontsize=14)
for a,b in zip(x,ivlist):
    plt.text(a,b+0.01,'%.4f'%b,ha='center',va='bottom',fontsize=10) #英文引号
plt.savefig('ch19_cs05.png',dpi=300,bbox_inches='tight')
plt.show()


#模型训练
def get_woe(df,cut,woe,feat):
    res=[]
    for row in feat.iteritems():
        value = row[1]
        j = len(cut)-2
        m = len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j-=1
                m-=1
        res.append(woe[m])
    return res
woe_train = pd.DataFrame()
woe_train['SeriousDlqin2yrs'] = train['SeriousDlqin2yrs']
woe_train['RevolvingUtilizationOfUnsecuredLines'] = get_woe(
    train['RevolvingUtilizationOfUnsecuredLines'],cutx1,woex1)
woe_train['age'] = get_woe(train['age'],cutx2,woex2)
woe_train['NumberOfTime30-59DaysPastDueNotWorse'] = get_woe(
    train['NumberOfTime30-59DaysPastDueNotWorse'],cutx3,woex3)
woe_train['DebtRatio'] = get_woe(train['DebtRatio'],cutx4,woex4)
woe_train['MonthlyIncome'] = get_woe(train['MonthlyIncome'],cutx5,woex5)
woe_train['NumberofOpenCreditLinesAndLoans'] = get_woe(
    train['NumberofOpenCreditLinesAndLoans'],cut6,woex6)
woe_train['NumberOfTimes90DaysLate'] = get_woe(
    train['NumberOfTimes90DaysLate'],cutx7,woex7)
woe_train['NumberRealEstateLoansOrLines'] = get_woe(
    train['NumberRealEstateLoansOrLines'],cutx8,woex8)
woe_train['NumberOfTime60-89DaysPastDueNotWorse'] = get_woe(
    train['NumberOfTime60-89DaysPastDueNotWorse'],cutx9,woex9)
woe_train['NumberOfDependents'] = get_woe(
    train['NumberOfDependents'],cutx10,woex10)

woe_test = pd.DataFrame()
woe_test['SeriousDlqin2yrs'] = test['SeriousDlqin2yrs']
woe_test['RevolvingUtilizationOfUnsecuredLines'] = get_woe(
    test['RevolvingUtilizationOfUnsecuredLines'],cutx1,woex1)
woe_test['age'] = get_woe(test['age'],cutx2,woex2)
woe_test['NumberOfTime30-59DaysPastDueNotWorse'] = get_woe(
    test['NumberOfTime30-59DaysPastDueNotWorse'],cutx3,woex3)
woe_test['DebtRatio'] = get_woe(test['DebtRatio'],cutx4,woex4)
woe_test['MonthlyIncome']=get_woe(test['MonthlyIncome'],cutx5,woex5)
woe_test['NumberOfOpenCreditLinesAndLoans']=get_woe(
    test['NumberOfOpenCreditLinesAndLoans'],cutx6,woex6)
woe_test['NumberOfTimes90DaysLate']=get_woe(
    test['NumberOfTimes90DaysLate'],cutx7,cwoex7)
woe_test['NumberRealEsteateLoansOrLines']=get_woe(
    test['NumberRealEsteateLoansOrLines'],cutx8,cwoex8)
woe_test['NumberOfTimes60-89DaysPastDueNotWorse']=get_woe(
    test['NumberOfTimes60-89DaysPastDueNotWorse'],cutx9,cwoex9)
woe_test['NumberOfDependents']=get_woe(
    test['NumberOfDependents'],cutx10,woex10)

import statsmodels.api as sm
from sklearn.metrics import roc_curve,auc

Y=woe_train['SeriousDlqin2yrs']
X=woe_train.drop(['SerousDlqn2yrs','DebtRatio',
                  'MonthlyIncome','NumberOfOpenCreditlinsAndLoans','NumberRealEstateLoansOrLines','NumberOfDepdents'],
                  axis=1)
X1 = sm.add_constant(X)
logit=sm.Logit(Y,X1)
Logit_model=logit.fit()
print('输出拟合的各项系数：')
print(Logit_model.params)

Y_test =woe_test['SerioisDlqin2yrs']
X_test=woe_test.drop(['SeriousDlqin2yrs','DebtRatio','MonthlyIncome',
                      'NumberOfOpenCreditLinesAndLoans',
                      'NumberRealEstateLoansOrLines',
                      'NumberOfDependents'],axis=1)
X3=sm.add_constant(X_test)
resu=Logit_model.predict(X3)
fpr,tpr,threshold=roc_curve(Y_test,resu)
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%rocauc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR(真正率)')
plt.xlabel('FPR(假正率)')
plt.savefig('ch17_cs06.png',dpi=300,bbox_inches='tight')
plt.show()

