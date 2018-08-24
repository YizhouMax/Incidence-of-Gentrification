
#The Incidence of Downward Mobility: Evidence from Chicago 

##Code Author: Yizhou Zhang, Ph.D. candidate in Agricultural & Consumer Economics, Univ of Illinois at Urbana-Champaign


```python
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from stargazer.stargazer import Stargazer
from IPython.core.display import display, HTML
```

Use the American Community Survey (ACS) data between 2010-2016 to identify which census tracts gentrified between 2010-2015. The data is already downloaded. 


```python
ACS1016 =  pd.read_csv("C:/Users/Max/Google Drive/My Research/Chicago Crime-Migration/data/ACS1016.csv")

# Create a new column named "MergeID" such that the table can be joined. THe MergeID is a concatination of "Year" and "GEOID"
ACS1016['MergeID'] =  ACS1016["Year"].apply(int).apply(str) +  ACS1016["GEOID"].apply(int).apply(str)

# drop the first column that is redundant

ACS1016 = ACS1016.drop(ACS1016.columns[0], 1)
```


```python
ACS1016[:5]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>under24</th>
      <th>A2549</th>
      <th>A5061</th>
      <th>A62more</th>
      <th>TotPop</th>
      <th>MHHI</th>
      <th>EDU</th>
      <th>P_Own</th>
      <th>pct_Built20y</th>
      <th>MHV</th>
      <th>MGR</th>
      <th>Less10k</th>
      <th>I10k_20k</th>
      <th>more20k</th>
      <th>TotRsp</th>
      <th>Year</th>
      <th>MergeID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17001000100</td>
      <td>0.261459</td>
      <td>0.327954</td>
      <td>0.148268</td>
      <td>0.262320</td>
      <td>4647</td>
      <td>40114.0</td>
      <td>0.178179</td>
      <td>0.864256</td>
      <td>0.883983</td>
      <td>96900.0</td>
      <td>587.0</td>
      <td>0.060122</td>
      <td>0.101926</td>
      <td>0.837952</td>
      <td>2129</td>
      <td>2010</td>
      <td>201017001000100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17001000201</td>
      <td>0.334850</td>
      <td>0.299818</td>
      <td>0.181074</td>
      <td>0.184258</td>
      <td>2198</td>
      <td>41356.0</td>
      <td>0.109190</td>
      <td>0.800712</td>
      <td>0.870700</td>
      <td>79800.0</td>
      <td>657.0</td>
      <td>0.087782</td>
      <td>0.098458</td>
      <td>0.813760</td>
      <td>843</td>
      <td>2010</td>
      <td>201017001000201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17001000202</td>
      <td>0.503276</td>
      <td>0.261778</td>
      <td>0.138222</td>
      <td>0.096724</td>
      <td>3205</td>
      <td>44076.0</td>
      <td>0.130421</td>
      <td>0.775216</td>
      <td>0.974063</td>
      <td>85100.0</td>
      <td>624.0</td>
      <td>0.097983</td>
      <td>0.060519</td>
      <td>0.841499</td>
      <td>1041</td>
      <td>2010</td>
      <td>201017001000202</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17001000400</td>
      <td>0.328177</td>
      <td>0.261878</td>
      <td>0.149613</td>
      <td>0.260331</td>
      <td>4525</td>
      <td>24230.0</td>
      <td>0.057017</td>
      <td>0.547820</td>
      <td>0.856864</td>
      <td>52800.0</td>
      <td>555.0</td>
      <td>0.111906</td>
      <td>0.303839</td>
      <td>0.584255</td>
      <td>1537</td>
      <td>2010</td>
      <td>201017001000400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17001000500</td>
      <td>0.364100</td>
      <td>0.313703</td>
      <td>0.159117</td>
      <td>0.163080</td>
      <td>1766</td>
      <td>31583.0</td>
      <td>0.107022</td>
      <td>0.614966</td>
      <td>0.952381</td>
      <td>67200.0</td>
      <td>580.0</td>
      <td>0.134694</td>
      <td>0.164626</td>
      <td>0.700680</td>
      <td>735</td>
      <td>2010</td>
      <td>201017001000500</td>
    </tr>
  </tbody>
</table>
</div>



now extract all observations (census tracts) from the year 2010


```python

ACS2010 = ACS1016[ACS1016.Year == 2010]


ACS2010 = ACS2010.rename(columns={'MHHI':'MHHI_10', 'EDU':'EDU_10','pct_Built20y':'pct_Built20y_10','MHV':'MHV_10','MGR':'MGR_10'})

ACS2010.columns

ACS2010 = ACS2010[['GEOID', 'MHHI_10', 'EDU_10','pct_Built20y_10','MHV_10','MGR_10']]


```


```python
ACS2010[:5]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>MHHI_10</th>
      <th>EDU_10</th>
      <th>pct_Built20y_10</th>
      <th>MHV_10</th>
      <th>MGR_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17001000100</td>
      <td>40114.0</td>
      <td>0.178179</td>
      <td>0.883983</td>
      <td>96900.0</td>
      <td>587.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17001000201</td>
      <td>41356.0</td>
      <td>0.109190</td>
      <td>0.870700</td>
      <td>79800.0</td>
      <td>657.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17001000202</td>
      <td>44076.0</td>
      <td>0.130421</td>
      <td>0.974063</td>
      <td>85100.0</td>
      <td>624.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17001000400</td>
      <td>24230.0</td>
      <td>0.057017</td>
      <td>0.856864</td>
      <td>52800.0</td>
      <td>555.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17001000500</td>
      <td>31583.0</td>
      <td>0.107022</td>
      <td>0.952381</td>
      <td>67200.0</td>
      <td>580.0</td>
    </tr>
  </tbody>
</table>
</div>




```python


#now join the dataset with 2010 housing values of its origin tract

#ACS2010 = subset(ACS1016, select = c(GEOID, PrevMHV, Year))


ACS2010 = ACS1016[ACS1016.Year == 2010]


ACS2010 = ACS2010.rename(columns={'MHHI':'MHHI_10', 'EDU':'EDU_10','pct_Built20y':'pct_Built20y_10','MHV':'MHV_10','MGR':'MGR_10'})


ACS2010 = ACS2010[['GEOID', 'MHHI_10', 'EDU_10','pct_Built20y_10','MHV_10','MGR_10']]




#also record the 2015 housing price. For measuring the intensity of gentrification 2015/2010
ACS2015 = ACS1016[ACS1016.Year == 2015]


ACS2015 = ACS2015.rename(columns={'MHHI':'MHHI_15', 'EDU':'EDU_15','pct_Built20y':'pct_Built20y_15',
                                  'MHV':'MHV_15','MGR':'MGR_15'})


ACS2015 = ACS2015[['GEOID', 'MHHI_15', 'EDU_15','pct_Built20y_15','MHV_15','MGR_15']]


```


```python

# merge MHV2010 and MHV2015 to get any neighborhood's changes between 2010-2015


Gent1015 =  pd.merge(ACS2010, ACS2015, how='inner', on = "GEOID")

Gent1015 = Gent1015.dropna(axis=0, how='any')


#now we have a dataset of tracts that can compare between 2010 and 2015. 


# Now we


Gent1015['EDU_1015'] = Gent1015.EDU_15 - Gent1015.EDU_10

Gent1015['MHHI_1015'] = Gent1015.MHHI_15 / Gent1015.MHHI_10

Gent1015['MHV_1015'] = Gent1015.MHV_15 / Gent1015.MHV_10

Gent1015['MGR_1015'] = Gent1015.MGR_15 / Gent1015.MGR_10

Gent1015['Built_1015'] = Gent1015.pct_Built20y_15 / Gent1015.pct_Built20y_10
```

Build a series of flags as criteria of gentrification 


```python

# build the indicator flags 
Gent1015['EDU_Flag'] = Gent1015.EDU_1015.map(lambda x : 1 if x > Gent1015.EDU_1015.quantile(0.5) else 0)


Gent1015['MHHI_Flag'] = Gent1015.MHHI_1015.map(lambda x : 1 if x > Gent1015.MHHI_1015.quantile(0.5) else 0)


Gent1015['MHV_Flag'] = Gent1015.MHV_1015.map(lambda x : 1 if x > Gent1015.MHV_1015.quantile(0.5) else 0)

#Gent1015$MGR_Flag = ifelse(Gent1015$MGR_1015 > median(Gent1015$MGR_1015), 1,0)



# find the gentrifiable tracts , and gentrified tracts 


Gent1015['gent_able'] = Gent1015.MHV_10.map(lambda x : 1 if x < Gent1015.MHV_10.quantile(0.5) else 0)



Gent1015['gented'] = Gent1015.EDU_Flag * Gent1015.MHV_Flag  * Gent1015.gent_able


Gent1015['gent_class'] =  Gent1015.gented + Gent1015.gent_able


# Now use replace function to encode


Gent1015['gent_class'] = Gent1015.gent_class.replace([0,1,2],["nongentrifiable","nongentrified","gentrified"])


# then generate the dataset to join INFO usa
Gentable1015 = Gent1015[Gent1015.gent_able ==1]
  

#Gentable1015['Gented'] = Gentable1015.EDU_Flag * Gentable1015.MHV_Flag 


#Gentable1015['gented'] = Gentable1015.gented.map(lambda x: 'gented' if x == 1 else 'nongented')


```

The next step is summary statistics of table "Gent1015" by group  (group ID  = "gent_class")

First compare neighborhoods' 2010 characterisitics by their gentrification status. The summary statistics is median. 


```python
Gent1015 = Gent1015.dropna()

Gent1015.groupby("gent_class")[['MHV_10','MGR_10','EDU_10',"MHHI_10"]].median()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MHV_10</th>
      <th>MGR_10</th>
      <th>EDU_10</th>
      <th>MHHI_10</th>
    </tr>
    <tr>
      <th>gent_class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gentrified</th>
      <td>100000.0</td>
      <td>644.0</td>
      <td>0.101959</td>
      <td>45263.0</td>
    </tr>
    <tr>
      <th>nongentrifiable</th>
      <td>291400.0</td>
      <td>958.0</td>
      <td>0.239663</td>
      <td>65838.0</td>
    </tr>
    <tr>
      <th>nongentrified</th>
      <td>121500.0</td>
      <td>706.5</td>
      <td>0.101850</td>
      <td>44046.0</td>
    </tr>
  </tbody>
</table>
</div>



The expected result is that the classes "gentrified" and "nongentrified" are comparable in their 2010 characterisitcs while both are distinct with the class "nongentrifiable". The above median comparision meets this expectation. 

Then compare neighborhoods' 2010-2015 percentage changes in selected characterisitics by their gentrification status. The summary statistics is median. 


```python
a = Gent1015.groupby("gent_class")[['MHV_1015','MGR_1015',"MHHI_1015"]].median() -1

print (a.to_string(float_format= lambda x: "{:.2f}%".format(x) ))
```

                     MHV_1015  MGR_1015  MHHI_1015
    gent_class                                    
    gentrified          0.03%     0.07%      0.07%
    nongentrifiable    -0.17%     0.09%      0.03%
    nongentrified      -0.09%     0.08%      0.02%
    

The expectation is that the class "gentrified" experienced the greatest positive changes relative to the other two classes. The results above confirms the expectation. 

The next step is to join mover_stayer dataset with the ACS1016 dataset. In other words, join the longitudinal dataset with information of neighborhood characteristics. 


```python
mover = pd.read_csv("C:/Users/Max/Google Drive/My Research/Chicago Crime-Migration/data/mover.csv")

```


```python

mover['Prev_MergeID'] = (mover.YearOnly).apply(int).apply(str) + mover.PrevGEOID.apply(int).apply(str)

```


```python

Prev_Charac = ACS1016[['MHV','MHHI','Year','GEOID']]


Prev_Charac =  Prev_Charac.rename(columns= {'MHV':"PrevMHV", 'MHHI':"PrevMHHI"})

Prev_Charac['MergeID'] = (Prev_Charac.Year).apply(int).apply(str) + Prev_Charac.GEOID.apply(int).apply(str)


```


```python
# now merge the mover file with ACS file 

mover_prev =  pd.merge(mover, Prev_Charac, how='inner',left_on= "Prev_MergeID", right_on ="MergeID"  )

mover_prev.shape

mover_gent = pd.merge(mover_prev, Gent1015, how='inner',left_on= "PrevGEOID", right_on ="GEOID"  )
mover_gent.shape
```




    (204966, 51)



We only study households with vulnerable housing status (Eligible ==1 )


```python
mover_gent = mover_gent[mover_gent.Eligible ==1]
```


```python


#now create the logged difference of MHV between this year and last year



mover_gent["Move_MHVdiff"] =  mover_gent.MHV / mover_gent.PrevMHV

mover_gent["Move_MHHIdiff"] =  mover_gent.MHHI / mover_gent.PrevMHHI



#mover_gent['Gented'] = relevel(mover_gent$gent_class, ref = "nongentrifiable")
#mover_gent = subset(mover_diff, gent_able ==1)
# & FIND_DIV_1000  < Inc
#& FIND_DIV_1000  < quantile(mover_diff$FIND_DIV_1000, 0.4, na.rm = TRUE)


mover_gent['MHVdown'] = mover_gent.Move_MHVdiff.map(lambda x:  1 if x <1 else 0)
mover_gent['MHHIdown'] = mover_gent.Move_MHHIdiff.map(lambda x:  1 if x <1 else 0)

mover_gent = mover_gent.dropna()
```

household summary statistics by the gentrification status of its origin neighborhood


```python
a = mover_gent.groupby("gent_class")[['FIND_DIV_1000','PrevLOR','Move_MHHIdiff',"Move_MHVdiff","HSize","MHHIdown","MHVdown"]].mean()

print (a.to_string(float_format= lambda x: "{:.2f}".format(x) ))

```

                     FIND_DIV_1000  PrevLOR  Move_MHHIdiff  Move_MHVdiff  HSize  MHHIdown  MHVdown
    gent_class                                                                                    
    gentrified            76143.47     7.52           1.49          1.50   2.15      0.22     0.22
    nongentrifiable      113705.51     8.39           1.16          1.10   2.25      0.43     0.48
    nongentrified         69519.41     8.69           1.43          1.57   2.20      0.23     0.18
    


```python
cat_mover =  pd.get_dummies(mover_gent.loc[:,['OWNER_RENTER_STATUS','HEAD_HH_AGE_CODE']])

cat_mover['Gent_Status'] = mover_gent.gent_class

print(cat_mover.groupby("Gent_Status").mean())
```

                     OWNER_RENTER_STATUS_Owner  OWNER_RENTER_STATUS_Renter  \
    Gent_Status                                                              
    gentrified                        0.718077                    0.281923   
    nongentrifiable                   0.829822                    0.170178   
    nongentrified                     0.745013                    0.254987   
    
                     HEAD_HH_AGE_CODE_25-29  HEAD_HH_AGE_CODE_30-34  \
    Gent_Status                                                       
    gentrified                     0.111549                0.150646   
    nongentrifiable                0.085130                0.132977   
    nongentrified                  0.106787                0.138563   
    
                     HEAD_HH_AGE_CODE_35-39  HEAD_HH_AGE_CODE_40-44  \
    Gent_Status                                                       
    gentrified                     0.142755                0.142755   
    nongentrifiable                0.145024                0.133070   
    nongentrified                  0.140888                0.131150   
    
                     HEAD_HH_AGE_CODE_45-49  HEAD_HH_AGE_CODE_50-54  \
    Gent_Status                                                       
    gentrified                     0.100789                0.087877   
    nongentrifiable                0.116283                0.097177   
    nongentrified                  0.115514                0.095431   
    
                     HEAD_HH_AGE_CODE_55-59  HEAD_HH_AGE_CODE_60-64  \
    Gent_Status                                                       
    gentrified                     0.074247                0.051650   
    nongentrifiable                0.077379                0.060296   
    nongentrified                  0.071405                0.055264   
    
                     HEAD_HH_AGE_CODE_65+ (inferred)  HEAD_HH_AGE_CODE_65-69  \
    Gent_Status                                                                
    gentrified                              0.003945                0.038737   
    nongentrifiable                         0.003867                0.040062   
    nongentrified                           0.004920                0.036595   
    
                     HEAD_HH_AGE_CODE_70-74  HEAD_HH_AGE_CODE_75+  \
    Gent_Status                                                     
    gentrified                     0.018293              0.040890   
    nongentrifiable                0.026194              0.051429   
    nongentrified                  0.022948              0.046300   
    
                     HEAD_HH_AGE_CODE_< 25  
    Gent_Status                             
    gentrified                    0.035868  
    nongentrifiable               0.031113  
    nongentrified                 0.034236  
    

The above household summary statistics indicate that households moving from gentrified neighborhoods are much more likely to be  renters. In other words, we have reason to believe that gentrification is impacting renters more than homeowners. Next step is to use multivariate regression to investigate this incidence difference. 

First prepare a table with households moving from "gentrified" and "nongentrified" neighborhoods


```python

mover_gentable = mover_gent.loc[mover_gent.gent_class != "nongentrifiable"]
mover_gentable.dropna(inplace=True)

mover_gentable['YearOnly'] = mover_gentable.YearOnly.apply(str)
```

    D:\Anaconda\Anaconda\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    D:\Anaconda\Anaconda\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    

Import necessary tools and packages to display regression results

The lm.cat_gent model is using binary indicator of gentrification, regardless of the gentrification intensity


```python

lm.cat_gent = smf.ols(formula='MHHIdown ~ gent_class  +\
C(OWNER_RENTER_STATUS, Treatment(reference="Owner")) + C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))*gent_class +\
YearOnly + HEAD_HH_AGE_CODE + MARITAL_STATUS + HSize   + FIND_DIV_1000  + PrevLOR', data=mover_gentable).fit()

```

Then replace the binary gentrification dummy with a continuous metric of gentrification intensity. The regression result is shown in lm.cnt_gent


```python
mover_gentable['gent_class'] = mover_gentable.gent_class.map(lambda x: 1 if x == "gentrified" else 0)

mover_gentable['Gentensity'] = (mover_gentable.gent_class) * ((mover_gentable.MGR_1015 + mover_gentable.MHV_1015)/2) 

Gent_Inten = mover_gentable.Gentensity

```

    D:\Anaconda\Anaconda\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    D:\Anaconda\Anaconda\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    

Replace all zeroes in the series with NaN. The reason is that zero should not enter the quantile cut function.


```python
Gent_Inten = Gent_Inten.replace(0, np.NaN)
```




    0   NaN
    1   NaN
    2   NaN
    3   NaN
    4   NaN
    Name: Gentensity, dtype: float64




```python

Gent_Inten = pd.qcut(Gent_Inten, 4, labels = ["low","med","med2","high"])

Gent_Inten = Gent_Inten.replace('med2', 'med')

Gent_Inten = Gent_Inten.replace(np.NaN, "nongented")

mover_gentable['Gentensity'] = Gent_Inten

```

    D:\Anaconda\Anaconda\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    




    0    nongented
    1    nongented
    2    nongented
    3    nongented
    4    nongented
    Name: Gentensity, dtype: object




```python
lm.cnt_gent = smf.ols(formula='MHHIdown ~ C(Gentensity, Treatment(reference="nongented")) +\
C(OWNER_RENTER_STATUS, Treatment(reference="Owner")) +\
C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))*C(Gentensity, Treatment(reference="nongented")) +\
             YearOnly + HEAD_HH_AGE_CODE + MARITAL_STATUS + HSize   + FIND_DIV_1000  + PrevLOR', data=mover_gentable).fit()

```


```python
stargazer = Stargazer([lm.cat_gent,lm.cnt_gent])

stargazer.covariate_order(['C(Gentensity, Treatment(reference="nongented"))[T.high]',
                           'C(Gentensity, Treatment(reference="nongented"))[T.med]',
                          'C(Gentensity, Treatment(reference="nongented"))[T.low]', 
                           'gent_class',
                            'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]:gent_class',
                           'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]',
                           'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]:C(Gentensity, Treatment(reference="nongented"))[T.high]',
                           'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]:C(Gentensity, Treatment(reference="nongented"))[T.med]',
                           'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]:C(Gentensity, Treatment(reference="nongented"))[T.low]'
                          
                          
                          ])
stargazer.rename_covariates({'C(Gentensity, Treatment(reference="nongented"))[T.high]': 'High Intensity',
                            'C(Gentensity, Treatment(reference="nongented"))[T.med]': 'Medium Intensity',
                             'C(Gentensity, Treatment(reference="nongented"))[T.low]': 'Low Intensity',
                            'gent_class':'Gentrified (Dummy)',
                             'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]:gent_class' : 'Renter * Gentrified',
                             'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]':'Renter (Dummy)',
                            'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]:C(Gentensity, Treatment(reference="nongented"))[T.high]':'Renter*High',
                            'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]:C(Gentensity, Treatment(reference="nongented"))[T.med]':'Renter*Medium',
                             'C(OWNER_RENTER_STATUS, Treatment(reference="Owner"))[T.Renter]:C(Gentensity, Treatment(reference="nongented"))[T.low]':'Renter*Low'
                            })
stargazer.dependent_variable_name('Dep Var: Downward Move (Dummy)')

b=stargazer.render_html()
```


```python

display(HTML(b))
```


<table style="text-align:center"><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="2"><em>Dep Var: Downward Move (Dummy)</em></td></tr><tr><td style="text-align:left"></td><tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">High Intensity</td><td></td><td>0.12<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(0.018)</td></tr><tr><td style="text-align:left">Medium Intensity</td><td></td><td>-0.006<sup></sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(0.013)</td></tr><tr><td style="text-align:left">Low Intensity</td><td></td><td>-0.007<sup></sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(0.017)</td></tr><tr><td style="text-align:left">Gentrified (Dummy)</td><td>0.024<sup>**</sup></td><td></td></tr><tr><td style="text-align:left"></td><td>(0.009)</td><td></td></tr><tr><td style="text-align:left">Renter * Gentrified</td><td>-0.062<sup>***</sup></td><td></td></tr><tr><td style="text-align:left"></td><td>(0.017)</td><td></td></tr><tr><td style="text-align:left">Renter (Dummy)</td><td>0.152<sup>***</sup></td><td>0.152<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td>(0.006)</td><td>(0.006)</td></tr><tr><td style="text-align:left">Renter*High</td><td></td><td>-0.067<sup>*</sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(0.038)</td></tr><tr><td style="text-align:left">Renter*Medium</td><td></td><td>-0.028<sup></sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(0.023)</td></tr><tr><td style="text-align:left">Renter*Low</td><td></td><td>-0.112<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(0.034)</td></tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Observations</td><td>32464.0</td><td>32464.0</td></tr><tr><td style="text-align: left">R<sup>2</sup></td><td>0.108</td><td>0.11</td></tr><tr><td style="text-align: left">Adjusted R<sup>2</sup></td><td>0.108</td><td>0.109</td></tr><tr><td style="text-align: left">Residual Std. Error</td><td>0.394(df = 32439.0)</td><td>0.394(df = 32435.0)</td></tr><tr><td style="text-align: left">F Statistic</td><td>163.977<sup>***</sup>(df = 24.0; 32439.0)</td><td>142.656<sup>***</sup>(df = 28.0; 32435.0)</td></tr><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Note:</td><td colspan="2" style="text-align: right"><em>p&lt;0.1</em>; <b>p&lt;0.05</b>; p&lt;0.01</td></tr></table>



```python

```
