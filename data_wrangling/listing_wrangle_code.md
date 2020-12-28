```python
import pandas as pd
import glob
import os
```


```python
path = r'/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings' 

all_files = glob.glob(path + "/*.csv")
```


```python
all_files
```




    ['/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_stclara.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_sanmateo.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_rhode.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_oslo.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_stcruz.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_dc.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_seattle.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_nash.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_ptl.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_jer.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_clb.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_pacf.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_chicago.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_salem.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_aus.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_den.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_cam.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_clk.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_bro.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_norl.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_la.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_sd.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_asH.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_bos.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_sf.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_nyc.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_haw.csv',
     '/Users/luyongfei/Desktop/UChi_2020_fall/large_scale/project/listings/listings_msa.csv']




```python
ls = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    ls.append(df)

frame = pd.concat(ls, axis=0, ignore_index=True)
```

    /Users/luyongfei/opt/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3071: DtypeWarning: Columns (43,61,62) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    /Users/luyongfei/opt/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3071: DtypeWarning: Columns (61,62) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    /Users/luyongfei/opt/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3071: DtypeWarning: Columns (61,62,94) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    /Users/luyongfei/opt/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3071: DtypeWarning: Columns (43,94) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    /Users/luyongfei/opt/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3071: DtypeWarning: Columns (61,62,94,95) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    /Users/luyongfei/opt/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3071: DtypeWarning: Columns (61,62,95) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,



```python
df1 = frame.copy()
```


```python
df1.isna().sum()
```




    host_is_superhost           119
    city                        170
    price                         0
    host_since                  119
    room_type                     0
    latitude                      0
    longitude                     0
    reviews_per_month         50210
    number_of_reviews             0
    cancellation_policy           9
    security_deposit          67994
    cleaning_fee              38655
    beds                        258
    bedrooms                    106
    bathrooms                   167
    accommodates                  0
    host_response_time        52135
    host_identity_verified      119
    availability_30               0
    instant_bookable              0
    review_scores_rating      53245
    host_response_rate        52135
    amenities                     0
    dtype: int64




```python
df1 = df1.dropna(subset=['city', 'host_is_superhost','host_since', 'cancellation_policy', 'host_identity_verified'])
```


```python
df1['price'] = df1['price'].str.replace('[\$,]','',regex=True).astype(float)

```


```python

rv_mean = df1['reviews_per_month'].mean()
df1['reviews_per_month'].fillna(rv_mean, inplace = True)

```


```python
df1['occupancy_rate'] = df1['reviews_per_month'] / 0.5 * (3 / 30) * 100
```


```python
p1 = df1[df1['occupancy_rate'] <= 100].copy()
p2 = df1[df1['occupancy_rate'] > 100].copy()
p2['occupancy_rate'] = 100
df1 = pd.concat([p1, p2])
```


```python

df1['revenue'] = df1['reviews_per_month'] / 0.5 * 3 * df1['price']
```


```python
# If security_deposit is missing, then we assume it to be 0 since charges should have been clarified on the platform
df1['security_deposit'].fillna('$0.00', inplace = True)
df1['security_deposit'] = df1['security_deposit'].str.replace('[\$,]','',regex=True).astype(float)
```


```python
# If cleaning fee is missing, then we assume it to be 0 since charges should have been clarified on the platform
df1['cleaning_fee'].fillna('$0.00', inplace = True)
df1['cleaning_fee'] = df1['cleaning_fee'].replace('[\$,]','',regex=True).astype(float)
```


```python
df1['host_response_time'].fillna('Missing', inplace = True)
```


```python
df1['host_response_time'].value_counts()
```




    within an hour        153060
    Missing                51984
    within a few hours     33933
    within a day           18076
    a few days or more      3860
    Name: host_response_time, dtype: int64




```python
df1['host_response_time'].fillna('Missing', inplace = True)
df1['host_response_time'] = df1['host_response_time'].where(df1['host_response_time'] != 'within an hour', 1.0)

df1['host_response_time'] = df1['host_response_time'].where(df1['host_response_time'] != 'within a few hours', 2.0)
df1['host_response_time'] = df1['host_response_time'].where(df1['host_response_time'] != 'within a day', 3.0)
df1['host_response_time'] = df1['host_response_time'].where(df1['host_response_time'] != 'a few days or more', 4.0)


df1['host_response_time'] = df1['host_response_time'].where(df1['host_response_time'] != 'Missing', 5.0)

df1.host_response_time.value_counts()
```




    1.0    153060
    5.0     51984
    2.0     33933
    3.0     18076
    4.0      3860
    Name: host_response_time, dtype: int64




```python
rating_mean = df1['review_scores_rating'].mean()
df1['review_scores_rating'].fillna(rating_mean, inplace = True)
```


```python
# deal with host_response_rate
df1['host_response_rate'].fillna('0%', inplace = True)
df1['host_response_rate'] = df1['host_response_rate'].str.rstrip('%').astype('float')
```


```python
df1['beds'].fillna(1.0, inplace = True)
df1['bedrooms'].fillna(1.0, inplace = True)
df1['bathrooms'].fillna(1.0, inplace = True)
```


```python
import datetime

df1['host since'] = pd.to_datetime(df1['host_since'], format = '%Y-%m-%d')
df1['now'] = datetime.datetime(2019, 9, 1)
df1['host_time'] = (df1['now'] - df1['host since']).dt.days
```


```python
df1['is_TV'] = df1.amenities.apply(lambda s: int('TV' in str(s)[1:].split(',')))
df1['is_Wifi'] = df1.amenities.apply(lambda s: int('Wifi' in str(s)[1:].split(',')))
df1['amenities_number'] = df1.amenities.apply(lambda s: len(str(s)[1:].split(',')))
```


```python
df1.to_csv('listings_overall.csv')
```


```python
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 260913 entries, 0 to 261087
    Data columns (total 31 columns):
     #   Column                  Non-Null Count   Dtype         
    ---  ------                  --------------   -----         
     0   host_is_superhost       260913 non-null  object        
     1   city                    260913 non-null  object        
     2   price                   260913 non-null  float64       
     3   host_since              260913 non-null  object        
     4   room_type               260913 non-null  object        
     5   latitude                260913 non-null  float64       
     6   longitude               260913 non-null  float64       
     7   reviews_per_month       260913 non-null  float64       
     8   number_of_reviews       260913 non-null  int64         
     9   cancellation_policy     260913 non-null  object        
     10  security_deposit        260913 non-null  float64       
     11  cleaning_fee            260913 non-null  float64       
     12  beds                    260913 non-null  float64       
     13  bedrooms                260913 non-null  float64       
     14  bathrooms               260913 non-null  float64       
     15  accommodates            260913 non-null  int64         
     16  host_response_time      260913 non-null  object        
     17  host_identity_verified  260913 non-null  object        
     18  availability_30         260913 non-null  int64         
     19  instant_bookable        260913 non-null  object        
     20  review_scores_rating    260913 non-null  float64       
     21  host_response_rate      260913 non-null  float64       
     22  amenities               260913 non-null  object        
     23  occupancy_rate          260913 non-null  float64       
     24  revenue                 260913 non-null  float64       
     25  host since              260913 non-null  datetime64[ns]
     26  now                     260913 non-null  datetime64[ns]
     27  host_time               260913 non-null  int64         
     28  is_TV                   260913 non-null  int64         
     29  is_Wifi                 260913 non-null  int64         
     30  amenities_number        260913 non-null  int64         
    dtypes: datetime64[ns](2), float64(13), int64(7), object(9)
    memory usage: 63.7+ MB



```python
df1[['host_is_superhost','city',
             'price','room_type','latitude','longitude',
             'reviews_per_month','number_of_reviews', 
             'cancellation_policy', 'security_deposit', 
             'cleaning_fee', 'beds', 'bedrooms', 'bathrooms',
             'accommodates', 'host_response_time', 
             'host_identity_verified', 'availability_30', 
             'instant_bookable','review_scores_rating',
             'host_response_rate', 'occupancy_rate', 'revenue',
             'host_time', 'is_TV', 'is_Wifi', 'amenities_number']].to_csv('ls.csv')
```


```python

```
