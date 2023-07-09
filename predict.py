import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from scipy import stats
import streamlit as st


# Sayfa Ayarları
st.set_page_config(
    page_title="House Classifier")

# Başlık Ekleme
st.title("House Price")


# Resim Ekleme
st.image("http://thecooperreview.com/wp-content/uploads/2015/04/NYCSF4.png")


st.image("https://resources.pollfish.com/wp-content/uploads/2020/11/MARKET_RESEARCH_FOR_REAL_ESTATE_IN_CONTENT_1.png")


# Pandasla veri setini okuyalım
df = pd.read_csv("C:/Users/Alimturk/Desktop/Bengaluru_House_Data.csv")

# Küçük bir düzenleme :)
df = df.drop(['area_type','society','balcony','availability'],axis='columns')
df = df.dropna()
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
df = df.drop(['size'],axis='columns')
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None 
df.total_sqft = df.total_sqft.apply(convert_sqft_to_num)
df = df[df.total_sqft.notnull()]
df['price_per_sqft'] = df['price']*100000/df['total_sqft']
df.location = df.location.apply(lambda x: x.strip())
location_stats = df['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
df = df[~(df.total_sqft/df.bhk<300)]
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df = remove_pps_outliers(df)
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df = remove_bhk_outliers(df)
df = df[df.bath<df.bhk+2]
df = df.drop(['price_per_sqft'],axis='columns')
dummies = pd.get_dummies(df.location)
df = pd.concat([df,dummies.drop('other',axis='columns')],axis='columns')
df = df.drop('location',axis='columns')
X = df.drop(['price'],axis='columns')
from sklearn.model_selection import train_test_split
X = df.drop(['price'],axis='columns')
y = df.price
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression(copy_X= True,fit_intercept= False, positive= False)
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]



#---------------------------------------------------------------------------------------------------------------------

# Sidebarda Kullanıcıdan Girdileri Alma
location = st.sidebar.text_input("location", help="Please location!")
bath = st.sidebar.number_input("Bath", min_value=1, format="%d")
sqft = st.sidebar.number_input("Square Feet of House", min_value=1)
bhk = st.sidebar.slider("bhk of House ", min_value=0, max_value=250)

#---------------------------------------------------------------------------------------------------------------------


input_df = pd.DataFrame({
    'location': [location],
    'bath': [bath],
    'sqft': [sqft],
    'bhk': [bhk]
})

predict=predict_price(input_df[0],input_df[1],input_df[2],input_df[3])


#---------------------------------------------------------------------------------------------------------------------
st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    # Sorgulama zamanına ilişkin bilgileri elde etme
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")
st.write(predict)
    

   