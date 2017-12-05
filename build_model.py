import pandas as pd
import sys
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.svm import SVC


def get_pca(X,n):
    """
    Transform data into . Should return an array with shape (, n).
    """
    flatten_model = make_pipeline(
        PCA(n_components=n)
    )
    X = flatten_model.fit_transform(X)
    assert X.shape == (X.shape[0], n)
    return X


input_csv = sys.argv[1]
print('Reading dataframe.')
data = pd.read_csv(input_csv)

pixel_value = data.iloc[:,8:]
print('Applying PCA to pixel data.')
X_pca= get_pca(pixel_value,250)
X = pd.DataFrame(X_pca)
X['avg_l']=data['avg_l']
X['edge_count']=data['edge_count']
y = pd.DataFrame(data={'weather':data['weather'],'weather_re':data['weather_re']})

score_list=[]

for i in range(20):
    print('Start to test on split '+str(i+1)+'.')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y)
    y_test_true = y_test[pd.notnull(y_test['weather'])].weather
    X_test_true= X_test.join(y_test_true,how = 'inner').iloc[:,:-1]
    y_train = y_train['weather_re']
    svc_model = SVC(C=70,gamma=0.000000001)
    svc_model.fit(X_train,y_train)
    score_list.append(svc_model.score(X_test_true,y_test_true))
print('Score of this model is: '+str(sum(score_list)/len(score_list)))    
