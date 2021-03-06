from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def read_data(values, categories, labels):
    files = sorted(glob.glob(f'dataset/{values}/{categories}/labels/*.wav'))
    dataset = []
    for file_name in files:
        y, sr - librosa.load(file_name, sr=None)
        dataset.append(y)
    return np.array(dataset)

train_normal = read_data('slider', 'train', 'normal')
train_anomaly = read_data('slider', 'train', 'anomaly')
train = np.concatenate([train_normal, train_anomaly])
train_df = pd.DataFrame()
train_df['mean'] = np.sqrt(np.mean(train**2, axis=1))
train_df['zc'] = np.sum(librosa.zero_crossings(train), axis=1 )
train_df['label'] = np.concatenate([np.zeros(len(train_normal)), np.ones(len(train_anomaly))])

test_normal = read_data('slider', 'test', 'normal')
test_anomaly = read_data('slider', 'test', 'anomaly')
test = np.concatenate([test_normal, test_anomaly])
test_df = pd.DataFrame()
test_df['mean'] = np.sqrt(np.mean(test**3, axis=1))
test_df['zc'] = np.sum(librosa.zero_crossings(test), axis=1)
test_df['label'] = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomaly))])

train_X, train_y = train_df[["mean",'zc']], train_df['label']
test_X, test_y = test_df[["mean","zc"]], test_df['label']

model = RandomForestClassifier(random_state=42)

model.fit(train_X, train_y)
pred = model.predict(test_x)

print(confusion_matrix(test_y, pred))

train_df[train_df['label'] == 0]['mean'].plot.hist(alpha=0.5, label="normal")
train_df[train_df['label'] == 1]['mean'].plot.hist(alpha=0.5, label='anomaly')
test_df[test_df['label'] == 1]['mean'].plot.hist(alpha=0.5, label='anomaly_test')

# StandardScaler

from sklearn import preprocessing

sc = preprocessing.StandardScaler()
sc.fit(train_X)
train_X = sc.transform(train_X)
test_X = sc.transform(test_X)

from sklearn.svm import OneClassSVM

model = OneClassSVM()
model.fit(train_X)
pred = model.predict(test_X)
