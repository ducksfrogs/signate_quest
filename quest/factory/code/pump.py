train_X, train_y = train_df[['mean','zc']], train_df['label']
test_X, test_y = test_df[['mean','zc']], test_df['label']

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(train_X)

train_X = sc.transform(train_X)
test_X = sc.transform(test_X)

model = OneClassSVM(nu=0.01)
model.fit(train_X)
pred = model.predict(test_X)
pred = np.where(pred == -1, 1, 0)

def create_power_spectral(data):
    N = data.shape[1]
    dt = 10/N
    F = np.abs(np.fft.fft(data)/(N/2))
    fq = np.linspace(0, 1/dt, N)
    return F[:,:int(N/2)+1], fq[:int(N/2)+1]

F, freq = create_power_spectral(train)
plt.plot(freq, F[0])
plt.show()

import librosa

melspec = librosa.feature.melspectrogram(train[0])

librosa.display.specshow(melspec, x_axis='time', y_axis='mel')
plt.show()

melspec_db = librosa.amplitude_to_db(melspec)
