counts = data['stage'].value_counts()

counts.plot.bar(title="stageの頻度")

plt.xlabel("stage")
plt.ylabel("count")

import seaborn as sns

sns.boxplot(data=data[data['stage']=='Ｊ１'], x='mathch_num', y='y')
sns.boxplot(data=data[data['stage']=='Ｊ１'], x='mathch_num', y='y')
sns.boxplot
