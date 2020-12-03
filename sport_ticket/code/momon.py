counts = data['stage'].value_counts()

counts.plot.bar(title="stageの頻度")

plt.xlabel("stage")
plt.ylabel("count")
