Locate 'EDU embedding' in here. <BR>
File format: .pkl file <BR>
# How to use the edu vector pickle<BR>
1. open the pickle file<BR>
f = open('edu_vecs.pickle')<BR>
2. load with cPickle<BR>
import cPickle<BR>
dic_eduidx2embedding = cPickle.load(f)<BR>
3. use the loaded dictionary<BR>
dic_eduidx2embedding['edu0'] => return 200 dimensional real-value array<BR>
