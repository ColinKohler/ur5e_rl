import pickle

f1 = ''
f2 = ''

r1 = pickle.load(open(f1, 'rb'))
r2 = pickle.load(open(f2, 'rb'))

for i in range(r2['num_eps']):
  r1['buffer'][r1['num_eps']+i+1] = r2['buffer'][i]

r1['num_eps'] += r2['num_eps']
r1['num_steps'] += r2['num_steps']

breakpoint()
pickle.dump(r1, open('combined_replay_buffer.pkl'), 'wb'))
