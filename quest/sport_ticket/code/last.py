data['last_match'] = 0

data['last_match'][((data['match_num']==33) | (data['match_num']==34) & (data['stage'] == 'j1'))] =1
data['last_match'][((data['match_num']==41) | (data['match_num']==42) & (data['stage'] == 'j1'))] =1
