import json

for fname in [r'd:\downloads\01.PMV\PMV - NoodleDude Megamix.alpha.funscript', r'd:\downloads\01.PMV\PMV - NoodleDude Megamix.beta.funscript']:
    with open(fname, 'r') as f:
        data = json.load(f)
    actions = data['actions']
    # 11:00.545 = 660545ms, get actions in range 659000-663000
    region = [a for a in actions if 659000 <= a['at'] <= 663000]
    short = fname.split('\\')[-1]
    print(f'\n=== {short} ===')
    print(f'Actions in 659000-663000ms ({len(region)} actions):')
    for a in region:
        t_sec = a['at'] / 1000
        mins = int(t_sec // 60)
        secs = t_sec % 60
        print(f'  {mins}:{secs:06.3f}  at={a["at"]}  pos={a["pos"]}')
    
    # Also compute intervals between actions in this region
    if len(region) > 1:
        intervals = [region[i+1]['at'] - region[i]['at'] for i in range(len(region)-1)]
        print(f'  Intervals (ms): {intervals}')
        print(f'  Min interval: {min(intervals)}ms, Avg: {sum(intervals)/len(intervals):.1f}ms')
