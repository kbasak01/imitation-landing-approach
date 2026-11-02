import json, sys
S = sys.argv[1]
s0 = json.load(open(f'{S}/r1_heldout_rest.json')); s0.update(json.load(open(f'{S}/r1_heldout_100k.json')))
d = {0: s0, 1: json.load(open(f'{S}/r1_heldout_s1.json')), 2: json.load(open(f'{S}/r1_heldout_s2.json'))}
steps = sorted(set().union(*[set(v) for v in d.values()]))
print('step   ' + '  '.join(f'held_s{s} train50_s{s}' for s in d) + '   held_mean  hi-noise(k75-99)/low(k0-24) s2')
for st in steps:
    row = []
    for s in d:
        r = d[s].get(st)
        row.append(f"{r['heldout']['anchors']['loss']:.4f}   {r['train50']['anchors']['loss']:.4f}  " if r else '   -        -     ')
    hm = [d[s][st]['heldout']['anchors']['loss'] for s in d if st in d[s]]
    r2 = d[2].get(st)
    extra = f"{r2['heldout']['anchors']['k75_99']:.4f}/{r2['heldout']['anchors']['k0_24']:.4f}" if r2 else ''
    print(st, '  '.join(row), f'  {sum(hm)/len(hm):.4f}', extra)
for s in d:
    h = {st: d[s][st]['heldout']['anchors']['loss'] for st in d[s]}
    m = min(h, key=h.get); last = h[max(h)]
    print(f's{s}: min {h[m]:.4f} @ {m}, 100k {last:.4f} (+{100*(last/h[m]-1):.0f}%); gap@100k held-train50 {last-d[s][max(h)]["train50"]["anchors"]["loss"]:.4f}')
    print('   by bucket min->100k', {b: (round(min(d[s][st]['heldout']['anchors'][b] for st in d[s]),4), round(d[s][max(h)]['heldout']['anchors'][b],4)) for b in ('k0_24','k25_49','k50_74','k75_99')})
