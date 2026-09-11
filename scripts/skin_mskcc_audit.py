"""Independent scalar CIEDE2000 + replay audit; source validation only."""
import json
import math as m
from pathlib import Path
import joblib
import numpy as np
from skin_mskcc_data import ROOT, MANIFEST, PROTOCOL, load_source, sha


def scalar_de(p, q):
    L1, a1, b1 = map(float, p); L2, a2, b2 = map(float, q)
    cb = (m.hypot(a1,b1)+m.hypot(a2,b2))/2
    g = .5 * (1-m.sqrt(cb**7/(cb**7+25**7)))
    a1 *= 1+g; a2 *= 1+g
    c1, c2 = m.hypot(a1,b1), m.hypot(a2,b2)
    h1 = m.degrees(m.atan2(b1,a1)) % 360 if c1 else 0.
    h2 = m.degrees(m.atan2(b2,a2)) % 360 if c2 else 0.
    dl, dc = L2-L1, c2-c1
    dh = h2-h1
    if c1*c2 == 0:
        dh = 0.
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    dH = 2*m.sqrt(c1*c2)*m.sin(m.radians(dh/2))
    lb, cp = (L1+L2)/2, (c1+c2)/2
    if c1*c2 == 0:
        hb = h1+h2
    elif abs(h1-h2) <= 180:
        hb = (h1+h2)/2
    elif h1+h2 < 360:
        hb = (h1+h2+360)/2
    else:
        hb = (h1+h2-360)/2
    cosine = lambda deg: m.cos(m.radians(deg))
    t = 1-.17*cosine(hb-30)+.24*cosine(2*hb)+.32*cosine(3*hb+6)-.20*cosine(4*hb-63)
    sl = 1+.015*(lb-50)**2/m.sqrt(20+(lb-50)**2)
    sc, sh = 1+.045*cp, 1+.015*cp*t
    rt = -2*m.sqrt(cp**7/(cp**7+25**7))*m.sin(m.radians(60*m.exp(-((hb-275)/25)**2)))
    return m.sqrt((dl/sl)**2+(dc/sc)**2+(dH/sh)**2+rt*(dc/sc)*(dH/sh))


def main():
    bench = ROOT/'docs/benchmarks/skin_mskcc_summary_v1'
    results = json.loads((bench/'results.json').read_bytes())
    for path, key in [(PROTOCOL,'protocol_sha256'),(MANIFEST,'manifest_sha256'),
                      (ROOT/'scripts/skin_mskcc_data.py','data_script_sha256'),
                      (ROOT/'scripts/skin_mskcc_summary_pilot.py','script_sha256'),
                      (ROOT/'src/luma_skin_vision/color.py','color_script_sha256')]:
        assert sha(path) == results[key]
    fixture = np.loadtxt(ROOT/'tests/fixtures/ciede2000_sharma.txt')
    reference_gap = max(abs(scalar_de(r[:3],r[3:6])-r[6]) for r in fixture)
    assert reference_gap < 5e-5
    rows, x, y, _ = load_source('validation')
    max_gap = 0.; cases = 0
    for model in results['models']:
        path = ROOT/'experiments/runs/skin_mskcc_summary_v1'/(model['name']+'.joblib')
        assert sha(path) == model['model_sha256']
        pred = joblib.load(path).predict(x)
        saved = np.load(path.with_name(model['name']+'_validation.npz'))
        assert np.array_equal(y,saved['target']) and np.array_equal(pred,saved['prediction'])
        errors = np.array([scalar_de(p,q) for p,q in zip(pred,y)])
        max_gap = max(max_gap,float(np.max(np.abs(errors-saved['delta_e00']))))
        order = np.lexsort((np.array([r['image'] for r in rows]),saved['risk']))
        for record in model['coverage']:
            k = m.ceil(record['coverage']*len(errors))
            e = sorted(float(errors[i]) for i in order[:k])
            mean = m.fsum(e)/len(e)
            median = e[len(e)//2] if len(e)%2 else (e[len(e)//2-1]+e[len(e)//2])/2
            gaps = [abs(mean-record['mean']),abs(median-record['median'])]
            for quantile,key in [(.9,'p90'),(.95,'p95')]:
                index = (len(e)-1)*quantile; lo = int(index)
                value = e[lo]+(e[min(lo+1,len(e)-1)]-e[lo])*(index-lo)
                gaps.append(abs(value-record[key]))
            gaps += [abs(sum(v>threshold for v in e)/len(e)-record[key]) for threshold,key in [(5,'above_5_fraction'),(10,'above_10_fraction')]]
            max_gap=max(max_gap,*gaps); cases += 1
    assert max_gap < 1e-10
    audit = {'status':'PASS','sharma_cases':len(fixture),'reference_rounding_gap_max':reference_gap,
             'exact_model_and_target_replays':len(results['models']), 'independent_coverage_cases':cases,
             'max_scalar_metric_gap':max_gap,'test_or_calibration_endpoints_decoded':False,
             'results_sha256':sha(bench/'results.json')}
    (bench/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
