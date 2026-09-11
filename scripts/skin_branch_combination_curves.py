"""Full integer-coverage curves and independent fixed-coverage aggregation checks."""
import csv,json,math
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_branch_combination import OUT


def main():
    summary=json.loads((OUT/'summary.json').read_bytes())
    cache=ROOT/'data/processed/skin_branch_combination_v1/predictions.npz'
    assert sha(cache)==summary['local_arrays_sha256']
    comparisons=0;maxgap=0.;points=0;path=OUT/'uncalibrated_risk_coverage.csv'
    with np.load(cache) as arrays,path.open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['protocol','method','seed','accepted','population','coverage','mean_delta_e00'])
        for r in summary['rows']:
            a=arrays[f"{r['protocol']}__{r['method']}__s{r['seed']}"];risk=a[:,4];error=a[:,3]
            order=sorted(range(len(a)),key=lambda i:(float(risk[i]),i));ordered=error[order]
            for k in range(1,len(a)+1):
                value=math.fsum(float(e) for e in ordered[:k])/k
                writer.writerow([r['protocol'],r['method'],r['seed'],k,len(a),k/len(a),value]);points+=1
            for row in r['coverage']:
                n=int(math.ceil(row['coverage']*len(a)));assert n==row['accepted']
                e=ordered[:n];mean=math.fsum(float(v) for v in e)/n
                checks={'mean':mean,'median':float(np.median(e)),'p95':float(np.quantile(e,.95)),
                        'above5':sum(float(v)>5 for v in e)/n,'above10':sum(float(v)>10 for v in e)/n}
                for k,v in checks.items():maxgap=max(maxgap,abs(v-row[k]))
                comparisons+=1
    assert comparisons==270 and maxgap<1e-12
    result={'status':'PASS','curve_points':points,'fixed_coverage_rows_checked':comparisons,'maximum_aggregation_gap':maxgap,
            'curves_sha256':sha(path),'parent_summary_sha256':sha(OUT/'summary.json'),'script_sha256':sha(Path(__file__))}
    (OUT/'curve_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
