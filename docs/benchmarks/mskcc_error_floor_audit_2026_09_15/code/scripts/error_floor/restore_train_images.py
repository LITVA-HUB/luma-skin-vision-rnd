"""Restore only original TRAIN photographs for a read-only capture audit."""
import concurrent.futures,hashlib,json,sys,time,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from skin_mskcc_data import read,patient_roles,RAW,sha
sites={r['tag_id']:r for r in read('s2.csv')};people={r['patient_id']:r for r in read('s1.csv')}
src=[r for r in read('s7.csv') if r['isic_id']!='not-available'];roles=patient_roles({sites[r['tag_id']]['patient_id']:people[sites[r['tag_id']]['patient_id']]['dermatoscope'] for r in src})
ids=sorted(r['isic_id'] for r in src if roles[sites[r['tag_id']]['patient_id']]=='train');assert len(ids)==966
for sub in ['api','images']:(RAW/sub).mkdir(exist_ok=True)
def get(url,cap):
 import urllib.request
 for attempt in range(3):
  try:
   with urllib.request.urlopen(url,timeout=45) as r:
    b=r.read(cap+1);h=dict(r.headers)
   if len(b)>cap:raise ValueError('size cap')
   return b,h
  except (OSError,TimeoutError):
   if attempt==2:raise
   time.sleep(attempt+1)
def one(image):
 api=RAW/'api'/f'{image}.json'
 if not api.exists():
  b,_=get('https://api.isic-archive.com/api/v2/images/'+image+'/',100000);info=json.loads(b)
  if info['isic_id']!=image or info['copyright_license']!='CC-BY':raise ValueError('identity/license')
  api.write_bytes(b)
 info=json.loads(api.read_bytes());assert info['isic_id']==image and info['copyright_license']=='CC-BY'
 url=info['files']['full']['url'];assert url.startswith('https://isic-archive.s3.amazonaws.com/images/')
 out=RAW/'images'/f'{image}.jpg';header=RAW/'api'/f'{image}_s3.json'
 if not out.exists():
  b,h=get(url,15000000);hl={k.lower():v for k,v in h.items()};etag=hl['etag'].strip('"')
  assert len(b)==int(hl['content-length']) and b.startswith(b'\xff\xd8') and hashlib.md5(b).hexdigest()==etag
  # Unique staging names also make an interrupted/restarted restoration safe.
  header.write_text(json.dumps(h))
  with tempfile.NamedTemporaryFile(dir=out.parent, suffix='.part', delete=False) as stream:
   stream.write(b);tmp=Path(stream.name)
  tmp.replace(out)
 h={k.lower():v for k,v in json.loads(header.read_bytes()).items()};b=out.read_bytes()
 assert len(b)==int(h['content-length']) and hashlib.md5(b).hexdigest()==h['etag'].strip('"')
 return {'image':image,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'license':'CC-BY','url':url}
t=time.perf_counter();receipts=[];failures=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
 futures={pool.submit(one,image):image for image in ids}
 for f in concurrent.futures.as_completed(futures):
  try:receipts.append(f.result())
  except Exception as e:
   failures.append({'image':futures[f],'error':str(e)})
   print(json.dumps({'failure_number':len(failures),'error':str(e)}),flush=True)
  if (len(receipts)+len(failures))%25==0:print(json.dumps({'verified':len(receipts),'failed':len(failures),'seconds':time.perf_counter()-t}),flush=True)
(RAW/'audit_train_image_receipts.private.json').write_text(json.dumps(receipts,indent=2))
(RAW/'audit_train_image_failures.private.json').write_text(json.dumps(failures,indent=2))
print(json.dumps({'verified':len(receipts),'failed':len(failures),'seconds':time.perf_counter()-t,'bytes':sum(x['bytes'] for x in receipts)}),flush=True)
if failures:raise SystemExit(1)
