"""Path-only private adapter. Dry preflight by default; --execute is explicit."""
import argparse, importlib.util, json, os, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LEGACY=ROOT/'src/legacy'
sys.path.insert(0,str(LEGACY))
def resolve(config):
    path=Path(config).resolve();c=json.loads(path.read_text(encoding='utf-8'))
    if set(c)!={'pwv_dir','radar_dir','dem','checkpoint','output_dir'}:raise ValueError('Only path configuration is supported; training constants remain original')
    return {k:str((ROOT/v).resolve()) for k,v in c.items()}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['phase1','phase2','evaluate'],required=True);p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();c=resolve(a.config)
    for key in ('pwv_dir','radar_dir'):
        if not Path(c[key]).is_dir():raise FileNotFoundError(key+': '+c[key])
    if not Path(c['dem']).is_file():raise FileNotFoundError('DEM required, no silent fallback')
    if a.stage!='phase1' and not Path(c['checkpoint']).is_file():raise FileNotFoundError('Trusted checkpoint required; random evaluation refused')
    output=Path(c['output_dir']);source=Path(__file__).resolve().parents[1]/'src'
    if output==ROOT or ROOT in output.parents and output.relative_to(ROOT).parts[0]!='runs':raise ValueError('Within repository only runs/ may be written')
    if output.exists() and any(output.iterdir()):raise FileExistsError('New empty output_dir required; old outputs protected')
    print(json.dumps({'stage':a.stage,'paths':c,'execution_enabled':a.execute},indent=2))
    if not a.execute:return
    # Private module: change dataset root arguments only, not indexing, stride or formulas.
    import dataset
    fn={'phase1':'train_phase1','phase2':'train_phase2','evaluate':'test'}[a.stage]
    spec=importlib.util.spec_from_file_location('_release_'+fn,LEGACY/(fn+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def data(*args,**kwargs):
        if args:args=(c['pwv_dir'],c['radar_dir'],*args[2:])
        else:kwargs.update(pwv_dir=c['pwv_dir'],radar_dir=c['radar_dir'])
        return dataset.WeatherDataset(*args,**kwargs)
    module.WeatherDataset=data
    output.mkdir(parents=True,exist_ok=True);shutil.copyfile(c['dem'],output/'dem_64x64.pt')
    if a.stage!='phase1':
        target=output/('checkpoints/best_model.pth' if a.stage=='phase2' else 'checkpoints_gan/best_gan_model.pth')
        target.parent.mkdir();shutil.copyfile(c['checkpoint'],target)
    previous=Path.cwd()
    try:os.chdir(output);module.main()
    finally:os.chdir(previous)
if __name__=='__main__':main()
