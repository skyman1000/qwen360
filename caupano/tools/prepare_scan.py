"""Prepare one new scan through existing parsers and run the panorama batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan-dir',type=Path,required=True)
    p.add_argument('--work-root',type=Path,default=Path('qwen_pano/outputs/caupano'))
    p.add_argument('--category-mapping',type=Path,default=Path('benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv'))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--sample-ids',type=Path)
    p.add_argument('--manifest-output',type=Path)
    p.add_argument('--split-root',type=Path,default=Path('benchmark_assets/Matterport3D_metadata/mp3d_skybox'))
    p.add_argument('--caption-root',type=Path,default=Path('benchmark_assets/Matterport3D_stitched_captions/mp3d_skybox'))
    p.add_argument('--split-policy',choices=['strict','defer-conflicts'],default='strict')
    args=p.parse_args()
    scan=args.scan_dir.resolve(); work=args.work_root.resolve(); mapping=args.category_mapping.resolve()
    root=Path(__file__).resolve().parents[3]
    paths={name:work/(name+'_pilot')/scan.name for name in ['camera','house','region','object_link','canonical']}
    logs=work/'scan_dataset'/scan.name/'prepare_logs'
    logs.mkdir(parents=True,exist_ok=True)
    required=['house_segmentations','region_segmentations','matterport_camera_intrinsics',
              'matterport_camera_poses','matterport_skybox_images','undistorted_camera_parameters',
              'undistorted_color_images','undistorted_depth_images','undistorted_normal_images']
    missing=[name for name in required if not (scan/(name+'.zip')).is_file()]
    if missing:
        raise FileNotFoundError('Download missing archives first: '+', '.join(missing))
    def run(module,arguments):
        print('Running '+module,flush=True)
        logfile=logs/(module.rsplit('.',1)[-1]+'.log')
        with logfile.open('w') as handle:
            result=subprocess.run([sys.executable,'-m','qwen_pano.caupano.'+module,*map(str,arguments)],
                                  cwd=root,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:
            print(logfile.read_text()[-6000:])
            raise SystemExit(f'Stopped at {module}; full log: {logfile}')
    index_report=paths['canonical']/'index_validation.json'
    if index_report.exists():
        report=json.loads(index_report.read_text())
        if report['status']!='PASS' or report['scan_id']!=scan.name:
            raise ValueError('Existing index did not pass for this scan')
        print('Reusing completed canonical index; inputs must be unchanged.',flush=True)
    else:
        for stage in ['camera','house']:
            run('data.matterport.'+stage+'_parser',['--scan-dir',scan,'--output',paths[stage]])
        # Compare camera count against the independently declared source house,
        # rather than assuming 18 images or a fixed panorama count for every scan.
        counts=json.loads((paths['house']/'parse_summary.json').read_text())['declared_counts']
        run('tools.validate_cameras',['--input',paths['camera'],'--expected-panoramas',counts['panoramas'],
                                     '--expected-observations',counts['images']])
        run('tools.validate_house',['--input',paths['house'],'--cameras',paths['camera'],'--category-mapping',mapping])
        run('data.matterport.region_parser',['--house',paths['house'],'--category-mapping',mapping,'--output',paths['region']])
        run('tools.validate_regions',['--input',paths['region'],'--category-mapping',mapping])
        run('data.matterport.object_linker',['--regions',paths['region'],'--category-mapping',mapping,'--output',paths['object_link']])
        run('tools.validate_object_links',['--input',paths['object_link']])
        run('data.canonical_index',['--cameras',paths['camera'],'--objects',paths['object_link'],'--output',paths['canonical']])
        run('tools.validate_canonical_index',['--input',paths['canonical'],'--expected-panoramas',counts['panoramas'],
                                             '--expected-observations',counts['images']])
    # Stream batch progress rather than hide this longer stage inside a log.
    extra=[]
    if args.sample_ids:extra+=['--sample-ids',str(args.sample_ids.resolve())]
    if args.manifest_output:extra+=['--manifest-output',str(args.manifest_output.resolve())]
    subprocess.run([sys.executable,'-m','qwen_pano.caupano.tools.build_scan_dataset',
                    '--index',str(paths['canonical']),'--work-root',str(work),'--threads',str(args.threads),
                    '--split-policy',args.split_policy,'--split-root',str(args.split_root.resolve()),
                    '--caption-root',str(args.caption_root.resolve()),*extra],
                   cwd=root,check=True)


if __name__=='__main__':
    main()
