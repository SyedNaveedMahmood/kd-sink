"""Finish independently checked cross-phase provenance and safe publication.

Waits for scientific completion, preserves a dirty repository, never schedules
GPU work and never turns an interrupted attempt into a complete observation.
"""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import time

from audit_mechanistic_scientific import read,write,sha,digest,envelope,require
from mechanistic_campaign_supervisor import atomic,exclusive,check_pins,utc
from e3_nodi_recovery import audited_summary,check_manifest_bytes,compatibility


def completion_gate(manifest,monitor,final):
    require(monitor['action']=='complete' and monitor['manifest_sha256']==manifest['file_sha256'] and
            final['status']=='COMPLETE_INDEPENDENTLY_AUDITED' and final['expected']==32 and
            final['manifest_sha256']==manifest['file_sha256'],'completion identity')
    require(set(final['states'])=={j['id'] for j in manifest['jobs']} and
            all(e['status']=='complete' for e in final['states'].values()) and
            final['reused_Adrita']==17 and final['new_Nodi']==15,'exact final scientific coverage')


def finish(path,reconstruction_pin,script_pin,publish):
    require(sha(Path(__file__))==script_pin,'finalizer source changed')
    document=read(path);manifest=envelope(document);root=Path(manifest['root']);external=root.parent
    manifest={**manifest,'file_sha256':sha(path)}
    with exclusive(root/'finalizer.lock'):
        while not (root/'MONITOR_COMPLETE.json').exists():
            atomic(external/'FINALIZER_STATUS.json',{'utc':utc(),'pid':os.getpid(),'status':'WAITING_FOR_32_AUDITED_BUNDLES','manifest_sha256':sha(path)})
            if (root/'MONITOR_DIAGNOSIS_REQUIRED.json').exists():
                atomic(root/'FINALIZATION_BLOCKED.json',{'status':'SCIENTIFIC_COMPUTATION_REQUIRES_DIAGNOSIS','utc':utc(),'no_completion_claim':True});return
            time.sleep(10)
        monitor=read(root/'MONITOR_COMPLETE.json')
        require(sha(monitor['final_audit'])==monitor['final_audit_sha256'],'final audit changed')
        final=read(monitor['final_audit']);completion_gate(manifest,monitor,final);check_pins(manifest)
        require(sha(external/'reconstruction01/RECOVERY_INVENTORY_COMPLETE.json')==reconstruction_pin,'original reconstruction seal changed')
        reconstruction=read(external/'reconstruction01/RECOVERY_INVENTORY_COMPLETE.json')
        output=root/'final_completion';output.mkdir(exist_ok=False)
        sources=[];summaries={};expected_items={}
        for job in manifest['jobs']:
            summary,audit,pair=audited_summary(manifest,job,final['states'][job['id']])
            summaries[job['id']]=summary;expected_items[job['panel']]=summary['items']
            sources.append({'job_id':job['id'],'phase':'E3','state':job['state'],'panel':job['panel'],
                'origin':final['states'][job['id']].get('origin','NodiPC'),'items':audit['items'],
                'protocol_sha256':audit['identity']['protocol_sha256'],'runtime_sha256':audit['identity']['runtime_sha256'],
                'source_weights_sha256':audit['identity']['source']['weights_sha256'],
                'manifest_sha256':audit['manifest_sha256'],'E2_dependency':pair})
        inventory=read(external/'reconstruction01/ALL_92_JOBS.json')['jobs']
        for row in inventory:
            if row['phase']=='E3':continue
            proof=row['selected'];bundle=Path(proof['attempt'])/'bundle';record=check_manifest_bytes(bundle)
            audit_path=Path(proof['independent_audit_path']);runner_path=Path(proof['runner_receipt_path'])
            for file in (audit_path,runner_path):
                require(sha(file)==reconstruction['files'][file.relative_to(external/'reconstruction01').as_posix()],'original dual audit changed')
            audit,runner=read(audit_path),read(runner_path)
            require(audit['status']==runner['status']=='PASS' and audit['items']==runner['items']==300 and
                audit['manifest_sha256']==runner['manifest_sha256']==sha(bundle/'manifest.json'),'original scientific completion')
            identity=audit['identity'];summary=read(bundle/'summary.json')
            lock=read(external/f"imported_adrita/campaign/input_provenance/{row['phase']}.approved.json")
            approved={'E1':'aa280a9a1244c7e00bff5e88fe342548ac587553788c24dd072eeab8542528f6',
                      'E2':'1ba9dff94bdd6735cb1cd1c871a655094a91bd1b96c746e9f26f14f942c3d586'}
            require(lock['sha256']==approved[row['phase']] and envelope(lock)['runtime']==envelope(read(manifest['original_lock']))['runtime'] and
                identity==record['identity']==summary['identity'] and identity['protocol_sha256']==lock['sha256'] and
                summary['items']==expected_items[row['panel']] and
                identity['source']['weights_sha256']==envelope(read(manifest['successor_lock']))['sources'][row['state']]['weights']['sha256'],
                'cross-phase source/ordering/own-seal mismatch')
            summaries[row['job_id']]=summary
            sources.append({'job_id':row['job_id'],'phase':row['phase'],'state':row['state'],'panel':row['panel'],'origin':'AdritaPC',
                'items':300,'protocol_sha256':identity['protocol_sha256'],'runtime_sha256':identity['runtime_sha256'],
                'source_weights_sha256':identity['source']['weights_sha256'],'manifest_sha256':audit['manifest_sha256']})
        require(len(sources)==92,'full cross-phase inventory')
        write(output/'ALL_PHASE_PORTABLE_JOIN.json',{'status':'PASS','sources':sources,'summaries':summaries,
            'compatibility':compatibility(read(manifest['original_lock']),read(manifest['successor_lock'])),
            'interpretation':'Each phase uses its own approved definition and each bundle its own original/successor runtime. Descriptive seed0, no independent seed replication.'})
        with (output/'MATCHED_SOURCE_COVERAGE.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['panel','state','weights_sha256','E1_available','E2_available','E3_available','E3_origin']);writer.writeheader()
            for job in manifest['jobs']:
                suffix=job['id'][3:]
                writer.writerow({'panel':job['panel'],'state':job['state'],
                    'weights_sha256':envelope(read(manifest['successor_lock']))['sources'][job['state']]['weights']['sha256'],
                    'E1_available':'E1_'+suffix in summaries,'E2_available':'E2_'+suffix in summaries,'E3_available':True,
                    'E3_origin':final['states'][job['id']].get('origin','NodiPC')})
        small={'status':'COMPLETE_INDEPENDENTLY_AUDITED_PORTABLE_E3','utc':utc(),'E1_original_valid':28,'E2_original_valid':32,
            'E3_valid_bundles':32,'E3_items':9600,'reused_Adrita_E3':17,'new_Nodi_E3':15,'scientific_field_diff':[],
            'final_audit_sha256':sha(monitor['final_audit']),'manifest_sha256':sha(path),
            'original_E3_digest':'5673eaefc30ff238ee338b0cff973032f6f2579e985592f0c837a04c6f24cf33',
            'successor_E3_digest':manifest['approved_lock_hashes']['E3'],
            'cross_phase_join_sha256':sha(output/'ALL_PHASE_PORTABLE_JOIN.json'),
            'original_vs_successor_provenance':sources[:32],
            'limits':'Exact fixed12-item agreement and full matching E2 checks; historical per-query residual norm tensors unavailable; original live correctness gates unchanged; seed0 descriptive, not independent replication.'}
        write(output/'FINAL_COMPLETION_REPORT.json',small)
        (output/'FINAL_COMPLETION_REPORT.md').write_text('E3 completed:32 independently audited state/panel bundles,9600 items;17 reused Adrita and15 fresh Nodi bundles. '
            'All32 matching E2 dependencies verified. Scientific inputs and intervention definitions are unchanged under the authorized successor runtime. '
            'ALL_PHASE_PORTABLE_JOIN.json checks all92 original logical jobs against their own seals and preserves full E1/E2/E3 summaries. '
            'MATCHED_SOURCE_COVERAGE.csv explicitly records unavailable E1 checkpoint pairs. No historical result was rewritten. '
            'Results remain descriptive seed0; cross-machine execution is explicitly documented.\n',encoding='utf-8')
        write(output/'DERIVED_INVENTORY.json',{'status':'PASS','files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
        if not publish:return
        repo=Path(manifest['repo'])
        def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
        if git('branch','--show-current')!='mechanistic-e0' or git('status','--porcelain'):
            atomic(root/'PUBLICATION_DEFERRED.json',{'utc':utc(),'reason':'Repository branch or local changes require operator reconciliation; complete external scientific reports preserved.'});return
        starting=git('rev-parse','HEAD')
        targets=['reports/e3_nodipc_final_completion_20261010.json','E3_NODIPC_COMPLETION.md',
            'implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_CORE.md','implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_S1.md',
            'implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_S4.md','NEXT_STEPS.md']
        (repo/targets[0]).write_text(__import__('json').dumps(small,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
        (repo/targets[1]).write_bytes((output/'FINAL_COMPLETION_REPORT.md').read_bytes())
        entry=f'\n\n## {utc()} - Codex authorized detached E3 completion finalizer\n\n- Starting commit{starting}. Complete32 independently audited E3 bundles:17 Adrita,15 Nodi,9600 items;all32 archived E2 dependencies. Own-seal rehash/audits and full92-job cross-phase provenance PASS;scientific field diff empty. Source hashes,original/successor locks and archived failures preserved. Exact command:scripts/finalize_e3_nodi_recovery.py --manifest {path} --reconstruction-sha256 {reconstruction_pin} --script-sha256 {script_pin} --publish;verification completed without changing scientific code or repeating model inference. External final reports:{output};inventorySHA{sha(output/"DERIVED_INVENTORY.json")}. No training/E4/E5/Stage09. Next:researcher review of descriptive seed0 scientific results. Milestone commit/push recorded in external PUBLICATION_RECEIPT.json.\n'
        for target in targets[2:5]:
            with (repo/target).open('ab') as stream:stream.write(entry.encode())
        next_file=repo/'NEXT_STEPS.md';content=next_file.read_text(encoding='utf-8')
        for line in ('Verify first new complete300-item bundle and sustained single-worker monitoring.',
                     'Finish15 missing confirmation bundles,32-state final audit/provenance/report/inventory.'):
            content=content.replace('- [ ] '+line,'- [x] '+line)
        next_file.write_text(content,encoding='utf-8',newline='\n')
        subprocess.run(['git','add','--',*targets],cwd=repo,check=True)
        subprocess.run(['git','diff','--cached','--check'],cwd=repo,check=True)
        subprocess.run(['git','commit','-m','docs(mechanistic): record audited NodiPC E3 completion'],cwd=repo,check=True)
        environment={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'Never'}
        result=subprocess.run(['git','push','origin','HEAD:mechanistic-e0'],cwd=repo,env=environment,capture_output=True,text=True)
        atomic(root/'PUBLICATION_RECEIPT.json',{'utc':utc(),'commit':git('rev-parse','HEAD'),'push_exit_code':result.returncode,
            'stdout':result.stdout,'stderr':result.stderr,'source_inventory_sha256':sha(output/'DERIVED_INVENTORY.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--reconstruction-sha256',required=True);parser.add_argument('--script-sha256',required=True);parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();finish(args.manifest.resolve(),args.reconstruction_sha256,args.script_sha256,args.publish)
