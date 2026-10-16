"""Setup/run orchestration checks; do not install packages or start training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_portability import load,write

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
cloud=load('entrypoint_cloud_cli','scripts/cloud_cli.py')


class EntrypointTests(unittest.TestCase):
    def test_t4_setup_saves_configuration_and_run_uses_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(cloud,'PROJECT',root),patch.object(cloud,'detect_platform',return_value='colab'),patch.object(cloud,'detect_precision',return_value='float16'),patch.object(cloud,'mount_drive') as mount,patch.object(cloud,'initialize_data',return_value='fresh pinned model and dataset'),patch.object(cloud.subprocess,'run') as execute,patch.object(sys,'argv',['setup.py','--work-root',str(root/'runtime')]):
                cloud.setup_main();mount.assert_called_once()
                config=json.loads((root/'.cloud/session.json').read_text())
                self.assertTrue(config['ready']);self.assertFalse(config['start_from_export'])
                self.assertEqual(config['precision'],'float16')
                self.assertNotIn('--prebuilt-kernels',execute.call_args.args[0])
                self.assertFalse((root/'.cloud/setup_in_progress.json').exists())
            for phase in ['train','infer']:
                path=root/f'.venv/{phase}/bin/python';path.parent.mkdir(parents=True);path.touch()
            with patch.object(cloud,'PROJECT',root),patch.object(cloud.subprocess,'run') as execute,patch.object(sys,'argv',['run.py','--hours','5']):
                cloud.run_main()
                command=execute.call_args.args[0]
                self.assertEqual(command[command.index('--precision')+1],'float16')
                self.assertEqual(command[command.index('--hours')+1],'5.0')
                self.assertEqual(execute.call_args.kwargs['env']['HF_HOME'],config['hf_home'])

    def test_bf16_defaults_to_exported_seed1_and_kaggle_uses_working_vault(self):
        self.assertTrue(cloud.make_config('colab','bfloat16',1)['start_from_export'])
        config=cloud.make_config('kaggle','float16',1)
        self.assertFalse(config['start_from_export'])
        self.assertEqual(config['vault'],'/kaggle/working/vault_float16_seed1')
        self.assertEqual(config['work_root'],'/tmp/rlvr_runtime')
        with self.assertRaisesRegex(ValueError,'seed 1'):cloud.make_config('local','bfloat16',42,start_from_export=True)

    def test_failed_setup_blocks_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(cloud,'PROJECT',root),patch.object(cloud.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'installer')),patch.object(sys,'argv',['setup.py','--platform','local','--precision','float16']):
                with self.assertRaises(subprocess.CalledProcessError):cloud.setup_main()
            self.assertFalse((root/'.cloud/session.json').exists())
            with patch.object(cloud,'PROJECT',root),patch.object(sys,'argv',['run.py']):
                with self.assertRaisesRegex(RuntimeError,'Setup is incomplete'):cloud.run_main()

    def test_existing_run_is_reused_without_downloading_or_preparing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=cloud.make_config('local','float16',1,root/'runtime',root/'vault')
            write(Path(config['run_dir'])/'inputs.json',{'unchanged':True})
            with patch.object(cloud.subprocess,'run') as execute:
                self.assertEqual(cloud.initialize_data(config),'existing local progress')
                execute.assert_not_called()


if __name__=='__main__':unittest.main()
