"""Exercise training configuration without installing or running ML models."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, Mock, patch


def load_module():
    names = ['torch', 'torch.nn', 'torchvision', 'torchvision.transforms',
             'torch.utils', 'torch.utils.data', 'PIL']
    modules = {name: types.ModuleType(name) for name in names}
    modules['torch.nn'].Module = object
    modules['torch.nn'].MSELoss = Mock()
    modules['torch'].optim = Mock()
    modules['torch'].save = Mock()
    modules['torch.utils.data'].Dataset = object
    modules['torch.utils.data'].DataLoader = Mock()
    modules['PIL'].Image = Mock()
    for name in ('Compose', 'Resize', 'ToTensor'):
        setattr(modules['torchvision.transforms'], name, Mock())
    path = Path(__file__).resolve().parents[1] / 'Denoising_model.py'
    spec = importlib.util.spec_from_file_location('denoising_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TrainingConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_cli_forwards_user_paths_and_epoch_count(self):
        with patch.object(self.module, 'train') as train:
            self.module.main(['--clean-dir', 'clean images', '--noisy-dir', 'noisy images',
                              '--output', 'weights/model.pth', '--epochs', '2'])
        train.assert_called_once_with('clean images', 'noisy images', 'weights/model.pth', num_epochs=2)

    def test_invalid_epoch_count_stops_before_training(self):
        with patch.object(self.module, 'train') as train, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.module.main(['--clean-dir', 'clean', '--noisy-dir', 'noisy',
                                  '--output', 'model.pth', '--epochs', '0'])
        self.assertEqual(error.exception.code, 2)
        train.assert_not_called()
        with self.assertRaises(ValueError):
            self.module.train('clean', 'noisy', 'model.pth', num_epochs=0)

    def test_training_uses_requested_folders_and_creates_checkpoint_directory(self):
        module = self.module
        model = MagicMock()
        loss = MagicMock()
        loss.item.return_value = 0.25
        module.nn.MSELoss.return_value.return_value = loss
        module.DataLoader.return_value = [('noisy batch', 'clean batch')]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new directory' / 'model.pth'
            with patch.object(module, 'PairedImageDataset') as dataset, \
                 patch.object(module, 'DenoiseAutoencoder', return_value=model), \
                 contextlib.redirect_stdout(io.StringIO()):
                module.train('custom/clean', 'custom/noisy', output, num_epochs=2)
            self.assertEqual(dataset.call_args.kwargs['clean_dir'], 'custom/clean')
            self.assertEqual(dataset.call_args.kwargs['noisy_dir'], 'custom/noisy')
            self.assertEqual(model.call_count, 2)
            self.assertTrue(output.parent.is_dir())
            module.torch.save.assert_called_once_with(model.state_dict.return_value, output)


if __name__ == '__main__':
    unittest.main()
