"""Verify generated runner/auditor exactly, without writing either file."""
import runpy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    captured = {}
    def capture(path, data, **kwargs):
        captured[path] = data
        return len(data)
    with patch.object(Path, 'exists', return_value=False), patch.object(Path, 'write_text', new=capture):
        runpy.run_path(str(ROOT/'scripts/prepare_projected_label_loss_recent_pilot.py'), run_name='__main__')
    assert len(captured) == 2
    for path, source in captured.items():
        assert path.read_text(encoding='utf-8') == source, path
        compile(source, str(path), 'exec')
        print(path.name, 'exact generation and syntax verified')
