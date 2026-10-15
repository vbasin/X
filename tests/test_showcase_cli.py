"""Launcher contracts: fail before side effects and own child shutdown."""
import socket
import subprocess
from unittest.mock import Mock

import pytest

from src.showcase import _free_port, _stop, main

pytestmark = pytest.mark.showcase


def test_busy_port_is_reported_without_starting_server():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        with pytest.raises(ValueError, match='in use'):
            _free_port(listener.getsockname()[1])


def test_invalid_port_rejected():
    with pytest.raises(ValueError, match='between'):
        _free_port(0)


def test_prepare_refuses_existing_output(tmp_path):
    assert main(['prepare', '--output', str(tmp_path)]) == 1
    assert not list(tmp_path.iterdir())


def test_launcher_stops_owned_process_group(monkeypatch):
    import src.showcase as launcher
    process = Mock(spec=subprocess.Popen)
    process.pid = 98765
    process.poll.return_value = None
    kill = Mock()
    monkeypatch.setattr(launcher.os, 'killpg', kill)
    monkeypatch.setattr(launcher.os, 'name', 'posix')
    _stop(process)
    kill.assert_called_once_with(98765, launcher.signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=10)
