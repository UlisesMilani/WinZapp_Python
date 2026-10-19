"""The one-time microphone noise-reduction explanation is not repeated."""

import wx

from core.utils import DEFAULT_SETTINGS
from main_window import settings as settings_module
from main_window.settings import SettingsMixin


class _I18n:
    def t(self, key):
        return {
            "noise_reduction_notice_title": "About noise reduction",
            "noise_reduction_notice_message": "The preview does not include processing.",
        }.get(key, key)


class _Window(SettingsMixin):
    def __init__(self, *, enabled=False, seen=False, background=False):
        self.settings = {
            "general": {
                "noise_reduction_enabled": enabled,
                "noise_reduction_notice_shown": seen,
            }
        }
        self.background_mode = background
        self.i18n = _I18n()
        self.saved = 0

    def save_settings(self):
        self.saved += 1


class _Dialog:
    def __init__(self, calls, parent, message, title, style):
        calls.append({
            "parent": parent,
            "message": message,
            "title": title,
            "style": style,
            "shown": False,
            "destroyed": False,
        })
        self.call = calls[-1]

    def ShowModal(self):
        self.call["shown"] = True
        return wx.ID_OK

    def Destroy(self):
        self.call["destroyed"] = True


def test_notice_is_shown_and_saved_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        settings_module.wx, "MessageDialog",
        lambda parent, message, title, style: _Dialog(calls, parent, message, title, style),
    )
    window = _Window()

    assert window._show_noise_reduction_notice() is True
    assert window.settings["general"]["noise_reduction_notice_shown"] is True
    assert window.saved == 1
    assert calls[0]["message"] == "The preview does not include processing."
    assert calls[0]["title"] == "About noise reduction"
    assert calls[0]["style"] & wx.OK
    assert calls[0]["shown"] and calls[0]["destroyed"]

    assert window._show_noise_reduction_notice() is False
    assert len(calls) == 1
    assert window.saved == 1


def test_startup_notice_only_shows_when_noise_reduction_is_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        settings_module.wx, "MessageDialog",
        lambda parent, message, title, style: _Dialog(calls, parent, message, title, style),
    )
    disabled = _Window(enabled=False)
    assert disabled._maybe_show_noise_reduction_notice() is False
    assert calls == []
    assert disabled.saved == 0

    enabled = _Window(enabled=True)
    assert enabled._maybe_show_noise_reduction_notice() is True
    assert len(calls) == 1
    assert enabled.saved == 1


def test_startup_notice_is_deferred_in_background_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        settings_module.wx, "MessageDialog",
        lambda parent, message, title, style: _Dialog(calls, parent, message, title, style),
    )
    window = _Window(enabled=True, background=True)

    assert window._maybe_show_noise_reduction_notice() is False
    assert calls == []
    assert window.settings["general"]["noise_reduction_notice_shown"] is False


def test_noise_reduction_notice_flag_is_a_false_default():
    assert DEFAULT_SETTINGS["general"]["noise_reduction_notice_shown"] is False
