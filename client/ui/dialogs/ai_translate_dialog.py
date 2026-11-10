"""Translate window for a received message: the message's own text, or the
stored transcription of a voice message, translated through Google Translate
(its own consent, no account and no provider key).

The original text sits read-only above the result; nothing on the message is
changed and nothing is ever sent. Owned by the conversation panel
(ui/conversation_panel/ai_writing.py), which builds it, runs it modally and
tears it down. The window machinery is TextAIResultDialog's.
"""
import wx

from core.ai_media import gtranslate
from ui.dialogs.ai_text_result_dialog import TextAIResultDialog


class TranslateDialog(TextAIResultDialog):
    """Original text, target language and the translation."""

    def __init__(self, panel, identity, text, locked, config, app_settings):
        self.text = text
        super().__init__(panel, identity, "text", locked, config, app_settings)

    def title_key(self):
        return "translate_title"

    def _regenerate_key(self):
        return "translate_again"

    def _extra_controls(self, layout):
        layout.Add(wx.StaticText(self, label=self.i18n.t("translate_original")), 0, wx.LEFT, 8)
        self.original = wx.TextCtrl(self, value=self.text, style=wx.TE_READONLY | wx.TE_MULTILINE,
                                    size=(-1, 120), name=self.i18n.t("translate_original"))
        layout.Add(self.original, 0, wx.EXPAND | wx.ALL, 8)
        layout.Add(wx.StaticText(self, label=self.i18n.t("writing_target_label")), 0, wx.LEFT, 8)
        self.target = wx.Choice(self, choices=[self.i18n.t(key) for _code, key in gtranslate.TARGET_LANGUAGES],
                                name=self.i18n.t("writing_target_label"))
        default = gtranslate.target_for_app_language(self.i18n.language)
        self.target.SetSelection(next((i for i, (code, _key) in enumerate(gtranslate.TARGET_LANGUAGES)
                                       if code == default), 0))
        layout.Add(self.target, 0, wx.EXPAND | wx.ALL, 8)

    def _question(self):
        return "translate"

    def _consent(self):
        return self._consent_translate()

    def _worker(self, operation):
        target = gtranslate.TARGET_LANGUAGES[self.target.GetSelection()][0]
        text = self.text

        def work():
            operation.check()
            translated, _detected = gtranslate.translate_text(text, target,
                                                              cancel_check=operation.check)
            return translated, "Google Translate"
        return work
