"""AIWritingMixin — part of ConversationsPanel (see ui/conversation_panel/__init__.py).

Writing help on the draft in the composer (fix, rephrase, restyle, translate)
and translation of a received message's text or stored voice-message
transcription. The windows live in ui/dialogs/ai_writing_dialog.py and
ui/dialogs/ai_translate_dialog.py; what a provider or Google may see is each
window's own consent (core/ai_media/config.py keeps the settings).

The composer context menu offers writing help on any non-empty draft short
enough to send; the message menu offers Translate on any message with usable
text. Translation needs no provider key, so neither entry checks the AI
providers' own on/off switches.

Methods run with ``self`` bound to the ConversationsPanel instance.
"""
from app_paths import active_account_id, global_dir
from app_settings import AppSettings
from core.ai_media.config import MAX_SOURCE_BYTES, preferences
from core.transcription import stored as stored_transcription


def _message_text(msg):
    """The translatable text of a message: its own text, or the stored
    transcription of a voice message; "" when there is nothing usable. A
    view-once message's text is never offered for AI (same rule as
    core/ai_media's eligible_kind), so it is not offered for translation
    either."""
    if not isinstance(msg, dict):
        return ""
    key = msg.get("key")
    key = key if isinstance(key, dict) else {}
    if not isinstance(key.get("id"), str) or not key.get("id"):
        return ""
    info = msg.get("message")
    info = info if isinstance(info, dict) else {}
    text = info.get("conversation")
    if not isinstance(text, str) or not text.strip():
        extended = info.get("extendedTextMessage")
        extended = extended if isinstance(extended, dict) else {}
        text = extended.get("text") or ""
    if not isinstance(text, str):
        text = ""
    text = text.strip()
    if text:
        return text
    record = stored_transcription.saved_transcription(msg)
    if record:
        return str(record.get("text") or "").strip()
    return ""


class AIWritingMixin:
    """Writing help on the draft and translation of a received message."""

    def _ai_prefs(self):
        mw = self.main_window
        app = getattr(mw, "_app_settings", None) or AppSettings(global_dir())
        return app, preferences(app)

    def _writing_help_offered(self, text=None):
        """The composer menu offers writing help on a non-empty draft short
        enough to send; translation works without any provider key."""
        if text is None:
            field = getattr(self, "message_field", None)
            if field is None:
                return False
            text = field.GetValue()
        text = text.strip()
        return bool(text) and len(text.encode("utf-8")) <= MAX_SOURCE_BYTES["writing"]

    def _on_writing_help(self, event=None):
        """The composer menu item: open the writing-help window on the draft."""
        existing = getattr(self, "_writing_dialog", None)
        if existing is not None:
            existing.Raise()
            return  # No second window or duplicate request.
        draft = self.message_field.GetValue().strip()
        if not draft:
            return
        mw = self.main_window
        app, config = self._ai_prefs()
        locked = mw.is_chat_locked((self.conversation or {}).get("remoteJid", ""))
        from ui.dialogs.ai_writing_dialog import WritingHelpDialog
        dialog = WritingHelpDialog(self, draft, locked, config, app)
        self._writing_dialog = dialog
        try:
            dialog.ShowModal()
        finally:
            dialog._dispose()
            self._writing_dialog = None
            dialog.Destroy()
            if mw.IsShown() and not getattr(mw, "_shutting_down", False):
                self.message_field.SetFocus()

    def _translate_offered(self, msg):
        """The message menu offers Translate when the message has usable text."""
        return bool(_message_text(msg))

    def _on_translate_message(self, event=None, message=None):
        """The message menu item: open the translate window on the message's text."""
        if message is None:
            message = self._focused_message()
        if message is None:
            return
        text = _message_text(message)
        if not text:
            return
        existing = getattr(self, "_translate_dialog", None)
        if existing is not None:
            existing.Raise()
            return
        mw = self.main_window
        app, config = self._ai_prefs()
        jid = (self.conversation or {}).get("remoteJid", "")
        locked = mw.is_chat_locked(jid)
        identity = (active_account_id(), jid, message["key"]["id"])
        from ui.dialogs.ai_translate_dialog import TranslateDialog
        dialog = TranslateDialog(self, identity, text, locked, config, app)
        self._translate_dialog = dialog
        try:
            dialog.ShowModal()
        finally:
            dialog._dispose()
            self._translate_dialog = None
            dialog.Destroy()
            self._restore_focus_after_ai(identity, locked)
