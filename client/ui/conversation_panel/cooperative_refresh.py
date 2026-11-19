"""Cooperative message refresh ownership and mutation boundary.

The driver owns just one generation. Any ordinary row mutation cancels that
work first, keeping its already-applied rows as the baseline for the next
refresh. Incoming data is never dropped, and the editor is never disabled.
"""
from functools import wraps
import logging
import wx
from core.cooperative_rows import CooperativeRows


def cancel_refresh(panel):
    job = getattr(panel, '_cooperative_rows_job', None)
    if job is not None and not job.running:
        panel._cooperative_rows_job = None
        job.cancel()
        panel._populating_messages = False
        panel._messages_signature_cache = None
        return True
    return False


def row_mutation(method):
    """Cancel a suspended generation before another path writes native rows."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        cancelled = cancel_refresh(self)
        try:
            return method(self, *args, **kwargs)
        finally:
            if cancelled:
                wx.CallAfter(self.refresh_messages_if_changed)
    return guarded


def start_refresh(panel, preserve_focus):
    """Coalesce background refreshes; navigation always invalidates the job."""
    old = getattr(panel, '_cooperative_rows_job', None)
    if old is not None and preserve_focus:
        panel._cooperative_refresh_pending = True
        return
    cancel_refresh(panel)
    panel._cooperative_refresh_pending = False
    generation = getattr(panel, '_cooperative_rows_generation', 0) + 1
    panel._cooperative_rows_generation = generation
    chat = (panel.conversation or {}).get('remoteJid', '')
    control = panel.messages_list

    def valid():
        return (panel._cooperative_rows_generation == generation
                and (panel.conversation or {}).get('remoteJid', '') == chat
                and panel.messages_list is control)

    def finished():
        if getattr(panel, '_cooperative_rows_job', None) is not job:
            return
        panel._cooperative_rows_job = None
        panel._populating_messages = False
        from ui.conversation_panel.typing_row import sync_typing_row
        sync_typing_row(panel)
        if panel._cooperative_refresh_pending:
            panel._cooperative_refresh_pending = False
            panel._messages_signature_cache = None
            wx.CallAfter(panel.refresh_messages_if_changed)

    def run_step():
        if getattr(panel, '_cooperative_rows_job', None) is not job:
            return
        if not valid():
            cancel_refresh(panel)
            return
        control.Freeze()
        try:
            job.step()
        except Exception:
            logging.exception('[cooperative-refresh] failed')
            cancel_refresh(panel)
            raise
        finally:
            control.Thaw()

    def schedule(callback):
        # A new timer, not Yield(): input and queued navigation can run first.
        wx.CallLater(1, run_step)

    from ui.conversation_panel.list_refresh import ListRefreshMixin
    iterator = ListRefreshMixin._populate_messages_steps(panel, preserve_focus)
    job = CooperativeRows(iterator, schedule, finished, valid)
    panel._cooperative_rows_job = job
    run_step()
