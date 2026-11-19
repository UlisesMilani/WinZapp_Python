"""Deterministic scheduling tests, no wx windows or desktop interaction."""
import ast
from pathlib import Path
import types
import pytest
from core.cooperative_rows import CooperativeRows
from core.list_row_diff import plan_row_diff

ROOT = Path(__file__).resolve().parents[1]


def method(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(f for c in tree.body if isinstance(c, ast.ClassDef)
                for f in c.body if isinstance(f, ast.FunctionDef) and f.name == name)
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, 'exec'), namespace)
    return namespace[name]


class Clock:
    def __init__(self): self.now = 0
    def __call__(self): return self.now
    def tick(self): self.now += .003


class ListControl:
    def __init__(self, rows, clock):
        self.rows = list(rows); self.clock = clock; self.frozen = 0
        self.focus = 'composer'; self.draft = 'writing a reply'; self.selection = 0
    def Freeze(self): self.frozen += 1
    def Thaw(self): self.frozen -= 1
    def GetItemText(self, i): self.clock.tick(); return self.rows[i]
    def SetItemText(self, i, text): self.clock.tick(); self.rows[i] = text
    def InsertItem(self, i, text): self.clock.tick(); self.rows.insert(i, text)
    def DeleteItem(self, i): self.clock.tick(); del self.rows[i]


def make_panel(rows, clock):
    ns = dict(plan_row_diff=plan_row_diff,
              message_row_count=lambda p:len(p.messages_list.rows),
              row_text_unchanged=lambda a,b:a==b)
    write = method('client/ui/conversation_panel/message_rows.py', '_sync_message_rows_steps', ns)
    panel = types.SimpleNamespace(_sorted_messages=list(rows),
        messages_list=ListControl(rows, clock), _message_row_key=lambda m:m,
        _render_message_line=lambda m,**kw:(clock.tick() or m))
    panel.write = types.MethodType(write, panel)
    return panel


def run_transition(panel, target, clock, interrupt=None):
    queue=[];done=[];steps=0
    iterator=panel.write(list(panel._sorted_messages),target)
    job=CooperativeRows(iterator, queue.append, lambda:done.append(True),clock=clock)
    job.step()
    while queue:
        assert panel.messages_list.frozen == 0
        assert panel.messages_list.rows == panel._sorted_messages
        assert panel.messages_list.focus == 'composer'
        assert panel.messages_list.draft == 'writing a reply'
        steps+=1
        if interrupt and interrupt(job,steps): break
        before=clock.now;queue.pop(0)()
        assert clock.now-before <= .014  # one native op may cross the 8ms budget
    return job,queue,done,steps


def test_1692_to_13005_and_back_never_blocks_one_ui_turn():
    clock=Clock();p=make_panel(list(range(1692)),clock)
    job,q,done,steps=run_transition(p,list(range(13005)),clock)
    assert done and steps>1000 and len(p.messages_list.rows)==13005
    job,q,done,steps=run_transition(p,list(range(1691)),clock)
    assert done and steps>1000 and p.messages_list.rows==list(range(1691))


def test_navigation_invalidates_scheduled_old_generation():
    clock=Clock();p=make_panel(list(range(1692)),clock)
    def navigate(job,steps):
        if steps==800:
            job.cancel();return True
    job,q,done,steps=run_transition(p,list(range(13005)),clock,navigate)
    # New chat starts with the actual partial native rows, not the old target.
    target=['new chat message']
    run_transition(p,target,clock)
    for callback in q: callback()
    assert not done and p.messages_list.rows==target and job.cancelled


def test_repeated_refresh_cancels_without_stale_writes_or_lost_rows():
    clock=Clock();p=make_panel(list(range(1692)),clock)
    def update(job,steps):
        if steps==1000: job.cancel();return True
    job,q,done,_=run_transition(p,list(range(13005)),clock,update)
    target=list(range(13007))
    run_transition(p,target,clock)
    for callback in q:callback()
    assert p.messages_list.rows==target and not done


def test_mutation_boundary_and_refresh_coalescing_are_wired():
    tree=ast.parse((ROOT/'client/ui/conversation_panel/list_refresh.py').read_text())
    funcs={f.name:f for c in tree.body if isinstance(c,ast.ClassDef)
           for f in c.body if isinstance(f,ast.FunctionDef)}
    source=ast.unparse(funcs['refresh_messages_if_changed'])
    assert source.index('_cooperative_rows_job')<source.index('self._messages_signature()')
    assert '_cooperative_refresh_pending = True' in source
    for file,name in [('conversation_navigation.py','navigate_to_conversation'),
                      ('text_sending.py','_send_new_text_message'),
                      ('list_refresh.py','on_incoming_message')]:
        s=(ROOT/'client/ui/conversation_panel'/file).read_text()
        assert '@row_mutation\n    def '+name in s


def test_small_window_preserves_sync_path():
    ns={'start_refresh':lambda *a:pytest.fail('small refresh was deferred')}
    populate=method('client/ui/conversation_panel/list_refresh.py','populate_messages',ns)
    calls=[]
    p=types.SimpleNamespace(conversation={'messages':{'messages':{'records':[1]}}},
        _sorted_messages=[1],_populate_messages_steps=lambda flag:(calls.append(flag) or iter([None])))
    ns['ListRefreshMixin']=types.SimpleNamespace(_populate_messages_steps=lambda panel, flag:panel._populate_messages_steps(flag))
    populate(p,True)
    assert calls==[True]


def load_coordinator(monkeypatch):
    import importlib.util
    import sys
    queue=[]
    wx=types.SimpleNamespace(CallLater=lambda delay, f:queue.append(f),
                             CallAfter=lambda f:queue.append(f))
    monkeypatch.setitem(sys.modules,'wx',wx)
    spec=importlib.util.spec_from_file_location('coordinator_under_test',
        ROOT/'client/ui/conversation_panel/cooperative_refresh.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,queue


def test_coalesces_many_requests_and_drops_cancelled_callbacks(monkeypatch):
    import sys
    module,queue=load_coordinator(monkeypatch)
    clock=Clock();p=make_panel(list(range(1692)),clock)
    p.conversation={'remoteJid':'chat-a'}
    p._cooperative_refresh_pending=False
    refreshes=[];p.refresh_messages_if_changed=lambda:refreshes.append(True)
    def populate(panel,preserve):
        yield from panel.write(list(panel._sorted_messages),list(range(13005)))
    monkeypatch.setitem(sys.modules,'ui.conversation_panel.list_refresh',
        types.SimpleNamespace(ListRefreshMixin=types.SimpleNamespace(_populate_messages_steps=populate)))
    monkeypatch.setitem(sys.modules,'ui.conversation_panel.typing_row',types.SimpleNamespace(sync_typing_row=lambda p:None))
    module.start_refresh(p,True);first=p._cooperative_rows_job
    for _ in range(30):module.start_refresh(p,True)
    assert p._cooperative_rows_job is first and len(queue)==1
    # Cancel via a real synchronous mutation decorator, then drain stale work.
    @module.row_mutation
    def navigate(panel):panel.conversation={'remoteJid':'chat-b'}
    navigate(p)
    baseline=list(p.messages_list.rows)
    while queue:queue.pop(0)()
    assert first.cancelled and p.messages_list.rows==baseline and len(refreshes)==1


def test_latest_background_request_gets_one_followup(monkeypatch):
    import sys
    module,queue=load_coordinator(monkeypatch)
    p=types.SimpleNamespace(conversation={'remoteJid':'chat-a'},messages_list=types.SimpleNamespace(Freeze=lambda:None,Thaw=lambda:None))
    refreshes=[];p.refresh_messages_if_changed=lambda:refreshes.append(True)
    def populate(panel,preserve):
        for _ in range(100):yield
    monkeypatch.setitem(sys.modules,'ui.conversation_panel.list_refresh',
        types.SimpleNamespace(ListRefreshMixin=types.SimpleNamespace(_populate_messages_steps=populate)))
    monkeypatch.setitem(sys.modules,'ui.conversation_panel.typing_row',types.SimpleNamespace(sync_typing_row=lambda p:None))
    module.start_refresh(p,True)
    for _ in range(30):module.start_refresh(p,True)
    while queue:queue.pop(0)()
    assert refreshes==[True] and p._cooperative_rows_job is None


def test_real_population_generator_keeps_partial_model_and_editor(monkeypatch):
    """Run the actual population wrapper/finalizers, not just its row writer."""
    import logging
    import sys
    clock=Clock();rows=[{'key':{'id':str(i)}} for i in range(1692)]
    p=make_panel(rows,clock);control=p.messages_list
    control.GetFocusedItem=lambda:0
    control.GetFirstSelected=lambda:0
    control.GetItemCount=lambda:len(control.rows)
    control.GetTopItem=lambda:0
    control.EnsureVisible=lambda i:None
    control.Focus=lambda i:None;control.Select=lambda i:None
    p.conversation={'remoteJid':'chat-a','messages':{'messages':{'records':rows}}}
    p._is_separator=lambda m:m.get('_type')=='unread_separator'
    p._focused_msg_id=lambda:'0'
    p.refresh_poll_votes=lambda:None;p._apply_pinned_message_flags=lambda:None
    p._clear_populating_messages_flag=lambda:None
    p._refresh_expanded_window_before_rebuild=lambda:None
    p._sorted_deduped_records=lambda r:r
    p._reaction_map_from_sorted=lambda r:{}
    p._is_displayable_message=lambda m:True
    p._place_unread_separator_for_rebuild=lambda r:r
    p._history_window_for_rebuild=lambda r,limit:(0,-1)
    p._remember_expanded_window=lambda:None
    p._messages_signature=lambda:tuple(m['key']['id'] for m in p._sorted_messages)
    p._update_focused_row_controls=lambda i:None
    p._message_row_key=lambda m:m['key']['id']
    p._render_message_line=lambda m,**kw:(clock.tick() or m['key']['id'])
    control.rows=[m['key']['id'] for m in rows]
    p.main_window=types.SimpleNamespace(settings={})
    queue=[]
    fakewx=types.SimpleNamespace(Window=types.SimpleNamespace(FindFocus=lambda:'composer'),
        CallAfter=lambda f:queue.append(f),CallLater=lambda delay,f:queue.append(f))
    ns={'wx':fakewx,'time':types.SimpleNamespace(monotonic=clock),'logging':logging,
        'message_row_count':lambda p:len(control.rows),'sync_typing_row':lambda p:None}
    steps=method('client/ui/conversation_panel/list_refresh.py','_populate_messages_steps',ns)
    writer=types.SimpleNamespace(_sync_message_rows_steps=lambda panel,old,new:panel.write(old,new))
    monkeypatch.setitem(sys.modules,'ui.conversation_panel.message_rows',types.SimpleNamespace(MessageRowsMixin=writer))
    p._cooperative_rows_job=object()
    target=rows+[{'key':{'id':str(i)}} for i in range(1692,13005)]
    p.conversation['messages']['messages']['records']=target
    it=steps(p,True)
    for _ in range(2500):
        next(it)
        assert control.frozen==0
        assert len(control.rows)==len(p._sorted_messages)
    # Cancel while writing thousands of rows. Finalization must balance Freeze
    # and must not cache the as-yet-unpainted target as though it were visible.
    it.close()
    assert control.frozen==0 and p._messages_signature_cache is None
    assert control.draft=='writing a reply' and control.focus=='composer'
    # Complete a fresh generation after the cancel. A user moves selection
    # and scrolls while it runs; finishing must not put either one back.
    control.EnsureVisible=lambda i:pytest.fail('restored stale scroll')
    control.Focus=lambda i:pytest.fail('restored stale focus')
    control.Select=lambda i:pytest.fail('restored stale selection')
    control.selection=123
    it=steps(p,True)
    for _ in it:
        assert control.frozen==0
        assert len(control.rows)==len(p._sorted_messages)
        assert control.selection==123
    assert control.rows==[str(i) for i in range(13005)]
    assert control.frozen==0 and control.draft=='writing a reply'
