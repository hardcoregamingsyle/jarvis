"""Triage, projects, and work that survives a reboot.

Three behaviours, each with a specific failure this suite exists to prevent:

* **Routing.** A greeting must not cost two minutes of dense inference, and a
  real instruction must never be swallowed by the cheap path. The second is
  the dangerous direction: a slow answer annoys, a confidently wrong one from
  a 1.7B model misleads. Every ambiguous case is asserted to escalate.
* **Projects.** Work is filed somewhere, so "pause" and "what's running" have
  an unambiguous subject.
* **Durability.** A task interrupted by a reboot comes back as *paused with
  its progress intact*, not as running (a lie about a dead thread) and not as
  lost.
"""

from __future__ import annotations

import pytest

from jarvis.agent import projects as P
from jarvis.agent.projects import ProjectStore
from jarvis.agent.router import (
    CONTROL,
    STATUS,
    TASK_EDIT,
    TASK_NEW,
    Router,
    looks_like_chitchat,
    looks_like_edit,
    match_control,
)
from jarvis.core.contracts import Task, TaskState


class FakeTasks:
    """A task pool with a fixed set of running goals."""

    def __init__(self, *goals: str) -> None:
        self._tasks = [
            Task(id=f"task-{i}", goal=g, state=TaskState.RUNNING)
            for i, g in enumerate(goals)
        ]

    def list(self, state=None):
        return list(self._tasks)


# --------------------------------------------------------------------------- #
#  Control verbs
# --------------------------------------------------------------------------- #
class TestControl:
    """"Stop" must work without a language model in the loop.

    If the agent is doing something you want halted, waiting on inference to
    understand the word "stop" is exactly the wrong time to need inference.
    """

    @pytest.mark.parametrize(
        "text, action",
        [
            ("pause", "pause"),
            ("pause it", "pause"),
            ("hold on", "pause"),
            ("resume", "resume"),
            ("carry on", "resume"),
            ("stop", "cancel"),
            ("cancel it", "cancel"),
            ("abort", "cancel"),
            ("never mind", "cancel"),
        ],
    )
    def test_control_verbs_are_recognised(self, text, action):
        assert match_control(text) == action

    @pytest.mark.parametrize(
        "text",
        [
            "the pause button is broken",
            "explain how to stop a running container",
            "write a resume for me",
        ],
    )
    def test_a_verb_used_as_a_noun_is_not_a_command(self, text):
        """"Write a resume" must not resume paused work."""
        assert match_control(text) is None

    def test_control_never_needs_the_big_model(self):
        decision = Router().route("pause")
        assert decision.route == CONTROL
        assert decision.action == "pause"
        assert decision.needs_big_model is False


# --------------------------------------------------------------------------- #
#  Chitchat
# --------------------------------------------------------------------------- #
class TestChitchat:
    @pytest.mark.parametrize(
        "text",
        ["hi", "hey jarvis", "good morning", "thanks", "cheers", "ok", "goodbye"],
    )
    def test_social_openers_stay_cheap(self, text):
        assert looks_like_chitchat(text) is True
        assert Router().route(text).needs_big_model is False

    @pytest.mark.parametrize(
        "text",
        [
            "good morning, now audit the disk usage",
            "hi, could you restart the server",
            "thanks! also please check the logs",
            "hello, I need you to refactor the auth module",
        ],
    )
    def test_a_greeting_with_a_request_attached_is_a_request(self, text):
        """The worst failure here: answering "Good morning, Sir" and silently
        dropping the instruction that followed it."""
        assert looks_like_chitchat(text) is False
        assert Router().route(text).needs_big_model is True

    def test_a_long_utterance_is_never_chitchat(self):
        text = "hello there " + "and then do the thing " * 5
        assert looks_like_chitchat(text) is False


# --------------------------------------------------------------------------- #
#  Status
# --------------------------------------------------------------------------- #
class TestStatus:
    def test_progress_questions_are_answered_locally_when_work_is_running(self):
        router = Router(task_manager=FakeTasks("audit the logs"))
        decision = router.route("what are you working on")
        assert decision.route == STATUS
        assert decision.needs_big_model is False

    def test_a_progress_question_with_nothing_running_escalates(self):
        """"How's it going" with an empty queue is small talk about the
        machine, not a request to recite an empty list."""
        decision = Router(task_manager=FakeTasks()).route("what are you working on")
        assert decision.needs_big_model is True

    @pytest.mark.parametrize("text", ["status report", "disk status", "memory report"])
    def test_questions_about_the_machine_are_not_task_status(self, text):
        """These need tools and the big model. Answering them with a task list
        would replace a real answer with a non sequitur."""
        router = Router(task_manager=FakeTasks("audit the logs"))
        assert router.route(text).needs_big_model is True


# --------------------------------------------------------------------------- #
#  Edits
# --------------------------------------------------------------------------- #
class TestEdits:
    @pytest.mark.parametrize(
        "text",
        [
            "actually also check the logs",
            "change it to use ripgrep instead",
            "one more thing, add tests",
        ],
    )
    def test_edit_phrasing_is_recognised(self, text):
        assert looks_like_edit(text) is True

    def test_an_edit_targets_the_running_task_it_resembles(self):
        """Spawning a rival subagent for an amendment gives you two agents
        editing the same files."""
        router = Router(
            task_manager=FakeTasks(
                "audit the authentication module",
                "write the release notes",
            )
        )
        decision = router.route("actually also check the authentication tests")
        assert decision.route == TASK_EDIT
        assert decision.target_task_id == "task-0"

    def test_an_edit_with_nothing_running_becomes_new_work(self):
        decision = Router(task_manager=FakeTasks()).route("actually also check the logs")
        assert decision.route == TASK_NEW

    def test_an_unclassifiable_turn_escalates_rather_than_guessing(self):
        decision = Router().route("refactor the payment pipeline for idempotency")
        assert decision.route == TASK_NEW
        assert decision.needs_big_model is True
        assert decision.confident is False


# --------------------------------------------------------------------------- #
#  Projects
# --------------------------------------------------------------------------- #
@pytest.fixture
def store(tmp_path):
    s = ProjectStore(tmp_path / "projects.db")
    yield s
    s.close()


class TestProjects:
    def test_there_is_always_an_active_project(self, store):
        assert store.active_project() == P.DEFAULT_PROJECT

    def test_switching_creates_the_target_if_needed(self, store):
        assert store.set_active_project("jarvis-rewrite") == "jarvis-rewrite"
        assert store.active_project() == "jarvis-rewrite"
        assert {p.name for p in store.list_projects()} >= {
            "jarvis-rewrite", P.DEFAULT_PROJECT
        }

    def test_exactly_one_project_is_active(self, store):
        store.set_active_project("a")
        store.set_active_project("b")
        assert sum(1 for p in store.list_projects() if p.active) == 1

    def test_tasks_are_filed_under_the_active_project(self, store):
        store.set_active_project("alpha")
        store.record_task("t1", "do the thing")
        store.set_active_project("beta")
        store.record_task("t2", "do the other thing")

        assert [t.id for t in store.list_tasks(project="alpha")] == ["t1"]
        assert [t.id for t in store.list_tasks(project="beta")] == ["t2"]


# --------------------------------------------------------------------------- #
#  Surviving a reboot
# --------------------------------------------------------------------------- #
class TestDurability:
    def test_a_task_interrupted_by_a_reboot_comes_back_paused(self, tmp_path):
        """RUNNING after a restart is a lie: that thread is gone. PAUSED is
        the truth, and it is recoverable."""
        path = tmp_path / "projects.db"
        first = ProjectStore(path)
        first.record_task("t1", "audit the auth module", state=P.RUNNING)
        first.add_progress("t1", "read src/auth/*.py, found 3 raw SQL builds")
        first.close()

        second = ProjectStore(path)
        try:
            assert second.mark_interrupted() == 1
            task = second.get_task("t1")
            assert task.state == P.PAUSED
            assert task.resumable is True
            assert task.progress == ["read src/auth/*.py, found 3 raw SQL builds"]
        finally:
            second.close()

    def test_the_resume_briefing_tells_the_agent_not_to_repeat_itself(self, store):
        store.record_task("t1", "audit the auth module", state=P.RUNNING)
        store.add_progress("t1", "read src/auth/login.py")
        store.add_progress("t1", "checking src/db/query.py next")

        briefing = store.get_task("t1").resume_briefing()

        assert "audit the auth module" in briefing
        assert "read src/auth/login.py" in briefing
        assert "checking src/db/query.py next" in briefing
        assert "do not repeat" in briefing.lower()

    def test_a_finished_task_is_not_resumable(self, store):
        store.record_task("t1", "done thing", state=P.RUNNING)
        store.update_state("t1", P.DONE, result="all good")
        task = store.get_task("t1")
        assert task.resumable is False
        assert task.result == "all good"

    def test_progress_notes_are_bounded(self, store):
        """An unbounded list eventually exceeds the context window it is meant
        to prime."""
        store.record_task("t1", "long job", state=P.RUNNING)
        for i in range(300):
            store.add_progress("t1", f"step {i}")
        notes = store.get_task("t1").progress
        assert len(notes) <= 200
        assert notes[-1] == "step 299", "the newest notes are the ones to keep"

    def test_state_survives_reopening(self, tmp_path):
        path = tmp_path / "projects.db"
        a = ProjectStore(path)
        a.set_active_project("gamma")
        a.record_task("t9", "persist me", state=P.PAUSED)
        a.close()

        b = ProjectStore(path)
        try:
            assert b.active_project() == "gamma"
            assert b.get_task("t9").goal == "persist me"
        finally:
            b.close()

    def test_the_summary_reads_as_a_sentence(self, store):
        store.set_active_project("alpha")
        store.record_task("t1", "audit the logs", state=P.RUNNING)
        store.record_task("t2", "write notes", state=P.PAUSED)
        summary = store.summary()
        assert "alpha" in summary
        assert "audit the logs" in summary
        assert "on hold" in summary

    def test_an_empty_project_says_so_plainly(self, store):
        assert "Nothing is running" in store.summary()
