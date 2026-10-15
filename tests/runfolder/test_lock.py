"""Two writers never share a run folder: the second is refused with run_locked."""

import errno
import inspect
import subprocess
import sys
import textwrap
import threading
import typing

import numpy as np
import pytest
from skimage import io

from gelpy.cli import _render
from gelpy.core.errors import RunLockedError, to_error_response
from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.lock import run_lock, takes_run_lock
from gelpy.schemas.requests import InitRunRequest, SetLanesRequest, StatusRequest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks")

HOLDER = textwrap.dedent(
    """
    import sys, time
    from gelpy.runfolder.lock import run_lock
    with run_lock(sys.argv[1]):
        print("held", flush=True)
        time.sleep(60)
    """
)


@pytest.fixture
def initialised(gel_image_path, run_dir):
    commands.init_run(
        InitRunRequest(run_dir=run_dir, image_path=gel_image_path[0], n_lanes=4, width=30)
    )
    return run_dir


def _stop(holder):
    """End a holder process and close its pipe, so neither a process nor a file outlives the test."""
    holder.kill()
    holder.wait()
    holder.stdout.close()


def _hold(run_dir):
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, run_dir], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "held"
    except BaseException:
        _stop(holder)
        raise
    return holder


@pytest.fixture
def held_elsewhere(initialised):
    holder = _hold(initialised)
    try:
        yield initialised
    finally:
        _stop(holder)


def test_a_writer_is_refused_while_another_process_holds_the_run(held_elsewhere):
    with pytest.raises(RunLockedError):
        commands.set_lanes(SetLanesRequest(run_dir=held_elsewhere, width=20))


def test_reads_do_not_wait_for_the_lock(held_elsewhere):
    with pytest.raises(RunLockedError):  # the lock is held, so a read that succeeds did not wait for it
        commands.set_lanes(SetLanesRequest(run_dir=held_elsewhere, width=20))
    assert commands.status(StatusRequest(run_dir=held_elsewhere)).n_lanes == 4


def test_the_lock_is_released_when_its_holder_dies(initialised):
    holder = _hold(initialised)
    _stop(holder)
    commands.set_lanes(SetLanesRequest(run_dir=initialised, width=20))  # no RunLockedError


def test_the_lock_is_reentrant_within_a_thread(initialised):
    with run_lock(initialised), run_lock(initialised):
        commands.set_lanes(SetLanesRequest(run_dir=initialised, width=20))


def test_another_thread_is_refused(initialised):
    refused = []

    def other():
        try:
            commands.set_lanes(SetLanesRequest(run_dir=initialised, width=20))
        except RunLockedError as exc:
            refused.append(exc)

    with run_lock(initialised):
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert len(refused) == 1


def test_leaving_an_inner_lock_keeps_the_outer_one(initialised):
    """A command that runs several locked steps holds the lock across all of them: the end of an
    inner block must not release the outer lock."""
    refused = []

    def other():
        try:
            commands.set_lanes(SetLanesRequest(run_dir=initialised, width=20))
        except RunLockedError as exc:
            refused.append(exc)

    with run_lock(initialised):
        with run_lock(initialised):
            pass  # the inner block ends here, the outer one is still held
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert len(refused) == 1
    commands.set_lanes(SetLanesRequest(run_dir=initialised, width=20))  # free after the outer block


def test_a_missing_run_folder_is_not_created_by_the_lock(tmp_path):
    missing = tmp_path / "typo"
    with pytest.raises(FileNotFoundError):
        commands.set_lanes(SetLanesRequest(run_dir=str(missing), width=20))
    assert not missing.exists()


def test_a_directory_that_is_not_a_run_is_left_untouched(tmp_path):
    """Only init needs the lock without a recipe.json. Any other writer fails with not_found, and
    must not leave a .gelpy.lock behind in an ordinary directory."""
    plain = tmp_path / "notes"
    plain.mkdir()
    with pytest.raises(FileNotFoundError):
        commands.set_lanes(SetLanesRequest(run_dir=str(plain), width=20))
    assert list(plain.iterdir()) == []


def test_a_filesystem_without_locks_is_not_reported_as_held(initialised, monkeypatch):
    import errno

    from gelpy.runfolder import lock

    def unsupported(fd):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(lock, "_acquire", unsupported)
    with pytest.raises(OSError) as info:
        commands.set_lanes(SetLanesRequest(run_dir=initialised, width=20))
    assert not isinstance(info.value, RunLockedError)


def test_a_lock_failure_names_the_lock_file_and_the_likely_cause(initialised, monkeypatch):
    import fcntl

    def no_locks(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks)
    with pytest.raises(OSError) as info:
        commands.set_lanes(SetLanesRequest(run_dir=initialised, width=20))
    assert not isinstance(info.value, RunLockedError)
    assert info.value.errno == errno.ENOLCK
    response = to_error_response(info.value)
    assert response.code == "os_error"
    assert str(RunPaths(initialised).lock) in response.error  # it names the file ...
    assert "file locks" in response.error  # ... and the likely cause


# --- every writer is pinned under the lock ------------------------------------------------------
#
# The two lists below come from the modules themselves, so a writer added later is checked without
# anyone editing this file: it fails here until it takes the lock.

#: Functions of runfolder.commands that are not wrapped with @locked, and why.
READS = {"get_results", "status"}  # they only read, so a long fit never blocks them
SELF_LOCKING = {"init_run"}  # it takes the lock itself, with create=True: it makes the folder a run
NOT_A_COMMAND = {"image_warnings"}  # a helper of the commands that touches no run folder


def _public_functions(module):
    """The public functions defined in ``module`` itself, not the ones it imports."""
    return {
        name: func
        for name, func in vars(module).items()
        if inspect.isfunction(func) and not name.startswith("_") and func.__module__ == module.__name__
    }


def test_every_writer_of_commands_takes_the_lock():
    public = _public_functions(commands)
    exempt = READS | SELF_LOCKING | NOT_A_COMMAND
    assert exempt <= set(public), f"not functions of commands: {sorted(exempt - set(public))}"
    unlocked = sorted(
        name for name, func in public.items() if name not in exempt and not takes_run_lock(func)
    )
    assert unlocked == [], f"writers without the run lock (decorate with @locked): {unlocked}"
    assert not [name for name in READS if takes_run_lock(public[name])], "a read must not lock"


def test_every_renderer_takes_the_lock():
    public = _public_functions(_render)
    assert public
    unlocked = sorted(name for name, func in public.items() if not takes_run_lock(func))
    assert unlocked == [], f"renderers without the run lock (decorate with @locked_run_dir): {unlocked}"


@pytest.fixture(scope="module")
def held_run(tmp_path_factory):
    """A run folder and its image; another process holds the run's lock for the whole module."""
    work = tmp_path_factory.mktemp("held")
    image = str(work / "gel.tif")
    io.imsave(image, np.random.default_rng(0).random((40, 60)).astype(np.float32))
    run = str(work / "run")
    commands.init_run(InitRunRequest(run_dir=run, image_path=image, n_lanes=2, width=10))
    holder = _hold(run)
    try:
        yield run, image
    finally:
        _stop(holder)


#: Request fields besides run_dir (and image_path) that have no default.
EXTRA_FIELDS = {"init_run": {"n_lanes": 2}, "refit_lane": {"lane": 0}}

CALLS = [
    pytest.param(commands, name, id=f"commands.{name}")
    for name in sorted(set(_public_functions(commands)) - NOT_A_COMMAND)
] + [
    pytest.param(_render, name, id=f"_render.{name}") for name in sorted(_public_functions(_render))
]


def _call(owner, name, run, image):
    func = getattr(owner, name)
    if owner is _render:
        return func(run)  # (run_dir, ...): the lock is taken before the other arguments matter
    model = typing.get_type_hints(func)["request"]
    fields = dict(EXTRA_FIELDS.get(name, {}))
    if "image_path" in model.model_fields:
        fields["image_path"] = image
    return func(model(run_dir=run, **fields))


@pytest.mark.parametrize(("owner", "name"), CALLS)
def test_a_held_lock_refuses_every_writer_and_no_read(held_run, owner, name):
    """Another process holds the run: every writer raises run_locked (init_run included), and so
    does every renderer, while status and get_results answer."""
    run, image = held_run
    if owner is commands and name in READS:
        assert _call(owner, name, run, image) is not None  # a read does not wait, and is not refused
    else:
        with pytest.raises(RunLockedError):
            _call(owner, name, run, image)
