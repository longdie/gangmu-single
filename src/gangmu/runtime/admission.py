"""Project-wide physical request slots, independent of session token ledgers.

Each live request holds one advisory OS lock until its charge is settled.
Different sessions and processes therefore share three provider slots. Process
death releases the lock, but never edits the original token liability: the
session ledger continues to retain the unfinished request and its reservation.
This cannot prove that a remote server stopped processing a disconnected call.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import threading
import weakref

from gangmu.single_document.core_client import ConcurrentLimit
from gangmu.single_document.implicit_client import _local

_LIVE = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


class PhysicalCallSlot:
    """Own one open lock descriptor; only closing it releases admission."""

    def __init__(self, descriptor):
        self._descriptor = descriptor
        with _LIVE_LOCK:
            _LIVE.add(self)

    def release(self):
        with _LIVE_LOCK:
            descriptor, self._descriptor = self._descriptor, None
            _LIVE.discard(self)
        if descriptor is not None:
            os.close(descriptor)

    def __del__(self):
        self.release()


def _after_fork():
    # A forked child must not keep a parent's paid request lock alive. Closing
    # the child's inherited descriptor leaves the parent's descriptor locked.
    global _LIVE_LOCK
    _LIVE_LOCK = threading.Lock()
    for slot in list(_LIVE):
        slot.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def acquire_physical_slot(root, provider):
    """Acquire a nonblocking slot shared by this project's provider sessions.

    Files are stable lock inodes, never replaced or unlinked by this module.
    Per-session concurrency and token checks still happen in their own ledger.
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("A provider identity is required for shared admission")
    root = Path(root).resolve(strict=True)
    provider_key = hashlib.sha256(provider.encode()).hexdigest()
    directory = _local(root, "artifacts/invocation-budgets/provider-slots/" + provider_key)
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(3):
        path = _local(root, directory / ("slot-" + str(number) + ".lock"))
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        else:
            return PhysicalCallSlot(descriptor)
    raise ConcurrentLimit("Project provider concurrency reached three physical requests")
