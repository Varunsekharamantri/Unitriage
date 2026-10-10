import threading
from collections import OrderedDict
from typing import Optional, Tuple


class PhotoCache:
    """Recent tenant photos, so a photo can be shown the moment it is sent.

    The same bytes are also saved in the ticket's checkpoint, but that only exists once the
    first graph step has finished, and the owner's phone asks for the photo straight away.
    Bounded, and a demo-grade stand-in for object storage.
    """

    def __init__(self, max_items: int = 200):
        self._items: "OrderedDict[str, Tuple[bytes, str]]" = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    def put(self, ticket_id: str, data: bytes, mime: str) -> None:
        with self._lock:
            self._items[ticket_id] = (data, mime)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def get(self, ticket_id: str) -> Optional[Tuple[bytes, str]]:
        with self._lock:
            return self._items.get(ticket_id)
