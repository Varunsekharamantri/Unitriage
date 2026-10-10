"""Messages and photos persist through an optional store (Postgres in production)."""
from app.adapters.mock_notifier import MockNotifier
from app.core.photos import PhotoCache


class FakeDb:
    """Stands in for app.core.db.Database: same four methods, kept in lists."""

    def __init__(self):
        self.messages, self.photos = [], {}

    def load_messages(self):
        return list(self.messages)

    def add_message(self, m):
        self.messages.append(m)

    def put_photo(self, ticket_id, data, mime):
        self.photos[ticket_id] = (data, mime)

    def get_photo(self, ticket_id):
        return self.photos.get(ticket_id)


def test_messages_survive_a_restart():
    db = FakeDb()
    a = MockNotifier(echo=False, store=db)
    a.send("manager", "Hello", "body", ticket_id="T-1", actions=[{"kind": "quote", "label": "Q"}], sender="tenant",
           conversation="tenant:tenant-12B")
    b = MockNotifier(echo=False, store=db)  # a new process reading the same database
    assert [(m.id, m.subject, m.ticket_id, m.actions) for m in b.sent] == [(1, "Hello", "T-1", [{"kind": "quote", "label": "Q"}])]
    b.send("tenant", "Next", "body")
    assert [m.id for m in b.sent] == [1, 2]  # ids continue instead of restarting at 1


def test_a_storage_failure_does_not_stop_a_notification():
    class Broken(FakeDb):
        def add_message(self, m):
            raise RuntimeError("database is down")

    n = MockNotifier(echo=False, store=Broken())
    n.send("manager", "x", "y")
    assert len(n.sent) == 1


def test_photos_fall_back_to_the_database_after_a_restart():
    db = FakeDb()
    PhotoCache(db=db).put("T-1", b"bytes", "image/png")
    assert PhotoCache(db=db).get("T-1") == (b"bytes", "image/png")
    assert PhotoCache(db=db).get("T-2") is None
    assert PhotoCache().get("T-1") is None  # without a database it is memory only, as before


def test_a_console_that_cannot_show_emoji_does_not_break_a_send(monkeypatch):
    import builtins

    real_print = builtins.print

    def narrow_print(text="", *a, **k):
        text.encode("cp1252")  # what a Windows console does: raises on an emoji
        real_print(text, *a, **k)

    monkeypatch.setattr(builtins, "print", narrow_print)
    n = MockNotifier(echo=True)
    n.send("tenant", "Done", "\u2705 It's fixed, thanks!")
    assert len(n.sent) == 1 and "\u2705" in n.sent[0].body  # the real message keeps its emoji
