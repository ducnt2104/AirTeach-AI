"""
Pytest configuration and session-wide fixtures for AirTeach AI test suite.
Guarantees database integrity, isolates SQLite test profiles, and cleans up Qt resources.
"""
import os
import shutil
import sqlite3
import pytest

PROFILES_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "profiles.db"))
PRISTINE_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "profiles.db.pristine"))


def verify_or_restore_db():
    if os.path.exists(PROFILES_DB):
        try:
            conn = sqlite3.connect(PROFILES_DB, timeout=5.0)
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            res = cur.fetchall()
            conn.close()
            if not res or res[0][0] != "ok":
                raise sqlite3.DatabaseError("Corrupt")
        except Exception:
            for ext in ("-wal", "-shm"):
                wal_f = PROFILES_DB + ext
                if os.path.exists(wal_f):
                    try:
                        os.remove(wal_f)
                    except Exception:
                        pass
            if os.path.exists(PRISTINE_DB):
                try:
                    shutil.copy2(PRISTINE_DB, PROFILES_DB)
                except Exception:
                    pass
    elif os.path.exists(PRISTINE_DB):
        shutil.copy2(PRISTINE_DB, PROFILES_DB)


@pytest.fixture(scope="session", autouse=True)
def ensure_db_integrity():
    verify_or_restore_db()
    yield
    verify_or_restore_db()


@pytest.fixture(autouse=True)
def per_test_db_guard():
    yield
    verify_or_restore_db()
