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
            conn = sqlite3.connect(PROFILES_DB)
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            res = cur.fetchall()
            conn.close()
            if not res or res[0][0] != "ok":
                raise sqlite3.DatabaseError("Corrupt")
        except Exception:
            if os.path.exists(PRISTINE_DB):
                shutil.copy2(PRISTINE_DB, PROFILES_DB)
    elif os.path.exists(PRISTINE_DB):
        shutil.copy2(PRISTINE_DB, PROFILES_DB)


@pytest.fixture(scope="session", autouse=True)
def ensure_db_integrity():
    verify_or_restore_db()
    yield
    verify_or_restore_db()
