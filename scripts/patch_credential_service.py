path = "src/services/credential_service.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn'''

replacement = '''    def _get_connection(self) -> sqlite3.Connection:
        import shutil
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            # Quick integrity probe
            cur = conn.cursor()
            cur.execute("PRAGMA quick_check(1);")
            res = cur.fetchone()
            if res and res[0] != "ok":
                raise sqlite3.DatabaseError(f"Integrity check failed: {res[0]}")
            return conn
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as err:
            err_msg = str(err).lower()
            if "malformed" in err_msg or "corrupt" in err_msg or "integrity" in err_msg:
                logger.error("Corrupted database detected at %s. Auto-recovering from pristine...", self.db_path)
                pristine = self.db_path.parent / (self.db_path.name + ".pristine")
                wal_file = self.db_path.parent / (self.db_path.name + "-wal")
                shm_file = self.db_path.parent / (self.db_path.name + "-shm")
                for f in (wal_file, shm_file):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
                if pristine.exists():
                    shutil.copy2(pristine, self.db_path)
                else:
                    try:
                        self.db_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._init_db()
                conn2 = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
                conn2.row_factory = sqlite3.Row
                conn2.execute("PRAGMA journal_mode=WAL")
                return conn2
            raise'''

if target in content:
    content = content.replace(target, replacement, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully patched credential_service.py with self-healing SQLite recovery!")
else:
    print("Target not found in credential_service.py")
