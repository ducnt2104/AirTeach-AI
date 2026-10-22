# AirTeach AI — Security, Privacy & Local-First Governance

**Security Class:** High-Assurance Educational Software  
**Data Residence:** 100% On-Device / Local-First (Zero Cloud Biometrics)

---

## 1. Local-First Privacy Principles

1. **Zero Raw Biometric Cloud Transmission:** Under no circumstance are webcam frames, teacher faces, or classroom video streams transmitted to any external server or cloud service.
2. **Ephemeral Image Processing:** Face enrollment frames are processed entirely in memory. Once 128-dimensional mathematical feature vectors are extracted by OpenCV SFace, the raw frames are immediately wiped from RAM and never stored to disk.
3. **Encrypted Vector Embeddings:** The 128-dim biometric vectors stored in SQLite (`data/profiles.db`) are encrypted with AES-128 via `cryptography.fernet.Fernet`. Even if the database file is copied, facial signatures cannot be inspected in plaintext.
4. **PBKDF2 Password/PIN Hashing:** Teacher PINs are salted with a cryptographically secure 32-byte salt (`secrets.token_bytes(32)`) and hashed using PBKDF2-HMAC-SHA256 with 100,000 iterations.
5. **Multi-Teacher Workspace Isolation:** Each teacher's lecture notes, `.airteach` bundles, and temporary caches are stored in an isolated filesystem partition:
   ```text
   data/workspaces/<teacher_uuid>/
   ```
   A teacher logging in cannot access, overwrite, or delete another teacher's lesson files.

---

## 2. Threat Modeling & Countermeasures

| Threat Vector | Severity | AirTeach AI Defense Mechanism |
|---|---|---|
| **Brute-Force PIN Attack** | High | Rate-limiting lockout: 5 consecutive incorrect attempts lock the teacher account for 30 seconds. Counter persists across window state. |
| **Presentation Attack / Photo Spoofing** | Medium | Liveness checker tracks eye aspect ratio (EAR) and facial bounding-box micro-movements across 15 consecutive frames before admitting face match. |
| **Offline Privacy Breach** | Critical | No cloud dependencies for core teaching; works seamlessly in air-gapped classrooms. |
| **LaTeX Injection / Shell Code** | High | `LaTeXRenderer` utilizes Matplotlib's internal MathText parser. It does NOT invoke system command-line TeX compilers, preventing arbitrary command execution. |
| **DirectShow Camera Hijack** | Medium | Camera handles are acquired with exclusive lock flags and released with explicit resource cleanup routines on application exit. |
