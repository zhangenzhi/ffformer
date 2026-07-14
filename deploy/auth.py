"""Authentication + account management for the ForestFormer3D dashboard.

Design goals:
  - Accounts persist on the PVC (survive pod restarts) — stored as JSON on
    /workspace/data, the same persistent Lustre volume as the model weights.
  - Passwords are never stored in plaintext: pbkdf2-hmac-sha256 with a
    per-user random salt (stdlib only).
  - Sessions are stateless signed tokens (itsdangerous TimestampSigner) with
    an expiry; the signing secret is generated once and persisted on the PVC.
  - First account created becomes admin; admins can list/delete users.
"""
import json
import os
import hashlib
import secrets
import threading
import time

from itsdangerous import TimestampSigner, BadSignature, SignatureExpired

# Store next to the model weights on the persistent volume.
DATA_DIR = os.environ.get('AUTH_DATA_DIR',
                          os.path.dirname(os.environ.get('CHECKPOINT_PATH',
                                          '/workspace/data/x')) or '/workspace/data')
USERS_FILE = os.path.join(DATA_DIR, 'ffformer_users.json')
SECRET_FILE = os.path.join(DATA_DIR, 'ffformer_secret.key')
SESSION_MAX_AGE = int(os.environ.get('AUTH_SESSION_DAYS', '7')) * 86400
PBKDF2_ROUNDS = 200_000

_lock = threading.Lock()


def _get_secret():
    """Load or create the persistent session-signing secret."""
    try:
        with open(SECRET_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        secret = secrets.token_hex(32)
        # 0600 so other tenants on the shared FS can't read it
        fd = os.open(SECRET_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(secret)
        return secret


_signer = TimestampSigner(_get_secret())


# --- User store ---

def _load():
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(users):
    tmp = USERS_FILE + '.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)


def _hash_pw(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt),
                               PBKDF2_ROUNDS).hex()


def user_count():
    return len(_load())


def list_users():
    """Public view of all accounts (no secrets)."""
    users = _load()
    return [{'username': u, 'role': d.get('role', 'user'),
             'created': d.get('created')} for u, d in sorted(users.items())]


def create_user(username, password, role=None):
    """Create an account. First-ever user becomes admin. Returns (ok, msg)."""
    username = (username or '').strip()
    if not username or not password:
        return False, "用户名和密码不能为空"
    if len(username) > 32 or not username.replace('_', '').replace('-', '').isalnum():
        return False, "用户名只能包含字母、数字、下划线或连字符(≤32)"
    if len(password) < 6:
        return False, "密码至少 6 位"
    with _lock:
        users = _load()
        if username in users:
            return False, "用户名已存在"
        if role is None:
            role = 'admin' if not users else 'user'
        salt = secrets.token_hex(16)
        users[username] = {
            'salt': salt,
            'hash': _hash_pw(password, salt),
            'role': role,
            'created': int(time.time()),
        }
        _save(users)
    return True, role


def verify_password(username, password):
    users = _load()
    u = users.get(username)
    if not u:
        return False
    candidate = _hash_pw(password, u['salt'])
    return secrets.compare_digest(candidate, u['hash'])


def change_password(username, old_password, new_password):
    if not verify_password(username, old_password):
        return False, "当前密码不正确"
    if len(new_password or '') < 6:
        return False, "新密码至少 6 位"
    with _lock:
        users = _load()
        u = users.get(username)
        if not u:
            return False, "用户不存在"
        salt = secrets.token_hex(16)
        u['salt'] = salt
        u['hash'] = _hash_pw(new_password, salt)
        users[username] = u
        _save(users)
    return True, "密码已更新"


def delete_user(username):
    with _lock:
        users = _load()
        if username not in users:
            return False, "用户不存在"
        if users[username].get('role') == 'admin' and \
                sum(1 for d in users.values() if d.get('role') == 'admin') <= 1:
            return False, "不能删除唯一的管理员"
        del users[username]
        _save(users)
    return True, "已删除"


def get_role(username):
    return _load().get(username, {}).get('role')


# --- Session tokens ---

def issue_token(username):
    return _signer.sign(username.encode()).decode()


def verify_token(token):
    if not token:
        return None
    try:
        raw = _signer.unsign(token, max_age=SESSION_MAX_AGE)
        username = raw.decode()
    except (BadSignature, SignatureExpired):
        return None
    # Ensure the account still exists (deleted users get invalidated)
    if username not in _load():
        return None
    return username
