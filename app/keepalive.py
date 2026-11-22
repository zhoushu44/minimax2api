"""Credential keep-alive.

The device-flow access token lives one hour (``expires_in=3600``).  A pool that
does not renew it is a pool that works for an hour and then answers 401 to
everything, which is exactly what happened to the first accounts registered here:
they were fine at 13:25 and dead by 14:30.

Two renewal paths, tried in order:

1. **Refresh grant** — ``POST /oauth2/token`` with ``grant_type=refresh_token``.
   Needs no mailbox and no password, and the response carries a *new* refresh
   token, so one successful refresh extends the credential indefinitely.  This is
   the path that matters.
2. **Password sign-in** — the same create-password/login call the registration
   uses, then a fresh device grant.  This is the fallback for an account whose
   refresh token was never stored or has been refused; it needs the email and
   password columns to be populated.

Neither path needs a browser, and neither consumes an emailed code.

The keeper runs on a timer rather than reacting to failures: refreshing a token
that is about to expire costs one request, while discovering the expiry from a
401 costs a failed user request, a cooldown, and a window where the account looks
broken.  The request path still renews on demand as a safety net (see ``pool``),
because a token can expire between two ticks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from . import signup
from .records import REGION_GLOBAL

# Refresh this long before the token dies.  The access token lives one hour and
# the keeper sweeps every half hour, so the margin has to exceed the sweep
# interval: a token that is "not yet due" on one pass must not be expired by the
# next.  Forty minutes leaves the sweep a window in which every token it will
# need to renew is already due, while still renewing each one only about once an
# hour.
REFRESH_MARGIN_SEC = 2400.0

# How long to wait after a pass that renewed nothing but had accounts due (a
# failure or a transport blip), so the retry is minutes away rather than a full
# interval.
_RETRY_INTERVAL_SEC = 120.0

# A token minted with no expires_in is treated as this; the endpoint always
# sends one, so this is only a guard against a malformed response.
DEFAULT_TOKEN_TTL_SEC = 3600.0


@dataclass
class RenewResult:
    ok: bool = False
    method: str = ""       # "refresh" | "password"
    token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    error: str = ""
    # A transport failure is not a verdict on the credential: the proxy blinked,
    # the network dropped.  The caller retries these later instead of treating
    # the account as un-renewable, which is what a blank error message hides.
    transient: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "method": self.method,
            "expiresAt": self.expires_at,
            "refreshRotated": bool(self.refresh_token),
            "transient": self.transient,
            "error": self.error,
        }


def _is_transient(err: BaseException) -> bool:
    """Whether a failure is the network rather than the credential."""
    return isinstance(err, (httpx.TransportError, httpx.TimeoutException))


async def refresh_with_token(
    client: httpx.AsyncClient,
    account_origin: str,
    refresh_token: str,
) -> RenewResult:
    """Exchange a refresh token for a new access token.

    Form-encoded, and ``app_id`` must not ride along — it belongs to the
    browser's authorize call; on the token endpoint it is an ``invalid_request``.
    """
    if not refresh_token:
        return RenewResult(error="no refresh token")
    try:
        response = await client.post(
            f"{account_origin}/oauth2/token",
            data={
                "client_id": "mcode-public",
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
            timeout=20.0,
        )
        payload = response.json()
    except Exception as err:  # noqa: BLE001 - any transport failure is one outcome
        return RenewResult(
            error=f"{type(err).__name__}: {err}"[:200],
            transient=_is_transient(err),
        )

    token = str(payload.get("access_token") or "")
    if not token:
        # A refused refresh is a real answer ("invalid_grant"), not a blip, and
        # the caller falls through to the password path.
        #
        # The OAuth ``error`` code is kept alongside the human description: the
        # death verdict is decided on the *protocol* field (RFC 6749 §5.2
        # ``invalid_grant`` means the refresh token is dead/revoked), while the
        # description alone is prose that upstream may reword at any time.
        code = str(payload.get("error") or "").strip()
        desc = str(payload.get("error_description") or "").strip()
        if code and desc:
            message = f"{code}: {desc}"
        else:
            message = code or desc or str(payload)
        return RenewResult(error=message[:200])
    ttl = float(payload.get("expires_in") or DEFAULT_TOKEN_TTL_SEC)
    return RenewResult(
        ok=True,
        method="refresh",
        token=token,
        # The server rotates the refresh token; keeping the old one works until
        # it does not, so the new one is stored whenever it is offered.
        refresh_token=str(payload.get("refresh_token") or ""),
        expires_at=time.time() + ttl,
    )


async def renew_with_password(
    client: httpx.AsyncClient,
    account: Any,
    settings: Any,
) -> RenewResult:
    """Renew by signing in with the stored password, then a device grant.

    Runs entirely against the account service: no mailbox, no emailed code.  It
    is the fallback for an account whose refresh token is missing or dead, and it
    is also how a brand-new registration's first token is produced.
    """
    if not account.email or not account.password:
        return RenewResult(error="no email/password stored for password renewal")

    origin = signup.origin_for(account.region or REGION_GLOBAL)
    session = signup._AccountSession(uuid=account.uuid, device_id=account.device_id)
    try:
        _status, payload = await signup._signed(
            client, origin, session, "/oauth2/login",
            {
                "loginType": signup.LOGIN_TYPE_PASSWORD,
                "email": account.email,
                "authToken": signup.rsa_encrypt(account.password),
                "deviceID": session.device_id,
            },
        )
        if not isinstance(payload, dict) or payload.get("code") != 0 or not session.sid:
            info = payload.get("statusInfo") if isinstance(payload, dict) else {}
            return RenewResult(error=f"password login: {str((info or {}).get('message') or payload)[:180]}")

        token, refresh = await signup._mint_token_full(client, origin, session, settings)
    except Exception as err:  # noqa: BLE001
        return RenewResult(
            error=f"{type(err).__name__}: {err}"[:200],
            transient=_is_transient(err),
        )

    return RenewResult(
        ok=True,
        method="password",
        token=token,
        refresh_token=refresh,
        expires_at=time.time() + DEFAULT_TOKEN_TTL_SEC,
    )


# ── 死号判定 ────────────────────────────────────────────────────────────
#
# 只认**协议级/业务级的确定答复**，不认任何"可能只是暂时"的错误。
#
# ``invalid_grant`` 来自 RFC 6749 §5.2：refresh token 无效、过期或已被撤销。
# 这是上游明确说"这个 refresh token 不能再用"，不是网络、不是限流、不是余额。
#
# 官方错误码表（platform.minimax.io/docs/api-reference/errorcode）里没有任何
# 一个码表示"账号永久失效"——1004 的官方处置是 "log in again"，2049 是
# "check your API Key"，两者都可恢复，因此都不作为删除依据。
_DEFINITIVE_GRANT_ERRORS = (
    "invalid_grant",                      # RFC 6749 §5.2 协议字段
    "this refresh token can no longer be used",   # 官方 error_description 原文
    "start a new authorization",          # 同上后半句
)

# 出现这些字样一律不当死号：网络、限流、额度、余额、服务端故障。
_NEVER_DEAD_MARKERS = (
    "timeout", "timed out", "connect", "connection", "network", "dns", "ssl",
    "rate limit", "too many", "usage limit", "quota", "balance", "insufficient",
    "overload", "high load", "temporarily", "unavailable", "internal error",
    "server error", "502", "503", "504", "429", "401", "403",
)


def is_definitive_death(refresh_error: str, password_error: str,
                        transient: bool) -> tuple[bool, str]:
    """一条记录是否确定死亡。返回 (能否删, 理由)。

    必须同时满足：
      1. 不是 transient（网络问题直接排除）；
      2. refresh 的错误命中 ``invalid_grant``；
      3. 密码路径也失败，且**不是**因为"没存密码"（那只是不可续期，不是死）；
      4. 两条错误里都不含任何"临时性"标记。

    任何一条不满足就不删 —— 宁可留一个死号，也不误杀一个活号。
    """
    if transient:
        return False, "transient（网络问题，凭据未被评判）"

    blob = f"{refresh_error} {password_error}".lower()
    for marker in _NEVER_DEAD_MARKERS:
        if marker in blob:
            return False, f"含临时性标记 {marker!r}"

    granted = any(code in refresh_error.lower() for code in _DEFINITIVE_GRANT_ERRORS)
    if not granted:
        return False, "refresh 未返回 invalid_grant"

    if not password_error:
        # 密码路径没报错却也没成功：状态不明，不删。
        return False, "密码路径无明确结论"

    if "no email/password stored" in password_error.lower():
        # 只是没存密码，号本身可能还活着（refresh 坏了但密码能手动补）。
        return False, "未存密码（不可续期，非死亡）"

    # "New user has not set a password" 说明这个账号在上游**本来就没有密码**，
    # 因此密码路径不是一条独立的旁证 —— 它失败与否都不能证明账号死了。
    # 只有"确实设过密码、但登录被拒"才算第二重确认。
    if "has not set a password" in password_error.lower():
        return False, "账号本身未设密码，密码路径不构成旁证"

    return True, f"refresh=invalid_grant；password={password_error[:80]}"


class Keeper:
    """Renews accounts before their tokens expire, and reports what it did."""

    def __init__(
        self,
        db: Any,
        client: Any,
        settings_fn: Callable[[], Any],
        interval_sec: float | None = None,
    ) -> None:
        self._db = db
        self._client = client
        self._settings_fn = settings_fn
        # None means "read the interval from settings on every pass", so a change
        # made in the console takes effect without a restart.  An explicit number
        # is kept for tests, which have no settings object to consult.
        self._fixed_interval = None if interval_sec is None else max(60.0, float(interval_sec))
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_run: dict[str, Any] = {}

    def _interval(self) -> float:
        if self._fixed_interval is not None:
            return self._fixed_interval
        return max(60.0, float(self._settings_fn().keepalive.interval_sec))

    def _enabled(self) -> bool:
        return bool(self._settings_fn().keepalive.enabled)

    # ------------------------------------------------------------------ public

    async def renew_account(self, account_id: str) -> RenewResult:
        """Renew one account and persist the result."""
        account = await self._db.account_by_id(account_id)
        if account is None:
            return RenewResult(error="account not found")
        if account.kind != "oauth":
            return RenewResult(error="not an oauth account")

        settings = self._settings_fn()
        proxy = (settings.upstream.proxy or "").strip() or None
        origin = signup.origin_for(account.region or REGION_GLOBAL)

        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=False, trust_env=False, proxy=proxy
        ) as client:
            # Refresh first: it is one request and needs no password.  When it is
            # missing or refused, the password path re-logs in and mints a *fresh*
            # refresh token as well, so the account is renewable again rather than
            # merely kept alive once.  Neither path is skipped for want of the
            # other: a stored password is what makes an account keepable even if
            # its refresh token was never captured.
            result = await refresh_with_token(client, origin, account.refresh_token)
            if not result.ok:
                first = result.error
                result = await renew_with_password(client, account, settings.signup)
                if not result.ok:
                    result.error = f"refresh: {first} | password: {result.error}"

        if result.ok:
            def apply(item: Any) -> None:
                item.token = result.token
                if result.refresh_token:
                    item.refresh_token = result.refresh_token
                item.token_expires_at = result.expires_at
                # A renewal is proof the credential works: clear whatever state
                # the last failure left, so a recovered account is scheduled again.
                item.status = "active"
                item.last_error = ""
                item.fail_count = 0
                item.cooldown_until = 0.0

            await self._db.update_account(account_id, apply)
            print(
                f"[keepalive] renewed {account.name or account_id} via {result.method}"
                f" until {time.strftime('%H:%M:%S', time.localtime(result.expires_at))}",
                flush=True,
            )
        elif result.transient:
            # The credential was never judged — the network failed.  Leave the
            # account and its stored refresh token exactly as they were; the next
            # sweep will try again.  Writing a failure here is what makes a flaky
            # proxy look like a dead account.
            print(
                f"[keepalive] {account.name or account_id}: transport failure,"
                f" will retry ({result.error})",
                flush=True,
            )
        else:
            def mark(item: Any) -> None:
                item.last_error = f"renew failed: {result.error}"[:500]

            await self._db.update_account(account_id, mark)
            print(
                f"[keepalive] renew FAILED {account.name or account_id}: {result.error}",
                flush=True,
            )
        return result

    async def sweep(self) -> dict[str, Any]:
        """Renew every enabled oauth account whose token is at or near expiry.

        Only enabled accounts: an account parked by the operator is not spending
        tokens, so renewing it is a request spent on a credential nobody is using,
        and it delays the renewals that matter.  An account that cannot be renewed
        at all — no refresh token and no stored password — is skipped rather than
        counted as a failure: it is not broken, it is simply not renewable, and
        reporting it every pass would bury the accounts that need attention.
        """
        now = time.time()
        due, skipped = [], 0
        for account in await self._db.list_accounts():
            if account.kind != "oauth" or not account.enabled:
                continue
            if account.token_expires_at and account.token_expires_at - REFRESH_MARGIN_SEC > now:
                continue
            if not account.refresh_token and not (account.email and account.password):
                skipped += 1
                continue
            due.append(account)

        renewed, failed, transient = 0, 0, 0
        doomed: list[tuple[str, str, str]] = []  # (account_id, name, reason)
        for account in due:
            try:
                result = await self.renew_account(account.id)
            except Exception as err:  # noqa: BLE001 - one account cannot kill the sweep
                failed += 1
                print(f"[keepalive] {account.name or account.id}: {err}", flush=True)
                continue
            if result.ok:
                renewed += 1
            elif result.transient:
                transient += 1
            else:
                failed += 1
                # 只有"确定性死亡"才进候选，其余一律留着。用本次的 result
                # 判定，不读库里的 last_error（那是上一次的旧值）。
                verdict, reason = self._death_verdict(account, result)
                if verdict:
                    doomed.append((account.id, account.name or account.id, reason))
            # Space the renewals: a burst of token grants from one address is its
            # own pattern, and there is no urgency — each account has minutes.
            await asyncio.sleep(2.0)

        pruned = await self._prune(doomed)

        self._last_run = {
            "at": now,
            "due": len(due),
            "renewed": renewed,
            "failed": failed,
            "transient": transient,
            "skipped": skipped,
            "doomed": len(doomed),
            "pruned": pruned,
        }
        return self._last_run

    def _death_verdict(self, account: Any, result: Any = None) -> tuple[bool, str]:
        """续期失败后，判断这个号是否确定死亡。

        判据完全基于**本次**续期的失败详情（``result.error``），不回头读
        ``account.last_error``：那是上一次写的旧值，可能为空或已过期。

        不推测、不"看次数猜"。
        """
        try:
            settings = self._settings_fn()
            cfg = settings.keepalive
            if not getattr(cfg, "auto_prune", False):
                return False, "auto_prune 关闭"
            rounds = int(getattr(cfg, "auto_prune_rounds", 1) or 1)
        except Exception:  # noqa: BLE001
            return False, "读配置失败"

        raw = str(getattr(result, "error", "") or "") if result is not None else ""
        if not raw:
            return False, "无错误记录"

        # renew_account 失败时写的是 "refresh: <x> | password: <y>"
        refresh_err, password_err = "", ""
        if "| password:" in raw:
            left, right = raw.split("| password:", 1)
            password_err = right.strip()
            refresh_err = left.split("refresh:", 1)[-1].strip() if "refresh:" in left else left.strip()
        elif "refresh:" in raw:
            refresh_err = raw.split("refresh:", 1)[-1].strip()
        else:
            refresh_err = raw

        # 密码路径只有在"存了密码却仍失败"时才算一次否定证据。
        has_pw = bool(getattr(account, "email", "")) and bool(getattr(account, "password", ""))
        if has_pw and not password_err:
            return False, "密码路径未尝试或结论不明"

        ok, reason = is_definitive_death(refresh_err, password_err, transient=False)
        if not ok:
            return False, reason

        # 需要连续 N 轮确认时，用 fail_count 累计（成功续期会把它清零）。
        if rounds > 1:
            fails = int(getattr(account, "fail_count", 0) or 0)
            if fails + 1 < rounds:
                return False, f"确定性失败第 {fails + 1}/{rounds} 轮，未达阈值"

        return True, reason

    async def _prune(self, doomed: list[tuple[str, str, str]]) -> int:
        """删除确定死亡的号。带上限，且先把判据写进日志。"""
        if not doomed:
            return 0
        try:
            cap = int(self._settings_fn().keepalive.auto_prune_max_per_run or 5)
        except Exception:  # noqa: BLE001
            cap = 5
        batch = doomed[: max(1, cap)]
        ids = [item[0] for item in batch]
        for _aid, name, reason in batch:
            # 删了也要留痕：控制台和 docker logs 都能查到为什么删。
            print(f"[keepalive] PRUNE {name}: {reason}", flush=True)
        try:
            removed = await self._db.delete_accounts(ids)
        except Exception as err:  # noqa: BLE001
            print(f"[keepalive] prune failed: {err}", flush=True)
            return 0
        print(f"[keepalive] pruned {removed}/{len(ids)} dead accounts", flush=True)
        return int(removed or 0)

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "enabled": self._enabled(),
            "intervalSec": int(self._interval()),
            "lastRun": self._last_run,
        }

    # --------------------------------------------------------------- background

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._loop(), name="minimaxcode2api-keepalive")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _loop(self) -> None:
        # First sweep waits one interval: on a fresh boot the accounts were just
        # loaded and a burst of grants at startup is the pattern to avoid.
        while self._running:
            try:
                await asyncio.sleep(self._interval())
                if not self._running:
                    return
                if not self._enabled():
                    continue
                summary = await self.sweep()
                # Something was due and did not get renewed: come back soon rather
                # than in a full interval.  The margin is minutes, but a retry that
                # waits ten of them is a retry that can lose a credential.
                if (summary.get("failed") or summary.get("transient")) and self._running:
                    await asyncio.sleep(_RETRY_INTERVAL_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - the loop outlives one bad sweep
                print(f"[keepalive] sweep error: {err}", flush=True)
