"""Billing dialogue detection for video turns.

Two upstream behaviours make a headless video turn stall, and both are
readable in the agent's own prose:

1. **The confirmation request.**  H3 / H3-Max are paid models, and the
   plugin's skill makes the agent *ask* before it orders anything.  A
   headless turn has nobody to answer, so the turn ends on a question and
   nothing is ever submitted.  The question is recognisable: the agent
   states a billing notice and waits for a reply.

2. **The refusal.**  The submit is attempted and the server answers `402`
   (or the older `50110:insufficient balance`).  Nothing is created, but
   the turn still bills the agent's thinking, so the operator pays for a
   failure that looks like a hang.

Both are detected from the *turn text*, which is a heuristic, so the
markers are deliberately narrow and high-signal.  The success case must
never fire: a finished video's reply talks about billing too ("计费：按你
确认的走"), so a detector that merely looks for the word "计费" would
destroy every good job.

The rules below are written against real captured turns, and the test
corpus in `tests/test_videobilling.py` replays them:

  * `fx_h3max.json`   — asks for a confirmation, then submits, then gets
                        `402`.  Must be detected as a confirmation request.
  * `fx_h3cheap.json` — submits, gets `402`.  Must be detected as a refusal.
  * `fx_insuff.json`  — `50110:insufficient balance (1008)`.  A refusal.
  * `fx_hailuo.json`  — a finished video.  Must be detected as neither.

The distinction that matters is *order and structure*, not vocabulary: a
refusal says the submit **failed**, a confirmation request says the submit
**has not happened yet**.  The success reply mentions billing as a fact
about a completed order, which is neither.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The reply that answers a confirmation request.  It says what the operator
# has already authorised and asks for no further decision, so the agent can
# go straight to the submit.
#
# Wording matters and this exact sentence is the one that was verified against
# the live upstream: it was replayed into the stalled H3-Max session and the
# agent went on to submit (and to report the 402) instead of asking again.
# Changing it means re-proving it, so it is kept verbatim.
CONFIRMATION_REPLY = "确认，请直接提交生成，不要再询问计费确认。"

# How many confirmation requests are answered before the turn is given up on.
# One is the normal case; a second means the agent asked again after a
# clear answer, which is not going to improve with another reply.
MAX_CONFIRMATIONS = 2


@dataclass
class Verdict:
    """What the turn text says about the video job's outcome."""

    #: The agent asked for a billing confirmation and has not submitted.
    needs_confirmation: bool = False
    #: The submit was refused for lack of credit.  `reason` explains it.
    refused: bool = False
    #: The reason to show the operator, in their language.
    reason: str = ""

    @property
    def actionable(self) -> bool:
        """True when the text says something the bridge should act on."""
        return self.needs_confirmation or self.refused


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------

# The upstream's *machine* strings.  These are copied out of the upstream's
# own answer, so they cannot appear in a successful generation's prose and
# they need no corroboration: seeing one *is* the refusal.
_REFUSAL_STRONG: tuple[tuple[str, str], ...] = (
    # The older error the plugin surfaces verbatim.
    ("insufficient balance", "上游账户余额不足（insufficient balance）"),
    ("50110", "上游账户余额不足（50110）"),
    # The HTTP status the commerce API answers a paid submit with.  The body
    # reaches the agent as JSON, so it arrives quoted, unquoted, and with
    # either spacing.
    ("code: 402", "上游计费接口返回 402，余额不足"),
    ('code":402', "上游计费接口返回 402，余额不足"),
    ('"code": 402', "上游计费接口返回 402，余额不足"),
    ("http 402", "上游计费接口返回 402，余额不足"),
    # The upstream's own sentence about the M Plan weekly limit.
    ("cannot cover this minimax-h3", "M Plan 周限额与账户积分不足以支付本次 H3 生成"),
)

# The *prose* refusals: Chinese paraphrases the agent wrote when it reported
# the rejection.  Weaker than the machine strings — "余额" on its own is
# ordinary billing talk — so one of these only counts when the text also
# shows a submission outcome.  That is what keeps the detector off a turn
# that merely explains the billing rules.
_REFUSAL_PROSE: tuple[tuple[str, str], ...] = (
    ("余额不够", "账户余额不足"),
    ("余额不足", "账户余额不足"),
    ("覆盖不了", "M Plan 周限额与账户积分不足以支付本次生成"),
    ("没有创建生成任务", "服务端未创建生成任务，未产生视频"),
    ("没有生成任何视频", "服务端未生成任何视频"),
    ("提交被服务端计费闸门拦下", "提交被上游计费闸门拦下，未产生视频"),
    ("扣费在服务端就被拒了", "扣费在服务端被拒，未产生视频"),
)

# The shapes that carry a submit outcome.  A prose refusal has to sit next to
# one of these to count; the successful Hailuo reply has none of them.
_FAILURE_MARKERS: tuple[str, ...] = (
    "提交",
    "返回",
    "被拒",
    "拒绝",
    "闸门",
    "扣费",
    "未创建",
    "没创建",
    "failed",
    "rejected",
    "denied",
    "402",
)


def detect_refusal(text: str) -> Verdict:
    """Does the turn text report that the upstream refused to bill this job?

    A machine string decides it on its own.  A prose paraphrase needs the
    text to also say something about the submission, because the successful
    Hailuo reply discusses the account's billing without ever reporting a
    failed order — and that reply must never be read as a refusal.
    """
    if not text:
        return Verdict()
    low = text.lower()
    for marker, reason in _REFUSAL_STRONG:
        if marker in low:
            return Verdict(refused=True, reason=reason)
    for marker, reason in _REFUSAL_PROSE:
        if marker in low and any(hit in low for hit in _FAILURE_MARKERS):
            return Verdict(refused=True, reason=reason)
    return Verdict()


# ---------------------------------------------------------------------------
# Confirmation request
# ---------------------------------------------------------------------------

# Phrases that mean "the agent has stopped and is waiting for the user to say
# yes".  One hit is enough; there is no second list to also satisfy.
#
# The entries come from *two independent* live sessions that phrased the same
# upstream gate differently — one asked "等你回一句确认才能真正下单", the
# other said it needed "用户对本次提交的明确确认才能提交" and called a
# pre-authorisation explicitly insufficient.  Covering both shapes is the
# point: matching one lucky transcript would leave the other stalling.
_BLOCKED_ON_CONFIRMATION: tuple[str, ...] = (
    # shape 1 — "waiting for your one-line confirmation"
    "等你回一句确认",
    "等你确认",
    "等你回复确认",
    "等你回复",
    "等你回",
    "等待你确认",
    "你回一句",
    "回一句确认",
    "待你确认",
    "收到后我",
    "你点头",
    "你确认后",
    "得先给你一条明确的计费告知",
    # shape 2 — "needs the user's explicit confirmation for this submission"
    "明确确认",
    "对本次提交的确认",
    "用户的确认",
    "需要你确认",
    "需要你回一句",
    "回复确认后",
    "确认后才能",
    "才算作确认",
    "才算确认",
    "不能真正下单",
    "不能直接开跑",
    "不能直接提交",
    "无法直接提交",
    "不能拿你这条预授权",
)

# A reply that shows the submit already happened.  If one of these is
# present the agent is past the question, whatever else the text says.
_ALREADY_SUBMITTED_MARKERS: tuple[str, ...] = (
    "已提交",
    "已经提交",
    "提交成功",
    "task_id",
    "task accepted",
    "任务已接受",
    "已接受",
    "生成任务已",
    "已创建生成",
)


def detect_confirmation_request(text: str) -> Verdict:
    """Does the turn text stop at a billing confirmation the agent wants?

    One narrow positive plus one veto:

    * the text must contain a phrase that means "waiting for the user to
      confirm" — not merely the word "确认", which the agent writes in every
      H3 reply including the ones that go on to submit; and
    * the text must not contain a marker showing the submit already
      happened.
    """
    if not text:
        return Verdict()
    if any(marker in text for marker in _ALREADY_SUBMITTED_MARKERS):
        return Verdict()
    if not any(marker in text for marker in _BLOCKED_ON_CONFIRMATION):
        return Verdict()
    return Verdict(needs_confirmation=True, reason="上游要求先确认计费，尚未提交生成")


def classify(text: str) -> Verdict:
    """Read a turn's text and say what it means for the job.

    A refusal wins over a confirmation request: a turn that reports a
    rejected submit has already had its chance to be confirmed, so
    answering it again would only waste a turn.
    """
    refusal = detect_refusal(text)
    if refusal.refused:
        return refusal
    return detect_confirmation_request(text)


# ---------------------------------------------------------------------------
# Model billing class
# ---------------------------------------------------------------------------

# Which models are billed from the M Plan weekly limit / paid credits and
# which from the legacy Token Plan allowance.  The upstream refuses the
# first group on an account that holds only the second, with a `402`.
#
# This is upstream vocabulary and is duplicated here only so the console can
# explain *before* a turn is spent; the actual gate lives upstream.
PAID_ONLY_MODELS: frozenset[str] = frozenset(
    {"MiniMax-H3", "MiniMax-H3-Max", "MiniMax-H3-Max-Turbo"}
)

# A model that can be paid for by a Token Plan allowance or gift credits.
TOKEN_PLAN_MODELS: frozenset[str] = frozenset({"MiniMax-Hailuo-2.3"})


def model_needs_m_plan(upstream_model: str) -> bool:
    """True when the model is billed from the M Plan / paid-credit bucket."""
    return (upstream_model or "").strip() in PAID_ONLY_MODELS


# A compact regex for the upstream's `402` body, kept for callers that want
# to match the raw JSON rather than the prose around it.
RAW_402 = re.compile(r'"code"\s*:\s*402')
