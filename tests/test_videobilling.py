"""Billing detectors for video turns, replayed against captured real turns.

The fixtures under `fixtures/` are verbatim dumps of five live sessions: an
H3-Max run that asked for a billing confirmation and was then refused, an H3
run refused straight away, a bare `50110:insufficient balance`, a second
H3-Max run that worded the same confirmation gate differently, and — the
negative that matters most — a Hailuo run that produced a real video.

The successful turn is the reason these tests exist.  It talks about billing
("计费：按你确认的走") and it must be read as neither a confirmation request
nor a refusal, because a detector that fires on it would abort every good
job.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import videobilling

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def assistant_texts(name: str) -> list[str]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    out: list[str] = []
    for turn in payload.get("summaries") or []:
        reply = turn.get("assistant_response")
        if isinstance(reply, dict) and reply.get("content_head"):
            out.append(str(reply["content_head"]))
        else:
            out.append("")
    return out


def test_h3max_first_turn_asks_for_a_billing_confirmation():
    turn = assistant_texts("fx_h3max.json")[0]
    verdict = videobilling.classify(turn)
    assert verdict.needs_confirmation is True
    assert verdict.refused is False


def test_h3max_second_turn_reports_the_refusal():
    turn = assistant_texts("fx_h3max.json")[1]
    verdict = videobilling.classify(turn)
    assert verdict.refused is True
    assert verdict.needs_confirmation is False
    assert "M Plan" in verdict.reason or "余额" in verdict.reason


def test_h3_refusal_is_detected():
    verdict = videobilling.classify(assistant_texts("fx_h3cheap.json")[0])
    assert verdict.refused is True


def test_bare_upstream_error_code_is_a_refusal():
    verdict = videobilling.classify(assistant_texts("fx_insuff.json")[0])
    assert verdict.refused is True


def test_a_differently_worded_confirmation_gate_is_detected():
    """A second live session phrased the same gate in its own words.

    It said it needed "用户对本次提交的明确确认" and called a prior
    authorisation explicitly insufficient.  The detector matches the meaning,
    not one lucky transcript.
    """
    verdict = videobilling.classify(assistant_texts("fx_cancelled.json")[0])
    assert verdict.needs_confirmation is True
    assert verdict.refused is False


def test_a_real_video_is_neither_a_refusal_nor_a_question():
    """The negative case: a finished job's reply also discusses billing."""
    verdict = videobilling.classify(assistant_texts("fx_hailuo.json")[0])
    assert verdict.refused is False
    assert verdict.needs_confirmation is False
    assert verdict.actionable is False


def test_raw_402_body_is_a_refusal():
    raw = (
        '{"code":402,"message":"the remaining M Plan weekly limit and account '
        'credits cannot cover this MiniMax-H3 video generation; this task costs '
        '480 credits"}'
    )
    assert videobilling.detect_refusal(raw).refused is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "好的，已收到。",
        # Explaining the billing rules is not the same as asking for a reply
        # and is not the same as reporting a failed order.
        "这是一段关于计费规则的说明：H3 走 M Plan 周限额。",
        "计费：按你确认的走，H2.3 走账户积分/周限额",
    ],
)
def test_ordinary_prose_stays_inert(text):
    verdict = videobilling.classify(text)
    assert verdict.actionable is False


def test_only_h3_models_need_the_m_plan_bucket():
    assert videobilling.model_needs_m_plan("MiniMax-H3") is True
    assert videobilling.model_needs_m_plan("MiniMax-H3-Max") is True
    assert videobilling.model_needs_m_plan("MiniMax-Hailuo-2.3") is False
