"""Verifier 断言去重：同一组数字只保留一条。"""

from __future__ import annotations

from agents.verifier import VerifiedClaim, _dedupe_verified_claims


def test_duplicate_verified_claims_collapse() -> None:
    claims = [
        VerifiedClaim(
            claim_id=f"c{index}",
            text="比亚迪2024年ROE为21.73%，相比2023年的21.64%基本持平，微升0.09个百分点。",
            verdict="verified",
            label="✅已验证",
            numbers={"roe": 0.2173},
        )
        for index in range(3)
    ]
    unique = _dedupe_verified_claims(claims)
    assert len(unique) == 1
    assert unique[0].numbers["roe"] == 0.2173


def test_verified_wins_over_unverified_duplicate() -> None:
    unverified = VerifiedClaim(
        claim_id="u",
        text="ROE 21.73%",
        verdict="unverified",
        numbers={"roe": 0.2173},
    )
    verified = VerifiedClaim(
        claim_id="v",
        text="ROE 21.73%",
        verdict="verified",
        numbers={"roe": 0.2173},
    )
    unique = _dedupe_verified_claims([unverified, verified])
    assert len(unique) == 1
    assert unique[0].verdict == "verified"
