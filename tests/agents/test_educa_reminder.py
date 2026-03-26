"""Test educa reminder workflow — real LLM."""

import asyncio
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

from hush.core import Hush
from agents.educa_reminder.workflow import educa_workflow


SCRIPT_DATA = {
    "student_name": "Minh",
    "class_time": "19:00",
    "program_name": "AI CLASS",
    "agent_name": "Linh",
    "company": "Edupia",
    "hotline": "1900636464",
    "phone_number": "0912345678",
}


async def test_case(name, customer_speech, agent_speech, current_state, intent_retry_counts=None):
    """Run one test case."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  State: {current_state}")
    print(f"  Customer: \"{customer_speech}\"")
    print(f"  Agent: \"{agent_speech[:50]}...\"" if len(agent_speech) > 50 else f"  Agent: \"{agent_speech}\"")

    wf = educa_workflow(
        customer_speech=customer_speech,
        agent_speech=agent_speech,
        current_state=current_state,
        script_data=SCRIPT_DATA,
        intent_retry_counts=intent_retry_counts or {},
        conversation_history=[],
    )
    engine = Hush(wf, env=str(ROOT / ".env"), resources=str(ROOT / "resources.yaml"))
    result = await engine.run(inputs={})

    print(f"  → Intent: {result.get('intent')}")
    print(f"  → New State: {result.get('new_state')}")
    print(f"  → Response: \"{result.get('response', '')[:100]}\"")
    print(f"  → Transfer: {result.get('should_transfer')}, Hangup: {result.get('should_hangup')}")
    return result


async def main():
    print("Educa Reminder Workflow Tests")
    print("=" * 60)

    # Test 1: Student joining — quick detect bypass, rule response
    await test_case(
        "Student joining (REMINDER)",
        "bé đang vào rồi em",
        "Chào anh chị, em gọi từ chương trình AI CLASS nhắc lịch học của bé Minh lúc 19:00",
        "REMINDER",
    )

    # Test 2: Silent — quick detect
    await test_case(
        "Silent (REMINDER)",
        "",
        "Chào anh chị",
        "REMINDER",
    )

    # Test 3: Confirm customer — LLM classify
    await test_case(
        "Confirm (CONFIRM_CUSTOMER)",
        "vâng đúng rồi",
        "Anh chị có phải là phụ huynh của bé Minh không ạ?",
        "CONFIRM_CUSTOMER",
    )

    # Test 4: Busy — LLM classify
    await test_case(
        "Busy (REMINDER)",
        "đang bận lắm",
        "Chào anh chị, em gọi từ chương trình AI CLASS",
        "REMINDER",
    )

    # Test 5: Technical issue — LLM classify, transfer hotline
    await test_case(
        "Technical issue (REMINDER)",
        "con vào mãi không được em ạ",
        "Chào anh chị, bé Minh có buổi học lúc 19:00",
        "REMINDER",
    )

    # Test 6: Phone number — quick detect in ASK_PHONE
    await test_case(
        "Read phone (ASK_PHONE)",
        "số 0987654321 em nhé",
        "Anh chị cho em xin số điện thoại liên hệ ạ",
        "ASK_PHONE",
    )

    # Test 7: Fallback with retry
    await test_case(
        "Fallback retry (REMINDER)",
        "trời mưa quá",
        "Bé Minh có buổi học lúc 19:00 anh chị nhắc bé vào lớp giúp em nhé",
        "REMINDER",
        intent_retry_counts={"fallback": 1},
    )

    print(f"\n{'='*60}")
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
