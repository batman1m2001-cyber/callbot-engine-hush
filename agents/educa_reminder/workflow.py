"""Educa Reminder LLM Workflow — Hush graph implementation.

Flow: normalize → quick_detect → [classify_intent | skip] → merge_intent
      → state_transition → generate_rule → [generate_llm | skip] → merge_response
"""

from hush.core import graph, PARENT, START, END
from hush.core.ops.flow import if_

from hush.providers.ops import ask, chat

from agents.educa_reminder.ops.normalize import normalize_text
from agents.educa_reminder.ops.quick_detect import quick_detect
from agents.educa_reminder.ops.build_intent_context import build_intent_context
from agents.educa_reminder.ops.state_transition import state_transition
from agents.educa_reminder.ops.generate_rule import generate_rule
from agents.educa_reminder.ops.merge import merge_intent, merge_response
from agents.educa_reminder.ops.skip import skip_classify, skip_generate
from agents.educa_reminder.ops.call_result import build_call_result


@graph
def educa_workflow(
    customer_speech,
    agent_speech,
    current_state,
    script_data,
    intent_retry_counts,
    conversation_history,
):
    # ── Node 1: Normalize ──
    norm = normalize_text(customer_speech=customer_speech)

    # ── Node 2: Quick detect ──
    detect = quick_detect(
        normalized_text=norm["normalized_text"],
        current_state=current_state,
    )

    # ── Define both paths ──

    # LLM path
    ctx = build_intent_context(
        current_state=current_state,
        normalized_text=norm["normalized_text"],
        agent_speech=agent_speech,
        script_data=script_data,
    )
    classify = ask(
        resource="default",
        template={"system": "{analyzer_system_prompt}", "user": "{intent_prompt}"},
        fields=["result: str"],
        parser="xml",
        retry=1,
        validators=ctx["validators"],
        analyzer_system_prompt=ctx["analyzer_system_prompt"],
        intent_prompt=ctx["intent_prompt"],
    )

    # Quick path
    skip_cls = skip_classify(
        intent=detect["intent"],
        confidence=detect["confidence"],
        extraction_data=detect["extraction_data"],
    )

    # ── Branch 1: needs_llm? ──
    router1 = if_(detect["needs_llm"] == True, ctx).else_(skip_cls)

    # ── Merge intent ──
    m_intent = merge_intent(
        quick_intent=skip_cls["quick_intent"],
        quick_confidence=skip_cls["quick_confidence"],
        quick_extraction_data=skip_cls["quick_extraction_data"],
        llm_result=classify["result"],
    )

    # ── Node 5: State transition ──
    trans = state_transition(
        intent=m_intent["intent"],
        current_state=current_state,
        intent_retry_counts=intent_retry_counts,
        extraction_data=m_intent["extraction_data"],
    )

    # ── Node 6: Generate rule response ──
    gen_rule = generate_rule(
        new_state=trans["new_state"],
        intent=m_intent["intent"],
        previous_state=trans["previous_state"],
        state_retry_count=0,
        script_data=script_data,
        conversation_history=conversation_history,
    )

    # ── Define both gen paths ──
    gen_llm = chat(
        resource="default",
        template={"system": "Bạn là trợ lý tổng đài. Trả lời ngắn gọn, lịch sự.", "user": "{llm_prompt_context}"},
        llm_prompt_context=gen_rule["llm_prompt_context"],
    )
    skip_gen = skip_generate(rule_response=gen_rule["rule_response"])

    # ── Branch 2: needs_llm_generate? ──
    router2 = if_(gen_rule["needs_llm_generate"] == True, gen_llm).else_(skip_gen)

    # ── Merge response ──
    m_response = merge_response(
        rule_response=skip_gen["rule_response"],
        llm_content=gen_llm["content"],
        new_state=trans["new_state"],
        script_data=script_data,
    )

    # ── Node 7: CRM call result ──
    crm = build_call_result(
        current_state=trans["new_state"],
        intent=m_intent["intent"],
        previous_state=trans["previous_state"],
        customer_speech=customer_speech,
        customer_confirmed=trans["customer_confirmed"],
        new_phone_number=trans["new_phone_number"],
        script_data=script_data,
    )

    # ── Forward key outputs to PARENT ──
    m_intent["intent"] >> PARENT["intent"]
    m_intent["extraction_data"] >> PARENT["extraction_data"]
    trans["new_state"] >> PARENT["new_state"]
    trans["should_transfer"] >> PARENT["should_transfer"]
    trans["should_hangup"] >> PARENT["should_hangup"]
    crm["call_result"] >> PARENT["call_result"]

    # ── Wiring ──
    START >> norm >> detect >> router1
    router1 >> ctx >> classify >> ~m_intent
    router1 >> skip_cls >> ~m_intent
    m_intent >> trans >> [gen_rule, crm]
    gen_rule >> router2
    router2 >> gen_llm >> ~m_response >> END
    router2 >> skip_gen >> ~m_response
