# Educa Reminder LLM Workflow Spec

## Overview

Luong LLM workflow nhan customer_speech (str) va tra ra response (str).
Tach hoan toan khoi shared_state, Redis, TTS, WebSocket.
Moi node chi dung primitive types: str, int, float, bool, list, dict.

## Nodes

### Node 1: normalize_text

Pure text processing, khong LLM.

| | |
|---|---|
| **Input** | `customer_speech: str` |
| **Output** | `normalized_text: str` |
| **Logic** | Lowercase, remove special chars, collapse whitespace |

```
re.sub(r'[^\w\s\+]', ' ', text)
re.sub(r'\s+', ' ', text).strip().lower()
```

---

### Node 2: quick_detect

Rule-based detection, khong LLM. Bypass LLM khi co the.

| | |
|---|---|
| **Input** | `normalized_text: str`, `current_state: str`, `script_data: dict` |
| **Output** | `intent: str \| None`, `confidence: float`, `extraction_data: dict`, `needs_llm: bool` |

**Branching:**

```
IF normalized_text rong:
    -> intent="silent", confidence=0.95, needs_llm=False

IF phone pattern match (0[0-9]{9}) AND current_state=="ASK_PHONE":
    IF valid phone:
        -> intent="read_phone", extraction_data={"phone_number": "09xx"}, needs_llm=False
    ELSE:
        -> intent="invalid_phone", needs_llm=False

ELSE:
    -> intent=None, needs_llm=True
```

---

### Node 3: classify_intent (LLM)

Goi LLM de phan loai intent. Chi chay khi needs_llm=True.

| | |
|---|---|
| **Input** | `normalized_text: str`, `agent_speech: str`, `current_state: str`, `allowed_intents: list[str]`, `script_data: dict`, `intent_definitions: list[dict]` |
| **Output** | `intent: str`, `confidence: float`, `raw_llm_response: str` |

**LLM Config:**
- max_tokens: 30
- temperature: 0.0
- model_key: "intent"
- parser: xml, extract `<result>intent_name</result>`
- validators: allowed_intents cho state hien tai (vd: `["confirm", "deny", "@fallback"]`)

**Hush implementation:**

Truoc classify_intent, can 1 custom `@op build_intent_context` de build XML blocks dynamic
(loop qua allowed_intents per state, build intent definitions XML, few-shot examples XML).

`build_intent_context` tra ra:
- `analyzer_system_prompt: str` — system prompt co dinh
- `intent_prompt: str` — user message da format (XML intents + context)
- `allowed_intents_list: list[str]` — cho validators dynamic

Sau do dung `extract()` voi:
```python
e = extract(
    template={"system": "{analyzer_system_prompt}", "user": "{intent_prompt}"},
    fields=["result: str"],
    parser="xml",
    retry=1,
    validators={"result": PARENT["allowed_intents_list"]},  # dynamic per state, @fallback
    # template vars forwarded tu build_intent_context qua PARENT
    analyzer_system_prompt=build_ctx["analyzer_system_prompt"],
    intent_prompt=build_ctx["intent_prompt"],
)
```

PromptOp dict template tu tach system/user messages.
Khong can sua PromptOp.

---

### Node 4: merge_intent

Merge ket qua tu quick_detect hoac classify_intent.

| | |
|---|---|
| **Input** | `quick_intent: str \| None`, `llm_intent: str \| None`, `quick_confidence: float`, `llm_confidence: float`, `extraction_data: dict` |
| **Output** | `intent: str`, `confidence: float`, `extraction_data: dict` |

**Logic:**
```
IF quick_intent is not None:
    -> intent=quick_intent, confidence=quick_confidence
ELSE:
    -> intent=llm_intent or "fallback", confidence=llm_confidence or 0.5
```

---

### Node 5: state_transition

Pure logic, khong LLM. Chuyen state dua tren intent.

| | |
|---|---|
| **Input** | `intent: str`, `current_state: str`, `intent_retry_counts: dict[str, int]`, `extraction_data: dict` |
| **Output** | `new_state: str`, `previous_state: str`, `should_transfer: bool`, `should_hangup: bool`, `intent_retry_counts: dict[str, int]` (updated), `call_result: dict` |

**Logic tom tat:**

```
# Immediate finish intents
IF intent IN {student_joining, already_absent, request_absent, cancel_course, deny}:
    -> new_state="FINISH", should_hangup=True

# Transfer hotline intents
IF intent IN {technical_issue, wrong_schedule, ask_about_program}:
    -> new_state="TRANSFER_HOTLINE", should_transfer=True

# Confirm
IF intent == "confirm" AND current_state == "CONFIRM_CUSTOMER":
    -> new_state="REMINDER"

# Retry intents (need_support, busy, fallback, unclear)
IF intent IN retry_intents:
    intent_retry_counts[intent] += 1
    IF intent_retry_counts[intent] >= max_retries[intent]:
        -> new_state="FINISH" hoac "TRANSFER_HOTLINE"
    ELSE:
        -> new_state=current_state (stay, loopback)

# Phone intents (ASK_PHONE state)
IF intent == "this_number": -> new_state="REMINDER"
IF intent == "read_phone":  -> new_state="FINISH"
IF intent == "other_number": -> retry hoac FINISH

# Wrong info
IF intent == "wrong_student_name": -> new_state="CONFIRM_CUSTOMER"
IF intent == "wrong_support_number": -> new_state="ASK_PHONE"
IF intent == "wrong_number":
    IF current_state == "REMINDER": -> "CONFIRM_CUSTOMER"
    ELSE: -> "FINISH"
```

**Max retries:**
```
need_support: 2, busy: 2, unclear: 3, fallback: 2, other_number: 2, invalid_phone: 2
```

**Call result mapping:**
```
(state, intent) -> {arid: str, comment: str}
VD: ("REMINDER", "student_joining") -> {arid: "SUCCESS_JOINING", comment: "Hoc sinh xac nhan vao lop"}
```

---

### Node 6: generate_rule_response

Rule-based response generation. Tra response tu template database.

| | |
|---|---|
| **Input** | `new_state: str`, `intent: str`, `previous_state: str`, `state_retry_count: int`, `script_data: dict` |
| **Output** | `response: str \| None`, `needs_llm_generate: bool` |

**Logic:**
```
templates = RESPONSE_DB[new_state][intent]  # list of template strings

IF templates:
    template = templates[min(state_retry_count, len(templates)-1)]
    response = substitute_variables(template, script_data)
    -> response=response, needs_llm_generate=False
ELSE:
    -> response=None, needs_llm_generate=True
```

**Variable substitution:**
```
"Chao anh chi, be [student_name] co buoi hoc luc [class_time]"
-> "Chao anh chi, be Minh co buoi hoc luc 19:00"

Keys: student_name, class_time, program_name, agent_name, company, hotline, ...
```

**Data source:** Python dict, copy tu `educa_reminder_parents_conversation_data.py`.
Chuyen sang YAML/DB sau khi test luong OK.

---

### Node 7: generate_llm_response (LLM)

Goi LLM generate response. Chi chay khi needs_llm_generate=True.

| | |
|---|---|
| **Input** | `current_state: str`, `intent: str`, `script_data: dict`, `conversation_history: list[dict]`, `state_retry_count: int` |
| **Output** | `response: str` |

**LLM Config:**
- max_tokens: 150
- temperature: 0.4
- model_key: "generation"

**Mapping Hush:** Dung `chat()`

---

### Node 8: merge_response

Merge ket qua tu rule-based hoac LLM generate.

| | |
|---|---|
| **Input** | `rule_response: str \| None`, `llm_response: str \| None`, `new_state: str`, `script_data: dict` |
| **Output** | `response: str` |

**Logic:**
```
IF rule_response:
    -> response=rule_response
ELSE IF llm_response:
    -> response=llm_response
ELSE:
    -> response=FALLBACK_RESPONSES[new_state]  # hardcoded ultimate fallback
```

**Ultimate fallback per state:**
```
REMINDER:          "Da em chua nghe ro a. Anh chi co the noi lai duoc khong a?"
CONFIRM_CUSTOMER:  "Da anh chi vui long xac nhan giup em..."
ASK_PHONE:         "Da anh chi vui long cho em xin so dien thoai a."
TALK_WITH_STUDENT: "Da co chua nghe ro..."
TRANSFER_HOTLINE:  "Da anh chi vui long lien he hotline {hotline}..."
FINISH:            "Da em cam on anh chi a. Chao anh chi."
```

---

## Graph Wiring

```
                    +--> classify_intent --+
                    |    (LLM, extract)    |
START >> normalize  |                      v
     >> quick_detect+--> [skip] ----------> merge_intent
                                                |
                                                v
                                         state_transition
                                                |
                    +--> generate_llm --+        v
                    |    (LLM, chat)   |  generate_rule
                    |                  v        |
                    +-- [skip] ------> merge_response >> END
```

**Routing & Wiring:**
- `if_(quick_detect["needs_llm"] == True)` -> classify_intent, else skip_classify
- `if_(generate_rule["needs_llm_generate"] == True)` -> generate_llm, else skip_generate
- Merge dung soft edge `>>~`: `[classify_intent, skip_classify] >>~ merge_intent`
- Soft edge dam bao chi path nao chay moi gui output, path ko chay -> input = None (default)
- Skip ops la pass-through `@op` forward data tu quick_detect/generate_rule
- `extract()` tra ve GraphOp, nest truc tiep trong parent graph: `router >> e >>~ merge`

---

## State & Intent Enums

### States
```
REMINDER, CONFIRM_CUSTOMER, ASK_PHONE, TALK_WITH_STUDENT,
TRANSFER_HOTLINE, FINISH
```

### Intents
```
confirm, deny, busy, unclear, fallback, silent,
student_joining, already_absent, request_absent, cancel_course,
need_support, technical_issue, wrong_schedule, ask_about_program,
wrong_student_name, wrong_support_number, wrong_number,
student_pickup, this_number, read_phone, other_number, invalid_phone
```

### Allowed Intents per State

```yaml
REMINDER:
  - need_support, student_joining, already_absent, request_absent
  - wrong_schedule, cancel_course, technical_issue, wrong_student_name
  - wrong_support_number, wrong_number, student_pickup, busy, deny
  - unclear, fallback, ask_about_program, confirm

CONFIRM_CUSTOMER:
  - confirm, wrong_number, busy, deny, unclear, fallback

ASK_PHONE:
  - this_number, read_phone, other_number, invalid_phone, fallback

TALK_WITH_STUDENT:
  - need_support, student_joining, already_absent, request_absent
  - wrong_schedule, technical_issue, unclear, fallback
```

### Intent Sets (for state_transition)

```python
IMMEDIATE_END_INTENTS = {"student_joining", "already_absent", "request_absent", "cancel_course"}
TRANSFER_HOTLINE_INTENTS = {"technical_issue", "wrong_schedule", "ask_about_program"}
RETRY_INTENTS = {"need_support", "busy", "unclear", "fallback", "other_number", "invalid_phone"}
TERMINAL_STATES = {"FINISH", "TRANSFER_HOTLINE"}
```

### INTENT_DEFINITIONS (for build_intent_context)

21 intents, moi cai co description + examples. Load tu YAML.
```yaml
# Vi du:
confirm:
  description: "Phu huynh xac nhan dung thong tin"
  examples: ["vang dung roi", "uh dung", "phai roi"]
student_joining:
  description: "Hoc sinh dang vao lop"
  examples: ["con dang vao roi", "be dang hoc roi", ...]
# ... 19 intents khac
```

### FEW_SHOT_EXAMPLES (for build_intent_context)

3-4 examples per state. Load tu YAML.
```yaml
REMINDER:
  - agent: "Chao anh chi, em goi tu chuong trinh..."
    customer: "con dang vao roi"
    intent: "student_joining"
  - ...
```

### STATE_GREETINGS (for generate_rule, khi retry_count=0)

6 variations per state — greeting khi vao state moi.
```python
STATE_GREETINGS = {
    "REMINDER": ["Chao anh chi, em lien he tu chuong trinh...", ...],  # 6 variants
    "CONFIRM_CUSTOMER": ["Da vang, bo me cho em xac nhan lai...", ...],
    "ASK_PHONE": ["Da anh chi vui long cho em xin so dien thoai...", ...],
    "TALK_WITH_STUDENT": ["Chao con, co la co giao...", ...],
}
```

### FINISH/TRANSFER Response Templates (for generate_rule)

Phu thuoc previous_state + intent:
```python
FINISH_RESPONSE_TEMPLATES = {
    "REMINDER": {
        "student_joining": ["Da cam on anh chi, chuc be hoc tot...", ...],
        "deny": ["Da vang a, em cam on anh chi...", ...],
        ...
    },
    "CONFIRM_CUSTOMER": {...},
    ...
}
TRANSFER_HOTLINE_RESPONSE_TEMPLATES = {
    "REMINDER": {
        "technical_issue": ["Da de em chuyen may cho bo phan ho tro...", ...],
        ...
    },
    ...
}
```

### CALL_RESULT_MAPPING (for state_transition output)

60+ entries, map (state, intent) -> CRM result:
```yaml
REMINDER__student_joining:
  code: "SUCCESS_JOINING"
  status: "success"
  comment: "Hoc sinh xac nhan vao lop"
  action: "joined"
# ...
```

### _STATE_GUIDANCE (for generate_llm prompt context)

```python
STATE_GUIDANCE = {
    "REMINDER": "Muc tieu: nhac lich hoc, xac nhan hoc sinh se vao lop",
    "CONFIRM_CUSTOMER": "Muc tieu: xac nhan dung phu huynh/hoc sinh",
    "ASK_PHONE": "Muc tieu: lay so dien thoai lien lac",
    "TALK_WITH_STUDENT": "Muc tieu: noi chuyen voi hoc sinh, dong vien vao lop",
    "TRANSFER_HOTLINE": "Muc tieu: chuyen sang hotline ho tro",
    "FINISH": "Muc tieu: ket thuc cuoc goi lich su",
}
```

### _RESPONSE_OPENINGS (for generate_llm prefix)

```python
RESPONSE_OPENINGS = ["Da vang a", "Da", "Vang a", "Da em hieu a"]
```

### key_replaces (for variable substitution)

```python
KEY_REPLACES = ["agent_name", "student_name", "class_time", "program_name",
                "company", "hotline", "subject", "teacher_name", "class_link"]
```

---

## Conversation History Format

```python
[
    {"role": "agent", "content": "Chao anh chi..."},
    {"role": "customer", "content": "vang dung roi"},
    {"role": "agent", "content": "Da cam on anh chi..."},
    ...
]
```

Dung cho: LLM intent prompt (context) va LLM response generation.

---

## Script Data Format

```python
{
    "student_name": "Minh",
    "class_time": "19:00",
    "program_name": "AI CLASS",
    "agent_name": "Linh",
    "company": "Edupia",
    "hotline": "1900xxxx",
    "phone_number": "0912345678",
    "call_result": {"arid": None, "comment": None}
}
```
