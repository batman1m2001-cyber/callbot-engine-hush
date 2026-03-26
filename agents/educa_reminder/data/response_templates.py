"""
Educa Reminder Parents Conversation Data - Rule-based response templates.

Module này quản lý các response templates cho Educa Reminder Parents Agent.
Cập nhật đa thoại (5 variations) theo RESPONSE_TEMPLATES trong educa_reminder_parents_prompts.py
"""

import json
import os
import secrets
import logging
import time
from typing import Dict, Any, Union, List

logger = logging.getLogger(__name__)

# JSON files nằm trong db/educa_reminder_parents/
DB_DIR = os.path.join(os.path.dirname(__file__), 'educa_reminder_parents')


# =============================================================================
#                           HELPER FUNCTIONS
# =============================================================================

def _load_data(json_name: str) -> Union[list, dict]:
    """Load data từ JSON file."""
    json_path = os.path.join(DB_DIR, json_name)
    try:
        with open(json_path, 'r', encoding="utf-8-sig") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return []  # Return empty list if file not found


def _save_data(json_name: str, data: Any) -> None:
    """Save data to JSON file."""
    json_path = os.path.join(DB_DIR, json_name)
    with open(json_path, 'w', encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def trigger_transfer_hotline_event(script_data: dict) -> Dict[str, Any]:
    """
    Tạo event data để gửi cho đối tác nhà mạng khi cần transfer cuộc gọi.
    
    Event này sẽ được gửi qua WebSocket để nhà mạng có thể bắt và thực hiện transfer.
    
    Args:
        script_data: Script data chứa thông tin hotline và các thông tin liên quan
        
    Returns:
        Dict chứa event data để gửi qua WebSocket
    """
    hotline = script_data.get('hotline', '0931208686')
    
    # Tạo event payload theo format mà nhà mạng mong đợi
    event_data = {
        'event': 'transfer_hotline',
        'transfer_to': hotline,
        'reason': 'customer_support_request',
        'timestamp': int(time.time() * 1000),
        'metadata': {
            'program_name': script_data.get('program_name', ''),
            'company': script_data.get('company', ''),
            'agent_name': script_data.get('agent_name', ''),
        }
    }
    
    logger.info(f"Transfer hotline event triggered: {event_data}")
    return event_data


def agent_sentence(sentences: list, script_data: dict, other_script_data: dict = None) -> str:
    """Chọn ngẫu nhiên 1 câu từ list và format với script_data."""
    if not sentences:
        return ""  # Return empty string if sentences list is empty
    sentence = secrets.choice(sentences)
    return refactor_sentence(sentence, script_data, other_script_data)


def refactor_sentence(sentence: str, script_data: dict, other_script_data: dict = None) -> str:
    """Thay thế các placeholder [key] trong câu bằng giá trị từ script_data."""
    if other_script_data:
        for key in key_other_replaces:
            value = other_script_data.get(key)
            if value and str(value).strip():
                sentence = sentence.replace(f'[{key}]', str(value))

    for key in key_replaces:
        value = script_data.get(key)
        if value and str(value).strip():
            sentence = sentence.replace(f'[{key}]', str(value))

    return sentence


# =============================================================================
#                           KEY REPLACEMENTS
# =============================================================================

# Keys to replace in sentences from script_data
key_replaces = [
    'agent_name',
    'student_name',
    'class_time',
    'program_name',
    'company',
    'hotline',
    'subject',
    'teacher_name',
    'class_link',
]

# Keys to replace from other_script_data (if any)
key_other_replaces = []


# =============================================================================
#                           GREETING SENTENCES
# =============================================================================
# Thoại khi mới vào state REMINDER - 5 variations
agent_state_reminders = [
    "Chào anh chị, em liên hệ từ chương trình on lai [program_name]. Hôm nay bé [student_name] có lịch học vào lúc [class_time] mà chưa thấy bé vào lớp, không biết gia đình có vấn đề gì cần em hỗ trợ không?",
    "Dạ em chào anh/chị, em gọi đến từ chương trình [program_name]. Hiện tại lớp học lúc [class_time] đã bắt đầu nhưng em chưa thấy bé [student_name] tham gia, không biết gia đình mình có quên lịch hay gặp sự cố gì không?",
    "Chào phụ huynh bé [student_name], em là cán bộ quản lý lớp của [program_name]. Đã đến giờ học [class_time] nhưng giáo viên chưa thấy con vào lớp, anh/chị kiểm tra giúp em xem con có gặp khó khăn gì cần hỗ trợ không nhé?",
    "Dạ xin chào anh/chị, em liên hệ từ lớp học trực tuyến [program_name]. Em thấy bé [student_name] vẫn chưa có mặt trong buổi học lúc [class_time], không biết có vấn đề gì về mạng hay máy tính khiến con chưa vào được không?",
    "Em chào gia đình, em là tư vấn viên của [program_name]. Ca học [class_time] của bé [student_name] đã diễn ra nhưng bé vẫn vắng mặt, không biết gia đình có việc bận đột xuất hay cần em giúp đỡ gì để bé vào học không?",
    "Chào anh/chị, em đại diện cho chương trình [program_name] xin phép gọi. Hệ thống ghi nhận bé [student_name] chưa vào lớp [class_time], anh/chị xem giúp em có lý do gì cản trở việc học của bé hôm nay không?",
]


# =============================================================================
#                           CONFIRM CUSTOMER
# =============================================================================
# Thoại khi mới vào state CONFIRM_CUSTOMER - 5 variations
agent_state_confirm_customers = [
    "Dạ vâng, bố mẹ cho em xác nhận lại. Nhà mình đang có bạn tham gia chương trình [program_name] đúng không?",
    "Dạ vâng, xin phép cho em hỏi lại một chút là gia đình mình có bé đang theo học tại chương trình [program_name] phải không?",
    "Dạ, để tiện hỗ trợ thì anh/chị xác nhận giúp em là nhà mình có bé đang tham gia khóa học của [program_name] đúng không?",
    "Dạ vâng, cho em xin phép xác minh thông tin, hiện tại bé nhà mình đang là học viên của chương trình [program_name] đúng không anh/chị?",
    "Dạ, phiền anh/chị xác nhận lại giúp em, có phải gia đình đang cho bé theo học chương trình [program_name] bên em không?",
    "Dạ vâng, em muốn xác nhận lại thông tin là nhà mình đang có bạn nhỏ tham gia lớp học của [program_name] phải không?"
]


# =============================================================================
#                           ASK_PHONE
# =============================================================================
# Thoại khi mới vào state ASK_PHONE - 5 variations
agent_state_ask_phone = [
    "Dạ vậy anh chị cho em xin lại số điện thoại để liên hệ hỗ trợ cho con.",
    "Dạ vâng, anh/chị vui lòng cung cấp lại giúp em số điện thoại để bên em tiện liên hệ hỗ trợ kỹ thuật cho bé ngay.",
    "Dạ, để việc hỗ trợ con được nhanh chóng nhất, anh/chị cho em xin lại số điện thoại liên lạc chính xác được không?",
    "Dạ vậy phiền anh/chị đọc giúp em số điện thoại hiện tại để bộ phận hỗ trợ có thể gọi và hướng dẫn bé vào lớp.",
    "Dạ, anh/chị cho em xin số điện thoại để em báo bộ phận kỹ thuật gọi lại xử lý vấn đề cho con ngay nhé.",
    "Dạ vâng, anh/chị làm ơn cho em xin lại số điện thoại cầm tay để em liên hệ hỗ trợ trực tiếp cho việc học của bé."
]


# =============================================================================
#                           TALK_WITH_STUDENT
# =============================================================================
# Thoại khi mới vào state TALK_WITH_STUDENT - 5 variations
agent_state_talk_with_students = [
    "Cô chào con, hôm nay con có lịch học vào lúc [class_time] mà cô chưa thấy con vào lớp, không biết con có vấn đề gì cần cô hỗ trợ không nha?",
    "Cô chào con, lớp học của mình lúc [class_time] đã bắt đầu rồi mà cô chưa thấy nick con sáng, con có gặp trục trặc gì cần cô giúp không?",
    "Chào con, cô nhắc nhẹ là mình có lịch học lúc [class_time] nhé. Cô chưa thấy con vào, không biết con có quên lịch hay cần cô hỗ trợ gì không ta?",
    "Cô chào con, đến giờ học [class_time] rồi nè mà cô tìm chưa thấy con trong lớp. Con có đang vướng mắc gì cần cô hỗ trợ để vào học ngay không?",
    "Chào con nha, hôm nay con có ca học lúc [class_time] đó. Cô đợi mãi chưa thấy con, con có cần cô hướng dẫn hay giúp đỡ gì để vào lớp không?",
    "Cô chào con, lịch học [class_time] của con đã đến giờ rồi. Cô thấy con chưa vào, có vấn đề gì thì báo cô để cô hỗ trợ con nha?"
]


# =============================================================================
#                           RESPONSE BY STATE - INTENT - REMINDER
# =============================================================================
# Thoại khi vào state REMINDER theo intent - 5 variations mỗi intent
state_reminder_intents = {
    # --- NEED_SUPPORT: Phụ huynh cần hỗ trợ ---
    'NEED_SUPPORT': [
        "Dạ, không biết con có đang cần hỗ trợ vấn đề gì mà chưa vào học kịp thời?",
        "Dạ, không biết con có gặp khó khăn hay vướng mắc gì khiến việc vào lớp bị chậm trễ không?",
        "Dạ, có vấn đề gì đang cản trở con tham gia lớp học cần em hỗ trợ xử lý ngay không?",
        "Dạ, không biết con có đang vướng ở khâu nào mà chưa thấy con vào học được?",
        "Dạ, gia đình mình có cần em giúp đỡ gì thêm để con có thể vào lớp kịp thời ngay lúc này không?",
        "Dạ, không biết có sự cố kỹ thuật hay vấn đề gì làm gián đoạn việc vào học của con không?"
    ],

    # --- BUSY: Phụ huynh bận ---
    'BUSY': [
        "Dạ em xin phép thông báo nhanh là bé có lịch học tại chương trình ây ai cờ lát lúc [class_time] nhưng hiện tại chưa thấy con vào lớp, không biết gia đình mình có đang gặp vấn đề gì cần em hỗ trợ không ạ?",
        "Dạ em gọi để thông báo nhanh với bố mẹ là bé có ca học ở chương trình ây ai cờ lát lúc [class_time] mà em chưa thấy con vào, không biết anh chị có cần em giúp đỡ vướng mắc gì để bé vào lớp không ạ?",
        "Dạ em xin phép báo nhanh thông tin bé có lịch học chương trình ây ai cờ lát vào [class_time] nhưng hiện chưa thấy con tham gia, gia đình mình có đang gặp khó khăn gì cần em hỗ trợ không ạ?",
        "Dạ em chỉ xin phép báo nhanh để gia đình nhắc bé có lịch học tại ây ai cờ lát lúc [class_time] mà hiện chưa thấy con online, không biết anh chị có cần em hỗ trợ vấn đề gì không ạ?",
        "Dạ em xin phép nhắc nhẹ bố mẹ là chương trình ây ai cờ lát của con có lịch lúc [class_time] mà chưa thấy bé vào, không biết bên mình có cần em hỗ trợ gì để con tham gia lớp được thuận lợi không ạ?",
        "Dạ em xin thông tin nhanh về lịch học của con tại ây ai cờ lát lúc [class_time], do em chưa thấy bé có mặt trong lớp nên muốn hỏi xem gia đình có trục trặc gì cần em hỗ trợ ngay không ạ?",
        "Dạ em xin phép thông báo nhanh để anh chị nhắc con vào lớp ây ai cờ lát có lịch lúc [class_time] vì em chưa thấy bé, không biết gia đình mình có vấn đề gì cần em hỗ trợ thêm lúc này không ạ?"
    ],

    # --- DENY: Phụ huynh không đồng ý ---
    'DENY': [
        "Dạ em xin phép thông báo nhanh là bé có lịch học tại chương trình ây ai cờ lát lúc [class_time] nhưng hiện tại chưa thấy con vào lớp, không biết gia đình mình có đang gặp vấn đề gì cần em hỗ trợ không ạ?",
        "Dạ em gọi để thông báo nhanh với bố mẹ là bé có ca học ở chương trình ây ai cờ lát lúc [class_time] mà em chưa thấy con vào, không biết anh chị có cần em giúp đỡ vướng mắc gì để bé vào lớp không ạ?",
        "Dạ em xin phép báo nhanh thông tin bé có lịch học chương trình ây ai cờ lát vào [class_time] nhưng hiện chưa thấy con tham gia, gia đình mình có đang gặp khó khăn gì cần em hỗ trợ không ạ?",
        "Dạ em chỉ xin phép báo nhanh để gia đình nhắc bé có lịch học tại ây ai cờ lát lúc [class_time] mà hiện chưa thấy con online, không biết anh chị có cần em hỗ trợ vấn đề gì không ạ?",
        "Dạ em xin phép nhắc nhẹ bố mẹ là chương trình ây ai cờ lát của con có lịch lúc [class_time] mà chưa thấy bé vào, không biết bên mình có cần em hỗ trợ gì để con tham gia lớp được thuận lợi không ạ?",
        "Dạ em xin thông tin nhanh về lịch học của con tại ây ai cờ lát lúc [class_time], do em chưa thấy bé có mặt trong lớp nên muốn hỏi xem gia đình có trục trặc gì cần em hỗ trợ ngay không ạ?",
        "Dạ em xin phép thông báo nhanh để anh chị nhắc con vào lớp ây ai cờ lát có lịch lúc [class_time] vì em chưa thấy bé, không biết gia đình mình có vấn đề gì cần em hỗ trợ thêm lúc này không ạ?"
    ],

    # --- SILENT: Không có phản hồi ---
    'SILENT': [
        "Dạ anh chị ơi, bé [student_name] có lịch học lúc [class_time] mà chưa thấy bé vào lớp, anh chị nhắc bé vào học giúp em nhé?",
        "Dạ anh/chị ơi, ca học lúc [class_time] của bé [student_name] đã bắt đầu rồi mà em chưa thấy bé, nhờ anh/chị nhắc con vào lớp ngay giúp em với.",
        "Dạ, hiện tại đã quá giờ vào lớp [class_time] của bạn [student_name] mà giáo viên vẫn chưa thấy con, phiền anh/chị gọi bé vào học liền giúp em nhé.",
        "Dạ vâng, em thấy đến lịch [class_time] rồi nhưng bé [student_name] vẫn chưa có mặt trong phòng học, gia đình mình nhắc bé vào sớm để kịp bài giúp em.",
        "Dạ thưa anh/chị, lớp học [class_time] đang diễn ra nhưng bé [student_name] chưa đăng nhập, anh/chị hỗ trợ nhắc con vào học giúp em để không bị muộn quá.",
        "Dạ, giờ học [class_time] của bé [student_name] đã điểm rồi mà lớp vẫn vắng bé, anh/chị xem và bảo con vào học ngay giúp em nhé."
    ],

    # --- SILENT: Không có phản hồi ---
    'UNCLEAR': [
        "Dạ anh chị ơi, bé [student_name] có lịch học lúc [class_time] mà chưa thấy bé vào lớp, anh chị nhắc bé vào học giúp em nhé?",
        "Dạ anh/chị ơi, ca học lúc [class_time] của bé [student_name] đã bắt đầu rồi mà em chưa thấy bé, nhờ anh/chị nhắc con vào lớp ngay giúp em với.",
        "Dạ, hiện tại đã quá giờ vào lớp [class_time] của bạn [student_name] mà giáo viên vẫn chưa thấy con, phiền anh/chị gọi bé vào học liền giúp em nhé.",
        "Dạ vâng, em thấy đến lịch [class_time] rồi nhưng bé [student_name] vẫn chưa có mặt trong phòng học, gia đình mình nhắc bé vào sớm để kịp bài giúp em.",
        "Dạ thưa anh/chị, lớp học [class_time] đang diễn ra nhưng bé [student_name] chưa đăng nhập, anh/chị hỗ trợ nhắc con vào học giúp em để không bị muộn quá.",
        "Dạ, giờ học [class_time] của bé [student_name] đã điểm rồi mà lớp vẫn vắng bé, anh/chị xem và bảo con vào học ngay giúp em nhé."
    ],

    # --- FALLBACK: Không hiểu ý định ---
    'FALLBACK': [
        "Dạ em chưa nghe rõ. Anh chị có thể cho em biết bé có cần hỗ trợ gì không?",
        "Dạ, tín hiệu bên em hơi chập chờn nên em nghe chưa rõ. Anh/chị có thể nói lại giúp em là bé có cần hỗ trợ gì không?",
        "Dạ xin lỗi anh/chị, em chưa nghe rõ câu trả lời vừa rồi. Không biết hiện tại gia đình mình có cần em hỗ trợ kỹ thuật hay vấn đề gì cho bé không?",
        "Dạ, em nghe tín hiệu chưa được rõ lắm. Anh/chị vui lòng cho em hỏi lại là mình có cần em giúp đỡ gì để bé vào lớp được không?",
        "Dạ, em chưa nghe rõ thông tin mình vừa trao đổi. Anh/chị có thể cho em biết cụ thể là bé có đang gặp khó khăn gì cần em hỗ trợ không?",
        "Dạ vâng, đường truyền có vẻ không ổn định nên em chưa nghe rõ. Anh/chị có thể xác nhận lại giúp em là mình có cần hỗ trợ gì ngay lúc này không?"
    ]
}

# =============================================================================
#                           RESPONSE BY STATE - INTENT - CONFIRM CUSTOMER
# =============================================================================
# Thoại khi vào state CONFIRM_CUSTOMER theo intent - 5 variations mỗi intent
state_confirm_customer_intents = {

    # --- SILENT: Không có phản hồi ---
    'SILENT': [
        "Dạ, nhà mình đang có bạn tham gia chương trình [program_name] đúng không?",
        "Dạ cho em hỏi thăm, gia đình mình có bé đang theo học tại chương trình [program_name] phải không?",
        "Dạ vâng, em xin phép xác nhận thông tin chút, có phải nhà mình đang cho bé tham gia khóa học ở [program_name] không?",
        "Dạ, anh/chị xác nhận giúp em là bé nhà mình đang là học viên của chương trình [program_name] đúng không?",
        "Dạ, em liên hệ để xác nhận là gia đình đang có bạn nhỏ học tại [program_name] phải không anh/chị?",
        "Dạ vâng, cho em hỏi là nhà mình có bé đang tham gia lớp học của [program_name] đúng không?"
    ],

    # --- SILENT_AGAIN: Không có phản hồi lần 2 ---
    'SILENT_AGAIN': [
        "Dạ anh chị ơi, anh chị xác nhận giúp em nhà mình có bé đang học [program_name] không?",
        "Dạ vâng, phiền anh/chị xác nhận lại giúp em xem có phải bé nhà mình đang tham gia khóa học [program_name] không?",
        "Dạ, để tiện trao đổi, anh/chị cho em hỏi xác nhận là gia đình mình có bé đang theo học tại [program_name] đúng không?",
        "Dạ thưa anh/chị, cho em hỏi lại là nhà mình có con đang học ở chương trình [program_name] không?",
        "Dạ, anh/chị cho em hỏi là nhà mình đang có con học [program_name] phải không?",
        "Dạ anh/chị ơi, anh/chị xem giúp em là hiện tại bé nhà mình có đang theo lớp của [program_name] không?"
    ],

    # --- FALLBACK: Không hiểu ý định ---
    'FALLBACK': [
        "Dạ anh chị cho em hỏi nhà mình có bạn tham gia chương trình [program_name] đúng không?",
        "Dạ, em xin phép xác nhận lại thông tin là gia đình mình có bé đang theo học chương trình [program_name] phải không?",
        "Dạ anh/chị cho em hỏi thăm một chút, có phải nhà mình đang có bạn nhỏ tham gia khóa học tại [program_name] không?",
        "Dạ, để tiện hỗ trợ, anh/chị xác nhận giúp em là hiện tại bé nhà mình đang học ở chương trình [program_name] đúng không?",
        "Dạ vâng, cho em hỏi là gia đình mình đang có con em tham gia lớp học của [program_name] phải không anh/chị?",
        "Dạ, phiền anh/chị cho em biết là nhà mình có đang cho bé theo học tại [program_name] không?"
    ],

    # --- UNCLEAR: Nói không rõ ---
    'UNCLEAR': [
        "Dạ, nhà mình đang có bạn tham gia chương trình [program_name] đúng không?",
        "Dạ cho em hỏi thăm, gia đình mình có bé đang theo học tại chương trình [program_name] phải không?",
        "Dạ vâng, em xin phép xác nhận thông tin chút, có phải nhà mình đang cho bé tham gia khóa học ở [program_name] không?",
        "Dạ, anh/chị xác nhận giúp em là bé nhà mình đang là học viên của chương trình [program_name] đúng không?",
        "Dạ, em liên hệ để xác nhận là gia đình đang có bạn nhỏ học tại [program_name] phải không anh/chị?",
        "Dạ vâng, cho em hỏi là nhà mình có bé đang tham gia lớp học của [program_name] đúng không?"
    ],

    # --- UNCLEAR_AGAIN: Nói không rõ lần 2 ---
    'UNCLEAR_AGAIN': [
        "Dạ anh chị ơi, anh chị xác nhận giúp em nhà mình có bé đang học [program_name] không?",
        "Dạ vâng, phiền anh/chị xác nhận lại giúp em xem có phải bé nhà mình đang tham gia khóa học [program_name] không?",
        "Dạ, để tiện trao đổi, anh/chị cho em hỏi xác nhận là gia đình mình có bé đang theo học tại [program_name] đúng không?",
        "Dạ thưa anh/chị, cho em hỏi lại là nhà mình có con đang học ở chương trình [program_name] không?",
        "Dạ, anh/chị cho em hỏi là nhà mình đang có con học [program_name] phải không?",
        "Dạ anh/chị ơi, anh/chị xem giúp em là hiện tại bé nhà mình có đang theo lớp của [program_name] không?"
    ],

}

# =============================================================================
#                           RESPONSE BY STATE - INTENT - ASK PHONE
# =============================================================================
# Thoại khi vào state ASK_PHONE theo intent - 5 variations mỗi intent
state_ask_phone_intents = {

    # --- SILENT: Không có phản hồi ---
    'SILENT': [
        "Dạ anh chị ơi, anh chị cho em xin số điện thoại để em liên hệ phụ huynh của bé.",
        "Dạ em chưa nghe thấy anh chị. Mình đọc số điện thoại giúp em nhé?",
        "Dạ anh chị vẫn đang nghe em? Mình cho em xin số liên lạc của phụ huynh bé.",
        "Dạ anh chị cho em hỏi, em có thể liên hệ phụ huynh bé qua số nào?",
        "Dạ anh chị cung cấp số điện thoại để em ghi nhận giúp em nhé?",
    ],

    # --- OTHER_NUMBER: Số khác ---
    'OTHER_NUMBER': [
        "Dạ vậy mình đọc giúp em số điện thoại để em liên hệ.",
        "Dạ vâng, anh chị cho em xin số điện thoại của phụ huynh bé.",
        "Dạ anh chị đọc số điện thoại để em ghi nhận giúp em nhé?",
        "Dạ em hiểu. Anh chị vui lòng cung cấp số liên lạc của phụ huynh bé.",
        "Dạ vâng, anh chị cho em biết số điện thoại để em liên hệ nhé?",
    ],

    # --- INVALID_PHONE: Số không hợp lệ ---
    'INVALID_PHONE': [
        "Dạ em chưa rõ, anh chị đọc lại giúp em số điện thoại.",
        "Dạ anh chị đọc lại số điện thoại giúp em được không? Em chưa ghi nhận được.",
        "Dạ em xin lỗi, em chưa nghe rõ số điện thoại. Anh chị nhắc lại giúp em nhé?",
        "Dạ anh chị vui lòng đọc lại số điện thoại chậm hơn giúp em.",
        "Dạ số điện thoại anh chị vừa đọc em chưa nghe rõ. Mình đọc lại giúp em nhé?",
    ],

    # --- FALLBACK: Không hiểu ý định ---
    'FALLBACK': [
        "Dạ em chưa nghe rõ. Em có thể liên hệ tới phụ huynh của bé qua số điện thoại này hay số khác?",
        "Dạ anh chị nói lại giúp em được không? Em cần số điện thoại để liên hệ phụ huynh bé.",
        "Dạ em chưa hiểu ý anh chị. Mình cho em xin số liên lạc của phụ huynh bé nhé?",
        "Dạ anh chị vui lòng cho em biết số điện thoại để em liên hệ phụ huynh.",
        "Dạ em xin lỗi, anh chị cung cấp số điện thoại của phụ huynh bé giúp em nhé?",
    ],
}


# =============================================================================
#                           RESPONSE BY STATE - INTENT - TALK WITH STUDENT
# =============================================================================
# Thoại khi vào state TALK_WITH_STUDENT theo intent - 5 variations mỗi intent
state_talk_with_student_intents = {

    # --- NEED_SUPPORT: Học sinh cần hỗ trợ ---
    'NEED_SUPPORT': [
        "Vậy con đang cần hỗ trợ vấn đề gì mà chưa vào học được nhỉ?",
        "Con có gặp khó khăn gì mà chưa vào lớp được không nè?",
        "Con cho cô biết con đang gặp vấn đề gì để cô hỗ trợ nha.",
        "Con cần cô giúp đỡ việc gì để vào học không nè?",
        "Vậy con có vấn đề gì cần cô hỗ trợ để vào lớp nhỉ?",
    ],

    # --- SILENT: Không có phản hồi ---
    'SILENT': [
        "Con ơi, hôm nay con có lịch học vào lúc [class_time] mà cô chưa thấy con vào lớp, không biết con có vấn đề gì cần cô hỗ trợ không nha?",
        "Con ơi, cô chưa nghe thấy con nói gì. Con có buổi học lúc [class_time] đang chờ con đó, con vào học nha.",
        "Con vẫn đang nghe cô chứ? Lớp học lúc [class_time] của con đã bắt đầu rồi đó.",
        "Con ơi, con có nghe cô không? Con có lịch học lúc [class_time] mà cô chưa thấy con vào lớp.",
        "Cô chưa nghe thấy con trả lời. Con có cần cô hỗ trợ gì để vào lớp học lúc [class_time] không?",
    ],

    # --- FALLBACK: Không hiểu ý định ---
    'FALLBACK': [
        "Cô chưa rõ, không biết con có vấn đề gì cần cô hỗ trợ để vào học không nha.",
        "Cô chưa hiểu ý con nè. Con có gặp khó khăn gì mà chưa vào lớp được không?",
        "Con nói lại giúp cô được không nha? Con có cần cô hỗ trợ gì không?",
        "Cô chưa nghe rõ con nói gì nè. Con có vấn đề gì cần cô giúp không nha?",
        "Con ơi, cô chưa hiểu. Con có cần hỗ trợ gì để vào lớp học không nè?",
    ],

    # --- UNCLEAR: Nói không rõ ---
    'UNCLEAR': [
        "Con ơi, hôm nay con có lịch học vào lúc [class_time] mà cô chưa thấy con vào lớp, không biết con có vấn đề gì cần cô hỗ trợ không nha?",
        "Con ơi, cô chưa nghe thấy con nói gì. Con có buổi học lúc [class_time] đang chờ con đó, con vào học nha.",
        "Con vẫn đang nghe cô chứ? Lớp học lúc [class_time] của con đã bắt đầu rồi đó.",
        "Con ơi, con có nghe cô không? Con có lịch học lúc [class_time] mà cô chưa thấy con vào lớp.",
        "Cô chưa nghe thấy con trả lời. Con có cần cô hỗ trợ gì để vào lớp học lúc [class_time] không?",
    ],

}


# =============================================================================
#                           RESPONSE BY STATE - INTENT - FINISH
# =============================================================================
# Thoại finish từ state + intent - 5 variations mỗi intent
state_finish_from_state_intents = {
    "REMINDER": {
        "STUDENT_JOINING": [
            "Dạ vâng, vậy anh chị giúp em cho con vào học kịp thời nhé. Nếu cần hỗ trợ vấn đề gì thì anh chị liên hệ qua zalo ban giáo vụ hoặc số hót nai[hotline] để được hỗ trợ. Em chào anh chị.",
            "Dạ em cảm ơn anh chị. Anh chị nhắc bé vào lớp học ngay giúp em nhé. Nếu cần hỗ trợ thêm anh chị liên hệ hót nai[hotline]. Em chào anh chị.",
            "Dạ vâng, anh chị cho bé vào học giúp em nhé. Có vấn đề gì cần hỗ trợ anh chị gọi hót nai[hotline] để được giải đáp. Em chào anh chị.",
            "Dạ cảm ơn anh chị đã phản hồi. Anh chị giúp em nhắc bé tham gia lớp học nhé. Chúc anh chị và bé buổi học vui vẻ. Em chào anh chị.",
            "Dạ vâng, em ghi nhận. Anh chị hỗ trợ bé vào lớp học giúp em nhé. Nếu cần gì anh chị liên hệ hót nai[hotline]. Em chào anh chị.",
        ],
        "ALREADY_ABSENT": [
            "Dạ em xin ghi nhận thông tin từ anh chị và trao đổi lại với ban giáo vụ. Em xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ vâng, em đã ghi nhận thông tin. Em sẽ báo lại với ban giáo vụ để cập nhật. Em xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ em hiểu. Em sẽ thông báo lại với giáo viên chủ nhiệm của bé. Cảm ơn anh chị đã phản hồi. Em chào anh chị.",
            "Dạ em xin ghi nhận. Em sẽ báo lại với ban giáo vụ để nắm thông tin. Em xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ vâng, em đã hiểu. Em sẽ trao đổi lại với ban giáo vụ về việc này. Cảm ơn anh chị. Em chào anh chị.",
        ],
        "REQUEST_ABSENT": [
            "Dạ em xin ghi nhận thông tin từ anh chị và báo lại với ban giáo vụ. Em chào anh chị.",
            "Dạ vâng, em đã ghi nhận yêu cầu của anh chị. Em sẽ báo lại với giáo viên. Em chào anh chị.",
            "Dạ em hiểu. Em sẽ thông báo lại với ban giáo vụ để cập nhật lịch học của bé. Em chào anh chị.",
            "Dạ em ghi nhận. Bé nhớ xem lại bài học trong record để nắm bài nhé. Em chào anh chị.",
            "Dạ vâng, em sẽ báo lại với giáo viên chủ nhiệm của bé. Cảm ơn anh chị đã thông báo. Em chào anh chị.",
        ],
        "CANCEL_COURSE": [
            "Dạ em xin lỗi đã làm phiền gia đình, em sẽ báo lại với ban giáo vụ để kiểm tra. Em chào anh chị.",
            "Dạ vâng, em ghi nhận thông tin. Em sẽ trao đổi lại với ban giáo vụ để xác nhận. Em chào anh chị.",
            "Dạ em hiểu. Em sẽ báo lại với bộ phận liên quan để kiểm tra thông tin. Em chào anh chị.",
            "Dạ em xin lỗi đã làm phiền anh chị. Em sẽ cập nhật lại với ban giáo vụ. Em chào anh chị.",
            "Dạ vâng, em ghi nhận và sẽ báo lại với ban giáo vụ để xác minh. Cảm ơn anh chị. Em chào anh chị.",
        ],
        "BUSY": [
            "Dạ vậy em xin phép liên hệ lại vào khi khác. Em chào anh chị.",
            "Dạ em hiểu. Em sẽ gọi lại anh chị vào thời điểm khác. Em chào anh chị.",
            "Dạ vâng, em xin phép liên hệ lại sau. Cảm ơn anh chị đã nghe máy. Em chào anh chị.",
            "Dạ em ghi nhận. Khi nào tiện anh chị nhắc bé vào học giúp em nhé. Em chào anh chị.",
            "Dạ không sao. Em sẽ liên hệ lại sau. Chúc anh chị buổi tối vui vẻ. Em chào anh chị.",
            "Dạ em hiểu. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ vâng, em sẽ liên hệ lại sau. Cảm ơn anh chị đã nghe máy. Em chào anh chị.",
            "Dạ em ghi nhận. Em xin phép gọi lại lúc khác. Em chào anh chị.",
            "Dạ không sao. Em sẽ liên hệ lại khi anh chị tiện. Em chào anh chị.",
        ],
        "DENY": [
            "Dạ vậy em xin phép liên hệ lại vào khi khác. Em chào anh chị.",
            "Dạ em hiểu. Em sẽ gọi lại anh chị vào thời điểm khác. Em chào anh chị.",
            "Dạ vâng, em xin phép liên hệ lại sau. Cảm ơn anh chị đã nghe máy. Em chào anh chị.",
            "Dạ em ghi nhận. Khi nào tiện anh chị nhắc bé vào học giúp em nhé. Em chào anh chị.",
            "Dạ không sao. Em sẽ liên hệ lại sau. Chúc anh chị buổi tối vui vẻ. Em chào anh chị.",
            "Dạ em hiểu. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ vâng, em sẽ liên hệ lại sau. Cảm ơn anh chị đã nghe máy. Em chào anh chị.",
            "Dạ em ghi nhận. Em xin phép gọi lại lúc khác. Em chào anh chị.",
            "Dạ không sao. Em sẽ liên hệ lại khi anh chị tiện. Em chào anh chị.",
        ],
        "SILENT": [
            "Dạ do đường truyền không tốt, em xin phép gọi lại trao đổi với nhà mình. Em chào anh chị.",
            "Dạ em chưa nghe rõ phản hồi của anh chị. Em sẽ gọi lại sau. Em chào anh chị.",
            "Dạ có vẻ đường truyền không ổn định. Em xin phép liên hệ lại sau. Em chào anh chị.",
            "Dạ em không nghe thấy anh chị phản hồi. Em sẽ gọi lại vào lúc khác. Em chào anh chị.",
            "Dạ đường truyền có vẻ không tốt. Em xin phép gọi lại để trao đổi với anh chị. Em chào anh chị.",
            "Dạ em chưa nghe rõ. Em sẽ liên hệ lại sau để trao đổi với anh chị. Em chào anh chị.",
            "Dạ có vẻ đường truyền bị nhiễu. Em xin phép gọi lại vào lúc khác. Em chào anh chị.",
            "Dạ em không nghe rõ phản hồi. Em sẽ gọi lại để trao đổi với anh chị. Em chào anh chị.",
            "Dạ đường truyền không ổn định. Em xin phép liên hệ lại nhà mình sau. Em chào anh chị.",
        ],
        "UNCLEAR": [
            "Dạ do đường truyền không tốt, em xin phép gọi lại trao đổi với nhà mình. Em chào anh chị.",
            "Dạ em chưa nghe rõ phản hồi của anh chị. Em sẽ gọi lại sau. Em chào anh chị.",
            "Dạ có vẻ đường truyền không ổn định. Em xin phép liên hệ lại sau. Em chào anh chị.",
            "Dạ em không nghe thấy anh chị phản hồi. Em sẽ gọi lại vào lúc khác. Em chào anh chị.",
            "Dạ đường truyền có vẻ không tốt. Em xin phép gọi lại để trao đổi với anh chị. Em chào anh chị.",
            "Dạ em chưa nghe rõ. Em sẽ liên hệ lại sau để trao đổi với anh chị. Em chào anh chị.",
            "Dạ có vẻ đường truyền bị nhiễu. Em xin phép gọi lại vào lúc khác. Em chào anh chị.",
            "Dạ em không nghe rõ phản hồi. Em sẽ gọi lại để trao đổi với anh chị. Em chào anh chị.",
            "Dạ đường truyền không ổn định. Em xin phép liên hệ lại nhà mình sau. Em chào anh chị.",
        ],
        "FALLBACK": [
            "Dạ, em chưa rõ phản hồi của anh chị lắm, em xin gọi trao đổi lại với nhà mình sau. Em chào anh chị.",
            "Dạ em chưa hiểu ý anh chị. Em sẽ liên hệ lại sau để trao đổi rõ hơn. Em chào anh chị.",
            "Dạ em chưa nắm được ý anh chị. Em xin phép gọi lại lúc khác. Em chào anh chị.",
            "Dạ em chưa nghe rõ. Em sẽ gọi lại để trao đổi với anh chị sau. Em chào anh chị.",
            "Dạ em xin lỗi, em chưa hiểu. Em sẽ liên hệ lại nhà mình vào lúc khác. Em chào anh chị.",
        ],
    },
    "CONFIRM_CUSTOMER": {
        "WRONG_NUMBER": [
            "Dạ vâng, xin lỗi đã làm phiền nhà mình, bên em sẽ kiểm tra lại. Em chào anh chị.",
            "Dạ em xin lỗi đã làm phiền anh chị. Em sẽ cập nhật lại thông tin. Em chào anh chị.",
            "Dạ vâng, em ghi nhận. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ em hiểu. Bên em sẽ kiểm tra lại số điện thoại. Xin lỗi anh chị. Em chào anh chị.",
            "Dạ xin lỗi anh chị. Em sẽ cập nhật lại thông tin liên hệ. Em chào anh chị.",
        ],
        "BUSY": [
            "Dạ em xin lỗi đã làm phiền anh chị, em xin phép liên hệ lại vào khi khác. Em chào anh chị.",
            "Dạ em hiểu. Em sẽ gọi lại anh chị vào thời điểm khác. Em chào anh chị.",
            "Dạ vâng, em xin phép liên hệ lại sau. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ không sao. Em sẽ liên hệ lại khi anh chị tiện. Em chào anh chị.",
            "Dạ em ghi nhận. Em xin phép gọi lại lúc khác. Em chào anh chị.",
            "Dạ vâng, em hiểu. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ em ghi nhận. Em sẽ liên hệ lại sau. Em chào anh chị.",
            "Dạ không sao. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ em hiểu. Em xin phép gọi lại lúc khác. Em chào anh chị.",
        ],
        "DENY": [
            "Dạ em xin lỗi đã làm phiền anh chị, em xin phép liên hệ lại vào khi khác. Em chào anh chị.",
            "Dạ em hiểu. Em sẽ gọi lại anh chị vào thời điểm khác. Em chào anh chị.",
            "Dạ vâng, em xin phép liên hệ lại sau. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ không sao. Em sẽ liên hệ lại khi anh chị tiện. Em chào anh chị.",
            "Dạ em ghi nhận. Em xin phép gọi lại lúc khác. Em chào anh chị.",
            "Dạ vâng, em hiểu. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ em ghi nhận. Em sẽ liên hệ lại sau. Em chào anh chị.",
            "Dạ không sao. Xin lỗi đã làm phiền anh chị. Em chào anh chị.",
            "Dạ em hiểu. Em xin phép gọi lại lúc khác. Em chào anh chị.",
        ],
        "SILENT": [
            "Dạ do đường truyền không tốt, em xin phép gọi lại trao đổi với nhà mình. Em chào anh chị.",
            "Dạ em chưa nghe rõ. Em sẽ liên hệ lại sau. Em chào anh chị.",
            "Dạ có vẻ đường truyền không ổn định. Em xin phép gọi lại. Em chào anh chị.",
            "Dạ em không nghe rõ phản hồi. Em sẽ gọi lại vào lúc khác. Em chào anh chị.",
            "Dạ đường truyền có vẻ bị nhiễu. Em xin phép liên hệ lại nhà mình. Em chào anh chị.",
            "Dạ em không nghe thấy phản hồi. Em sẽ liên hệ lại sau. Em chào anh chị.",
            "Dạ có vẻ đường truyền không ổn định. Em xin phép gọi lại. Em chào anh chị.",
            "Dạ em chưa nghe thấy anh chị. Em sẽ gọi lại vào lúc khác. Em chào anh chị.",
            "Dạ đường truyền có vẻ bị nhiễu. Em xin phép liên hệ lại. Em chào anh chị.",
        ],
        "UNCLEAR": [
            "Dạ do đường truyền không tốt, em xin phép gọi lại trao đổi với nhà mình. Em chào anh chị.",
            "Dạ em chưa nghe rõ. Em sẽ liên hệ lại sau. Em chào anh chị.",
            "Dạ có vẻ đường truyền không ổn định. Em xin phép gọi lại. Em chào anh chị.",
            "Dạ em không nghe rõ phản hồi. Em sẽ gọi lại vào lúc khác. Em chào anh chị.",
            "Dạ đường truyền có vẻ bị nhiễu. Em xin phép liên hệ lại nhà mình. Em chào anh chị.",
            "Dạ em không nghe thấy phản hồi. Em sẽ liên hệ lại sau. Em chào anh chị.",
            "Dạ có vẻ đường truyền không ổn định. Em xin phép gọi lại. Em chào anh chị.",
            "Dạ em chưa nghe thấy anh chị. Em sẽ gọi lại vào lúc khác. Em chào anh chị.",
            "Dạ đường truyền có vẻ bị nhiễu. Em xin phép liên hệ lại. Em chào anh chị.",
        ],
        "FALLBACK": [
            "Dạ, em chưa rõ phản hồi của anh chị lắm, em xin gọi trao đổi lại với nhà mình sau. Em chào anh chị.",
            "Dạ em chưa hiểu ý anh chị. Em sẽ liên hệ lại sau. Em chào anh chị.",
            "Dạ em chưa nắm được ý anh chị. Em xin phép gọi lại lúc khác. Em chào anh chị.",
            "Dạ em chưa nghe rõ. Em sẽ gọi lại để trao đổi với anh chị. Em chào anh chị.",
            "Dạ em xin lỗi, em chưa hiểu. Em sẽ liên hệ lại nhà mình sau. Em chào anh chị.",
        ],
    },
    "ASK_PHONE": {
        "READ_PHONE": [
            "Dạ em cảm ơn anh chị. Em chào anh chị.",
            "Dạ vâng, em đã ghi nhận số điện thoại. Cảm ơn anh chị. Em chào anh chị.",
            "Dạ em cảm ơn anh chị đã cung cấp thông tin. Em chào anh chị.",
            "Dạ em đã ghi nhận. Cảm ơn anh chị đã hỗ trợ. Em chào anh chị.",
            "Dạ vâng, em cảm ơn anh chị. Chúc anh chị buổi tối vui vẻ. Em chào anh chị.",
        ],
        "OTHER_NUMBER": [
            "Dạ em cảm ơn anh chị đã phản hồi. Em chào anh chị.",
            "Dạ vâng, em ghi nhận thông tin. Cảm ơn anh chị. Em chào anh chị.",
            "Dạ em cảm ơn anh chị. Em sẽ liên hệ lại theo số đó. Em chào anh chị.",
            "Dạ em đã ghi nhận. Cảm ơn anh chị đã cung cấp thông tin. Em chào anh chị.",
            "Dạ vâng, em cảm ơn anh chị đã hỗ trợ. Em chào anh chị.",
        ],
        "INVALID_PHONE": [
            "Dạ em cảm ơn anh chị. Em chào anh chị.",
            "Dạ vâng, em ghi nhận. Cảm ơn anh chị đã phản hồi. Em chào anh chị.",
            "Dạ em cảm ơn anh chị đã hỗ trợ. Em chào anh chị.",
            "Dạ em đã ghi nhận thông tin. Cảm ơn anh chị. Em chào anh chị.",
            "Dạ vâng, cảm ơn anh chị đã nghe máy. Em chào anh chị.",
        ],
        "FALLBACK": [
            "Dạ em cảm ơn anh chị đã phản hồi. Em chào anh chị.",
            "Dạ vâng, em ghi nhận. Cảm ơn anh chị. Em chào anh chị.",
            "Dạ em cảm ơn anh chị đã nghe máy. Em chào anh chị.",
            "Dạ em đã ghi nhận thông tin. Cảm ơn anh chị đã hỗ trợ. Em chào anh chị.",
            "Dạ vâng, cảm ơn anh chị. Chúc anh chị buổi tối vui vẻ. Em chào anh chị.",
        ],
    },
    "TALK_WITH_STUDENT": {
        "STUDENT_JOINING": [
            "Vậy con vào lớp học nha. Nếu có vấn đề gì không vào được con nhờ bố mẹ hỗ trợ con nhé. Cô chào con.",
            "Con vào học ngay nha. Nếu cần gì con liên hệ giáo viên chủ nhiệm hoặc nhờ bố mẹ hỗ trợ nhé. Cô chào con.",
            "Vậy con tham gia lớp học luôn nha. Có vấn đề gì con nhờ bố mẹ giúp con nhé. Cô chào con.",
            "Con đăng nhập vào lớp học ngay giúp cô nha. Chúc con buổi học vui vẻ. Cô chào con.",
            "Vậy con vào học nha. Nếu gặp khó khăn gì con báo bố mẹ hoặc liên hệ cô giáo nhé. Cô chào con.",
        ],
        "ALREADY_ABSENT": [
            "Vậy để cô báo lại với gia sư của con nha. Con nhớ xem lại record buổi học hôm nay để nắm được bài học nha. Cô chào con.",
            "Cô đã ghi nhận nha. Con nhớ xem lại video bài giảng để không bỏ lỡ kiến thức nhé. Cô chào con.",
            "Vậy cô sẽ báo lại với giáo viên nha. Con nhớ ôn lại bài qua record buổi học nhé. Cô chào con.",
            "Cô hiểu rồi nha. Con nhớ xem lại bài học trong record để nắm bài nhé. Cô chào con.",
            "Vậy cô ghi nhận nha. Con nhớ xem lại video buổi học hôm nay để theo kịp bài nhé. Cô chào con.",
        ],
        "REQUEST_ABSENT": [
            "Vậy để cô báo lại với gia sư của con nha. Con nhớ xem lại record buổi học hôm nay để nắm được bài học nha. Cô chào con.",
            "Cô đã ghi nhận nha. Con nhớ xem lại video bài giảng để không bỏ lỡ kiến thức nhé. Cô chào con.",
            "Vậy cô sẽ thông báo lại với giáo viên nha. Con nhớ ôn lại bài qua record nhé. Cô chào con.",
            "Cô hiểu rồi nha. Con nhớ xem lại buổi học trong record để nắm bài nhé. Cô chào con.",
            "Vậy cô ghi nhận nha. Con nhớ xem lại video buổi học để theo kịp bài nhé. Cô chào con.",
        ],
        "SILENT": [
            "Do đường truyền không tốt, cô sẽ gọi lại cho con sau nha. Cô chào con nha.",
            "Cô chưa nghe thấy con nói gì nè. Cô sẽ liên hệ lại sau nha. Cô chào con.",
            "Có vẻ đường truyền không ổn định nha. Cô sẽ gọi lại cho con lúc khác. Cô chào con.",
            "Cô không nghe thấy con phản hồi nè. Cô sẽ gọi lại sau nha. Cô chào con.",
            "Đường truyền có vẻ bị nhiễu nha. Cô xin phép gọi lại cho con sau. Cô chào con.",
        ],
        "UNCLEAR": [
            "Do đường truyền không tốt, cô sẽ gọi lại cho con sau nha. Cô chào con nha.",
            "Cô chưa nghe rõ con nói gì nè. Cô sẽ liên hệ lại sau nha. Cô chào con.",
            "Có vẻ đường truyền không ổn định nha. Cô sẽ gọi lại cho con lúc khác. Cô chào con.",
            "Cô không nghe rõ nè. Cô sẽ gọi lại sau nha. Cô chào con.",
            "Đường truyền có vẻ bị nhiễu nha. Cô xin phép gọi lại cho con sau. Cô chào con.",
        ],
    },
}


# =============================================================================
#                           RESPONSE BY STATE - INTENT - TRANSFER HOTLINE
# =============================================================================
# Thoại transfer hót naitừ state + intent - 5 variations mỗi intent
state_transfer_hotline_from_state_intents = {
    "REMINDER": {
        'NEED_SUPPORT': [
            "Dạ, vậy anh chị chờ trong giây lát em sẽ kết nối anh chị với ban giáo vụ để được hỗ trợ.",
            "Dạ vâng, anh chị giữ máy giúp em một chút, em sẽ chuyển anh chị sang ban giáo vụ để được hỗ trợ.",
            "Dạ em hiểu. Anh chị chờ em một chút, em sẽ kết nối anh chị với bộ phận hỗ trợ.",
            "Dạ vâng, anh chị đợi em giây lát, em sẽ chuyển cuộc gọi sang ban giáo vụ để hỗ trợ anh chị.",
            "Dạ anh chị giữ máy giúp em nha, em sẽ kết nối với ban giáo vụ để hỗ trợ gia đình mình.",
        ],
        "WRONG_SCHEDULE": [
            "Dạ, vậy anh chị chờ trong giây lát em sẽ kết nối anh chị với ban giáo vụ để anh chị trao đổi lại lịch học của con nhé.",
            "Dạ vâng, anh chị giữ máy giúp em, em sẽ chuyển sang ban giáo vụ để kiểm tra lại lịch học của bé.",
            "Dạ em hiểu. Anh chị chờ em một chút, em sẽ kết nối với ban giáo vụ để xác nhận lại lịch học.",
            "Dạ vâng, anh chị đợi em giây lát, em sẽ chuyển cuộc gọi để anh chị trao đổi về lịch học của bé.",
            "Dạ anh chị giữ máy giúp em nha, em sẽ kết nối với ban giáo vụ để xem lại lịch học của con.",
        ],
        "TECHNICAL_ISSUE": [
            "Dạ, vậy anh chị chờ trong giây lát em sẽ kết nối anh chị với ban giáo vụ để được hỗ trợ kỹ thuật.",
            "Dạ vâng, anh chị giữ máy giúp em, em sẽ chuyển sang bộ phận kỹ thuật để hỗ trợ anh chị.",
            "Dạ em hiểu. Anh chị chờ em một chút, em sẽ kết nối với ban giáo vụ để xử lý vấn đề kỹ thuật.",
            "Dạ vâng, anh chị đợi em giây lát, em sẽ chuyển cuộc gọi để hỗ trợ anh chị về vấn đề kỹ thuật.",
            "Dạ anh chị giữ máy giúp em nha, em sẽ kết nối với bộ phận hỗ trợ để giải quyết vấn đề cho gia đình mình.",
        ],
        "ASK_ABOUT_PROGRAM": [
            "Dạ, vậy anh chị chờ trong giây lát em sẽ kết nối anh chị với ban giáo vụ để được tư vấn thêm về chương trình nhé.",
            "Dạ vâng, anh chị giữ máy giúp em, em sẽ chuyển anh chị sang ban giáo vụ để tư vấn chi tiết hơn.",
            "Dạ em hiểu. Anh chị chờ em một chút, em sẽ kết nối với ban giáo vụ để anh chị được tư vấn.",
            "Dạ vâng, anh chị đợi em giây lát, em sẽ chuyển cuộc gọi để anh chị trao đổi thêm về chương trình.",
            "Dạ anh chị giữ máy giúp em nha, em sẽ kết nối với ban giáo vụ để hỗ trợ anh chị tìm hiểu thêm.",
        ],
    },
    "TALK_WITH_STUDENT": {
        'NEED_SUPPORT': [
            "Con giữ máy giúp cô một chút nha, cô sẽ kết nối con với ban giáo vụ để các thầy cô hỗ trợ con vào lớp nha.",
            "Con chờ cô một chút nha, cô sẽ chuyển con sang nói chuyện với thầy cô giáo vụ để được hỗ trợ nha.",
            "Con đợi cô giây lát nha, cô sẽ kết nối con với ban giáo vụ để hỗ trợ con nha.",
            "Con giữ máy nha, cô sẽ chuyển cuộc gọi để thầy cô hỗ trợ con vào lớp học nha.",
            "Con chờ một chút nha, cô sẽ kết nối con với bộ phận hỗ trợ để giúp con nha.",
        ],
        "WRONG_SCHEDULE": [
            "Con chờ chút nha, cô sẽ kết nối con với ban giáo vụ để thầy cô kiểm tra lại lịch học của con nha.",
            "Con giữ máy giúp cô nha, cô sẽ chuyển con sang ban giáo vụ để xác nhận lại lịch học nha.",
            "Con đợi cô một chút nha, cô sẽ kết nối để thầy cô kiểm tra lịch học cho con nha.",
            "Con chờ cô giây lát nha, cô sẽ chuyển cuộc gọi để thầy cô xem lại lịch học của con nha.",
            "Con giữ máy nha, cô sẽ kết nối con với ban giáo vụ để trao đổi về lịch học nha.",
        ],
        "TECHNICAL_ISSUE": [
            "Con giữ máy giúp cô một chút nha, cô sẽ kết nối con với ban giáo vụ để các thầy cô hỗ trợ con vào lớp nha.",
            "Con chờ cô một chút nha, cô sẽ chuyển con sang bộ phận kỹ thuật để được hỗ trợ nha.",
            "Con đợi cô giây lát nha, cô sẽ kết nối con với thầy cô để xử lý vấn đề kỹ thuật cho con nha.",
            "Con giữ máy nha, cô sẽ chuyển cuộc gọi để bộ phận hỗ trợ giúp con vào lớp học nha.",
            "Con chờ một chút nha, cô sẽ kết nối con với ban giáo vụ để giải quyết vấn đề kỹ thuật nha.",
        ],
        "FALLBACK": [
            "Con giữ máy giúp cô một chút nha, cô sẽ kết nối con với ban giáo vụ để các thầy cô hỗ trợ con nha.",
            "Con chờ cô một chút nha, cô sẽ chuyển con sang nói chuyện với thầy cô giáo vụ để hỗ trợ con nha.",
            "Con đợi cô giây lát nha, cô sẽ kết nối con với ban giáo vụ để thầy cô giúp con nha.",
            "Con giữ máy nha, cô sẽ chuyển cuộc gọi để thầy cô hỗ trợ con nha.",
            "Con chờ một chút nha, cô sẽ kết nối con với bộ phận hỗ trợ để giúp con nha.",
        ],
    },
}


# =============================================================================
#                           COMBINED RESPONSE TEMPLATES
# =============================================================================

# Greeting sentences khi mới vào state (chưa có intent từ user)
STATE_GREETINGS = {
    'REMINDER': agent_state_reminders,
    'CONFIRM_CUSTOMER': agent_state_confirm_customers,
    'ASK_PHONE': agent_state_ask_phone,
    'TALK_WITH_STUDENT': agent_state_talk_with_students,
}

# Response templates theo state -> intent -> list of responses
RESPONSE_TEMPLATES = {
    'REMINDER': state_reminder_intents,
    'CONFIRM_CUSTOMER': state_confirm_customer_intents,
    'ASK_PHONE': state_ask_phone_intents,
    'TALK_WITH_STUDENT': state_talk_with_student_intents,
}

# Response cho FINISH (cần from_state): from_state -> intent -> list of responses
FINISH_RESPONSE_TEMPLATES = state_finish_from_state_intents

# Response cho TRANSFER_hót nai(cần from_state): from_state -> intent -> list of responses
TRANSFER_HOTLINE_RESPONSE_TEMPLATES = state_transfer_hotline_from_state_intents


def get_greeting_for_state(state: str, script_data: dict) -> str:
    """
    Get a random greeting sentence when entering a state.

    Args:
        state: State name (REMINDER, CONFIRM_CUSTOMER, ASK_PHONE, TALK_WITH_STUDENT)
        script_data: Script data for placeholder replacement

    Returns:
        Formatted greeting string
    """
    sentences = STATE_GREETINGS.get(state, [])
    return agent_sentence(sentences, script_data)


def get_response_for_state_intent(state: str, intent: str, script_data: dict, state_intent_retry_count: int = 0) -> str:
    """
    Get a random response for given state and intent.

    Args:
        state: Current state name (REMINDER, CONFIRM_CUSTOMER, ASK_PHONE, TALK_WITH_STUDENT)
        intent: Current intent name
        script_data: Script data for placeholder replacement
        retry_count: Number of retries for this intent

    Returns:
        Formatted response string
    """
    if state_intent_retry_count == 0:
        return get_greeting_for_state(state, script_data)
    else:
        state_responses = RESPONSE_TEMPLATES.get(state, {})

        # Try to get retry-specific responses first (e.g., SILENT_AGAIN, UNCLEAR_AGAIN)
        if state_intent_retry_count > 1:
            retry_key = f"{intent}_AGAIN"
            intent_responses = state_responses.get(retry_key)
            if intent_responses:
                return agent_sentence(intent_responses, script_data)

        # Get normal intent responses
        intent_responses = state_responses.get(intent)
        if not intent_responses:
            # Fallback to generic fallback
            intent_responses = state_responses.get('FALLBACK', [
                "Dạ em chưa nghe rõ. Anh chị vui lòng nói lại được không?"
            ])

        return agent_sentence(intent_responses, script_data)


def get_response_for_finish(from_state: str, intent: str, script_data: dict) -> str:
    """
    Get a random finish response based on which state we're finishing from.

    Args:
        from_state: The state we're transitioning from (REMINDER, CONFIRM_CUSTOMER, ASK_PHONE, TALK_WITH_STUDENT)
        intent: The intent that triggered the finish
        script_data: Script data for placeholder replacement

    Returns:
        Formatted finish response string, or empty string if no template found
    """
    state_responses = FINISH_RESPONSE_TEMPLATES.get(from_state, {})
    intent_responses = state_responses.get(intent)

    if not intent_responses:
        # Try FALLBACK intent for this state first
        intent_responses = state_responses.get('FALLBACK')

    if not intent_responses:
        # Still no response - use state-appropriate default fallback
        if from_state == 'TALK_WITH_STUDENT':
            intent_responses = [
                "Cô chào con nha.",
                "Chào con nhé.",
            ]
        else:
            intent_responses = [
                "Dạ em cảm ơn anh chị đã nghe máy. Em chào anh chị.",
                "Dạ vâng, em chào anh chị.",
            ]

    return agent_sentence(intent_responses, script_data)


def get_response_for_transfer_hotline(from_state: str, intent: str, script_data: dict) -> str:
    """
    Get a random transfer hotline response based on which state we're transferring from.
    
    This function also marks that a transfer_hotline event should be triggered.
    The actual event will be sent by the agent after the response is spoken.

    Args:
        from_state: The state we're transitioning from (REMINDER, TALK_WITH_STUDENT)
        intent: The intent that triggered the transfer (NEED_SUPPORT, WRONG_SCHEDULE, TECHNICAL_ISSUE)
        script_data: Script data for placeholder replacement

    Returns:
        Formatted transfer hotline response string
        
    Note:
        The transfer_hotline event will be sent via WebSocket after this response is spoken.
        Event format: {
            'event': 'transfer_hotline',
            'call_id': '<call_id>',
            'transfer_to': '<hotline>',
            'reason': 'customer_support_request',
            'timestamp': <unix_timestamp_ms>,
            'metadata': {...}
        }
    """
    state_responses = TRANSFER_HOTLINE_RESPONSE_TEMPLATES.get(from_state, {})
    intent_responses = state_responses.get(intent)

    if not intent_responses:
        # Try NEED_SUPPORT intent for this state first
        intent_responses = state_responses.get('NEED_SUPPORT')

    if not intent_responses:
        # Still no response - use state-appropriate default fallback
        if from_state == 'TALK_WITH_STUDENT':
            intent_responses = [
                "Con giữ máy giúp cô một chút nha, cô sẽ kết nối con với ban giáo vụ để các thầy cô hỗ trợ con nha.",
                "Con chờ cô một chút nha, cô sẽ chuyển con sang nói chuyện với thầy cô giáo vụ để được hỗ trợ nha.",
            ]
        else:
            intent_responses = [
                "Dạ, vậy anh chị chờ trong giây lát em sẽ kết nối anh chị với ban giáo vụ để được hỗ trợ.",
                "Dạ vâng, anh chị giữ máy giúp em một chút, em sẽ chuyển anh chị sang ban giáo vụ để được hỗ trợ.",
            ]
    
    # Log transfer event info
    hotline = script_data.get('hotline', '0931208686')
    logger.info(f"Transfer hotline response generated. Will transfer to: {hotline}")

    return agent_sentence(intent_responses, script_data)
