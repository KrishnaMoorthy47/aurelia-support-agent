"""Language detection and the fixed (non-model) replies in English and Arabic.

Fixed replies are written by hand, not generated, so refusals and approval messages say exactly
the same thing every time in both languages.
"""
from __future__ import annotations

import re

_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """'ar' if most letters in the message are Arabic, otherwise 'en'."""
    arabic = len(_ARABIC_LETTER.findall(text or ""))
    latin = len(_LATIN_LETTER.findall(text or ""))
    return "ar" if arabic and arabic >= latin else "en"


LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}


def reply_language_line(lang: str) -> str:
    """Appended to model inputs so the reply comes back in the customer's language."""
    if lang == "ar":
        return (
            "\n\nREPLY_LANGUAGE: Arabic (Modern Standard Arabic). Write numbers as Western digits (0-9), "
            "keep currency codes such as USD and AED, keep email addresses as they are, and keep the [n] citation markers."
        )
    return ""


MESSAGES: dict[str, dict[str, str]] = {
    "safe_fallback": {
        "en": "I'm not able to help with that in a way that fits our support policy. "
        "A colleague can follow up if you'd like - just say 'log a ticket'.",
        "ar": "لا أستطيع المساعدة في ذلك بطريقة تتوافق مع سياسة الدعم لدينا. "
        "يمكن لأحد الزملاء متابعة طلبك إذا رغبت - فقط اكتب «افتح تذكرة».",
    },
    "advice_refusal": {
        "en": "I can't give investment advice, trading signals, or price predictions, and nobody at Aurelia support can promise "
        "outcomes - trading CFDs carries a high risk of losing money quickly [1]. I can explain our fees, margin and "
        "leverage rules, verification, or how a demo account works if that helps.",
        "ar": "لا يمكنني تقديم نصائح استثمارية أو إشارات تداول أو توقعات للأسعار، ولا يستطيع أحد في دعم Aurelia أن يعد بأي نتائج - "
        "فتداول عقود الفروقات ينطوي على مخاطر عالية بخسارة الأموال بسرعة [1]. يمكنني أن أشرح الرسوم وقواعد الهامش والرافعة المالية "
        "والتحقق من الهوية، أو كيف يعمل الحساب التجريبي إذا كان ذلك مفيدًا.",
    },
    "out_of_scope": {
        "en": "I can only help with Aurelia Markets support: fees, margin and leverage, verification, deposits and withdrawals, "
        "account types, complaints, or the status of your own account. What would you like to know?",
        "ar": "يمكنني المساعدة فقط في دعم Aurelia Markets: الرسوم، والهامش والرافعة المالية، والتحقق من الهوية، والإيداع والسحب، "
        "وأنواع الحسابات، والشكاوى، أو حالة حسابك. بماذا يمكنني مساعدتك؟",
    },
    "need_sign_in": {
        "en": "Please sign in first so I can look at your own account. (In this demo, pick a demo customer at the top.)",
        "ar": "يرجى تسجيل الدخول أولاً حتى أتمكن من الاطلاع على حسابك. (في هذا العرض التجريبي، اختر عميلاً تجريبيًا من أعلى الصفحة.)",
    },
    "blocked_prompt_injection": {
        "en": "I can't act on instructions that try to change how I work. I'm happy to help with questions about your account or our policies.",
        "ar": "لا يمكنني تنفيذ تعليمات تحاول تغيير طريقة عملي. يسعدني مساعدتك في الأسئلة المتعلقة بحسابك أو بسياساتنا.",
    },
    "blocked_empty_message": {
        "en": "Could you type your question? I can help with fees, margin, verification, withdrawals, or your own account.",
        "ar": "هل يمكنك كتابة سؤالك؟ يمكنني المساعدة في الرسوم والهامش والتحقق من الهوية والسحب، أو في حسابك.",
    },
    "blocked_message_too_long": {
        "en": "That message is too long for me to process. Could you shorten it to the main question?",
        "ar": "هذه الرسالة أطول من أن أعالجها. هل يمكنك اختصارها إلى السؤال الرئيسي؟",
    },
    "not_in_documents": {
        "en": "I don't have that in the Aurelia Markets documents I can see. I can log a ticket so a colleague can answer - just say 'log a ticket'.",
        "ar": "لا تتوفر لدي هذه المعلومة في مستندات Aurelia Markets المتاحة لي. يمكنني فتح تذكرة ليجيبك أحد الزملاء - فقط اكتب «افتح تذكرة».",
    },
    "complaint_sign_in": {
        "en": "I'm sorry you're having a bad experience. Please sign in so I can log your complaint, or email "
        "complaints@aurelia-markets.example and we'll respond within 15 business days.",
        "ar": "نأسف لأن تجربتك لم تكن جيدة. يرجى تسجيل الدخول حتى أتمكن من تسجيل شكواك، أو راسلنا على "
        "complaints@aurelia-markets.example وسنرد خلال 15 يوم عمل.",
    },
    "complaint_logged": {
        "en": "I'm sorry about that. I've logged your complaint as ticket #{ticket_id}. You'll get an acknowledgement within "
        "2 business days and a final written response within 15 business days [1]. If you're unhappy with the response, you can ask "
        "for it to be escalated to the Head of Compliance.",
        "ar": "نأسف لذلك. سجّلت شكواك برقم التذكرة #{ticket_id}. سيصلك تأكيد بالاستلام خلال 2 يوم عمل، وردّ كتابي نهائي "
        "خلال 15 يوم عمل [1]. وإذا لم يرضك الرد، يمكنك طلب تصعيد الشكوى إلى رئيس قسم الامتثال.",
    },
    "limit_ask_amount": {
        "en": "Happy to submit that for review. What daily withdrawal limit (in USD) would you like?",
        "ar": "يسعدني إرسال طلبك للمراجعة. ما حد السحب اليومي الذي تريده (بالدولار الأمريكي USD)؟",
    },
    "limit_ask_reason": {
        "en": "Thanks. What is the reason for the higher limit? A compliance reviewer will read it.",
        "ar": "شكرًا. ما سبب طلب الحد الأعلى؟ سيطّلع عليه أحد مراجعي الامتثال.",
    },
    "limit_not_eligible": {
        "en": "I can't send that request for review: it must be above your current {current} USD{ceiling_part} [1]. "
        "Completing the next level of verification is usually the faster route - I can tell you what's needed.",
        "ar": "لا يمكنني إرسال هذا الطلب للمراجعة: يجب أن يكون المبلغ أعلى من حدك الحالي البالغ {current} USD{ceiling_part} [1]. "
        "إكمال المستوى التالي من التحقق هو عادةً الطريق الأسرع - يمكنني أن أوضح لك ما هو مطلوب.",
    },
    "limit_ceiling_part": {
        "en": " and no higher than {ceiling} USD (the next verification tier's limit)",
        "ar": " وألا يتجاوز {ceiling} USD (حد مستوى التحقق التالي)",
    },
    "closure_ask_reason": {
        "en": "I can pass a closure request to our compliance team. Could you tell me briefly why you'd like to close the account?",
        "ar": "يمكنني تحويل طلب إغلاق الحساب إلى فريق الامتثال. هل يمكنك أن تخبرني باختصار لماذا تريد إغلاق الحساب؟",
    },
    "closure_open_positions": {
        "en": "You still have open positions. Please close them (and withdraw any balance) first, then ask me again [1].",
        "ar": "لا تزال لديك صفقات مفتوحة. يرجى إغلاقها (وسحب أي رصيد متبقٍ) أولاً، ثم اطلب ذلك مرة أخرى [1].",
    },
    "review_declined": {
        "en": "A compliance reviewer looked at your request and could not approve it: {note}. "
        "You can ask me about the verification steps that would raise your limits.",
        "ar": "راجع أحد مراجعي الامتثال طلبك ولم يتمكن من الموافقة عليه: {note}. "
        "يمكنك أن تسألني عن خطوات التحقق التي ترفع حدودك.",
    },
    "review_declined_default_note": {
        "en": "it didn't meet our policy requirements at this time",
        "ar": "لم يستوفِ متطلبات سياستنا في الوقت الحالي",
    },
    "limit_granted": {
        "en": "Good news - a compliance reviewer approved your request. Your daily withdrawal limit is now {amount} USD.",
        "ar": "خبر سار - وافق أحد مراجعي الامتثال على طلبك. أصبح حد السحب اليومي لديك الآن {amount} USD.",
    },
    "limit_apply_failed": {
        "en": "The reviewer approved it, but the system could not apply it ({error}). I've kept your request on file.",
        "ar": "وافق المراجع على الطلب، لكن النظام لم يتمكن من تطبيقه ({error}). احتفظنا بطلبك في السجل.",
    },
    "closure_approved": {
        "en": "A compliance reviewer approved your request. Your account is now pending closure; you'll receive a confirmation email.",
        "ar": "وافق أحد مراجعي الامتثال على طلبك. حسابك الآن قيد الإغلاق، وستصلك رسالة تأكيد بالبريد الإلكتروني.",
    },
    "closure_apply_failed": {
        "en": "The reviewer approved it, but it could not be applied ({error}).",
        "ar": "وافق المراجع على الطلب، لكن تعذّر تطبيقه ({error}).",
    },
    "holding": {
        "en": "Thanks - I've sent your request to a compliance reviewer. They'll review it and you'll see the decision here. "
        "(This usually takes up to 2 business days.)",
        "ar": "شكرًا - أرسلت طلبك إلى أحد مراجعي الامتثال. سيراجعه وسترى القرار هنا. (يستغرق ذلك عادةً حتى 2 يوم عمل.)",
    },
    "still_pending": {
        "en": "Your earlier request is still with a compliance reviewer - I'll show the decision as soon as it's made.",
        "ar": "طلبك السابق لا يزال لدى مراجع الامتثال - سأعرض القرار فور صدوره.",
    },
}


def msg(key: str, lang: str = "en", **values: object) -> str:
    texts = MESSAGES[key]
    return texts.get(lang, texts["en"]).format(**values)
