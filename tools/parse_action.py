#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parse_action.py — 解析并校验智能导购机器人输出的 JSON 动作指令。

分类 3（位置与带看）和分类 5（问候与动作）的提示词都要求「只输出一个 JSON」，
但模型实际回复里常带上 <think> 思考块、Markdown 代码围栏，或前后夹一句多余的话。
本脚本负责：

1. 从原始回复中把 JSON 抠出来（剥离思考块与代码围栏，按括号配对截取）
2. 判断属于带看指令（分类 3）还是动作指令（分类 5）
3. 按提示词约定的动作白名单和字段约束做校验
4. 把话术与动作分开打印，便于接入下位机

仅依赖标准库，Python 3.8+。

用法：
    python tools/parse_action.py '{"speech": "你好", "action": "wave"}'
    python tools/parse_action.py --file reply.txt
    type reply.txt | python tools/parse_action.py
    python tools/parse_action.py --demo

退出码：0 = 校验通过（允许警告），1 = 解析失败或存在错误级问题。
"""

import argparse
import json
import re
import sys

# 动作白名单，与 prompts/chatflow-complete.md 中的提示词保持一致
GUIDE_ACTIONS = ("guide_to", "none")
GREET_ACTIONS = ("wave", "handshake", "dance", "spin", "none")

GUIDE_KEYS = ("action", "target", "speech_on_start", "speech_on_arrival")
GREET_KEYS = ("action", "speech")

# 提示词要求话术里不出现 Markdown 符号和括号
MARKDOWN_CHARS = ("*", "#", "`", "_")
BRACKET_CHARS = ("（", "）", "(", ")")

# 知识库没有价格、库存、质保信息，出现这些词大概率是模型编造
FABRICATION_WORDS = ("价格", "多少钱", "库存", "有货", "在售", "质保", "一共几款")
# 疑似报价，如 "3999元"、"¥3999"
PRICE_PATTERN = re.compile(r"\d+\s*元|[¥￥]\s*\d")

# 中文输入法下的全角引号，会让 JSON 直接解析失败
FULLWIDTH_QUOTES = {
    "\u201c": '"',  # “
    "\u201d": '"',  # ”
    "\uff02": '"',  # ＂
    "\u2018": "'",  # ‘
    "\u2019": "'",  # ’
}

ERROR = "错误"
WARN = "警告"


def strip_think(text):
    """剥离 DeepSeek 的 <think>...</think> 思考块。"""
    without_pairs = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.S | re.I)
    # 兼容只开了头没闭合的情况
    return re.sub(r"<think\b[^>]*>.*$", "", without_pairs, flags=re.S | re.I)


def strip_fences(text):
    """剥离 ```json ... ``` 代码围栏，保留内容。"""
    pattern = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.S)
    return pattern.sub(lambda m: m.group(1), text)


def find_json_object(text):
    """在文本中找到第一个括号配平的 JSON 对象。

    返回 (obj, start, end)；找不到返回 (None, -1, -1)。
    按字符扫描并跟踪字符串状态，避免把话术里的花括号当成结构。
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate), start, i + 1
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None, -1, -1


def normalize_quotes(text):
    """把全角引号换成半角引号，返回 (新文本, 替换次数)。"""
    count = 0
    for src, dst in FULLWIDTH_QUOTES.items():
        hits = text.count(src)
        if hits:
            count += hits
            text = text.replace(src, dst)
    return text, count


def extract(raw):
    """从模型原始回复中提取 JSON。

    返回 (obj, notes, error, quote_fixed)。
    """
    notes = []

    cleaned = strip_think(raw)
    if cleaned != raw:
        notes.append("已剥离 <think> 思考块")

    unfenced = strip_fences(cleaned)
    if unfenced != cleaned:
        notes.append("已剥离 Markdown 代码围栏")
    cleaned = unfenced

    obj, start, end = find_json_object(cleaned)
    quote_fixed = False
    if obj is None:
        # 模型有时会输出中文全角引号，先尝试归一化再解析
        normalized, replaced = normalize_quotes(cleaned)
        if replaced:
            obj, start, end = find_json_object(normalized)
            if obj is not None:
                quote_fixed = True
                cleaned = normalized
                notes.append("回复使用了全角引号，已替换为半角后解析（%d 处）" % replaced)

    if obj is None:
        return None, notes, "回复中没有找到合法的 JSON 对象", False

    if cleaned[:start].strip() or cleaned[end:].strip():
        notes.append("JSON 前后存在多余文字，已忽略")

    if not isinstance(obj, dict):
        return None, notes, "JSON 顶层不是对象", False

    return obj, notes, None, quote_fixed


def classify(obj):
    """按字段判断指令类型：guide（分类 3）/ greeting（分类 5）/ unknown。"""
    if any(key in obj for key in ("target", "speech_on_start", "speech_on_arrival")):
        return "guide"
    if "speech" in obj or "action" in obj:
        return "greeting"
    return "unknown"


def check_text(field, value, issues):
    """检查话术字段的书写规范，只产生警告。"""
    if not isinstance(value, str) or not value:
        return
    hits = [c for c in MARKDOWN_CHARS if c in value]
    if hits:
        issues.append((WARN, "%s 含 Markdown 符号：%s" % (field, " ".join(hits))))
    hits = [c for c in BRACKET_CHARS if c in value]
    if hits:
        issues.append((WARN, "%s 含括号：%s" % (field, " ".join(hits))))
    hits = [w for w in FABRICATION_WORDS if w in value]
    if hits:
        issues.append((WARN, "%s 出现知识库没有的信息：%s" % (field, "、".join(hits))))
    price = PRICE_PATTERN.search(value)
    if price:
        issues.append((WARN, "%s 出现疑似报价：%s" % (field, price.group(0).strip())))


def validate_guide(obj, issues):
    for key in GUIDE_KEYS:
        if key not in obj:
            issues.append((ERROR, "缺少字段 %s" % key))

    action = obj.get("action")
    if action is not None and action not in GUIDE_ACTIONS:
        hint = ""
        if action in GREET_ACTIONS:
            hint = "（这是分类 5 的动作，分类 3 只能用 guide_to 或 none）"
        issues.append((ERROR, "action=%r 不在白名单内%s" % (action, hint)))

    if action == "guide_to":
        if not obj.get("target"):
            issues.append((ERROR, "action=guide_to 时 target 不能为空"))
        if not obj.get("speech_on_start"):
            issues.append((ERROR, "action=guide_to 时 speech_on_start 不能为空"))
        if not obj.get("speech_on_arrival"):
            issues.append((ERROR, "action=guide_to 时 speech_on_arrival 不能为空"))
    elif action == "none":
        if obj.get("speech_on_arrival"):
            issues.append((WARN, "action=none 时 speech_on_arrival 应为空字符串"))

    for key in ("target", "speech_on_start", "speech_on_arrival"):
        check_text(key, obj.get(key), issues)


def validate_greeting(obj, issues):
    for key in GREET_KEYS:
        if key not in obj:
            issues.append((ERROR, "缺少字段 %s" % key))

    action = obj.get("action")
    if action is not None and action not in GREET_ACTIONS:
        hint = ""
        if action == "guide_to":
            hint = "（guide_to 属于分类 3 的带看指令）"
        issues.append((ERROR, "action=%r 不在白名单内%s" % (action, hint)))

    speech = obj.get("speech")
    if isinstance(speech, str):
        if not speech.strip():
            issues.append((ERROR, "speech 不能为空"))
        elif len(speech) > 60:
            issues.append((ERROR, "speech 长度 %d 字，超过自我介绍上限 60 字" % len(speech)))
        elif len(speech) > 20:
            issues.append((WARN, "speech 长度 %d 字，超过常规上限 20 字" % len(speech)))
        check_text("speech", speech, issues)


def validate(obj):
    """返回 (类型, 问题列表)。"""
    issues = []
    kind = classify(obj)
    if kind == "guide":
        validate_guide(obj, issues)
    elif kind == "greeting":
        validate_greeting(obj, issues)
    else:
        issues.append((ERROR, "无法判断指令类型：既没有 target/speech_on_start，也没有 speech"))
    return kind, issues


def render(kind, obj, notes, issues):
    """把结果排成便于阅读的文本。"""
    lines = []
    label = {"guide": "带看指令（分类 3）", "greeting": "动作指令（分类 5）"}.get(kind, "未知类型")
    lines.append("类型：%s" % label)

    if kind == "guide":
        lines.append("动作：%s" % obj.get("action", "—"))
        lines.append("位置：%s" % (obj.get("target") or "—"))
        lines.append("出发话术：%s" % (obj.get("speech_on_start") or "—"))
        lines.append("到达话术：%s" % (obj.get("speech_on_arrival") or "—"))
    elif kind == "greeting":
        lines.append("动作：%s" % obj.get("action", "—"))
        speech = obj.get("speech") or ""
        lines.append("话术：%s（%d 字）" % (speech or "—", len(speech)))

    if notes:
        lines.append("")
        for note in notes:
            lines.append("· %s" % note)

    errors = [m for level, m in issues if level == ERROR]
    warns = [m for level, m in issues if level == WARN]

    lines.append("")
    if not issues:
        lines.append("校验结果：通过")
    else:
        lines.append("校验结果：%d 个错误，%d 个警告" % (len(errors), len(warns)))
        for level, message in issues:
            lines.append("  [%s] %s" % (level, message))
    return "\n".join(lines)


DEMO_CASES = [
    ('{"speech": "你好", "action": "handshake"}', "正常动作指令"),
    ('{"action": "guide_to", "target": "A区2号展位", '
     '"speech_on_start": "好的，请跟我来", '
     '"speech_on_arrival": "这款65寸电视支持4K分辨率，适合客厅使用。"}', "正常带看指令"),
    ('<think>用户要握手</think>\n```json\n{"speech": "你好", "action": "handshake"}\n```', "带思考块与代码围栏"),
    ('{"action": "none", "target": "", "speech_on_start": "这个商品的位置我暂时无法确认，请咨询现场导购员", '
     '"speech_on_arrival": "这款在A区"}', "action=none 但到达话术非空"),
    ('{"speech": "这款冰箱**特价**只要3999元，库存充足", "action": "wave"}', "含 Markdown 与编造信息"),
    ('{"action": "dance", "target": "B区1号展位", "speech_on_start": "走", "speech_on_arrival": "到了"}',
     "分类 3 用了分类 5 的动作"),
    ('{\u201caction\u201d: \u201cguide_to\u201d, \u201ctarget\u201d: \u201cA区2号展位\u201d, '
     '\u201cspeech_on_start\u201d: \u201c好的，请跟我来。\u201d, '
     '\u201cspeech_on_arrival\u201d: \u201c这款是65英寸4K量子点电视，支持120Hz刷新率。\u201d}',
     "全角引号（中文输入法直接输出）"),
    ("抱歉，我不太明白你的意思。", "根本没有 JSON"),
]


def read_input(args):
    if args.demo:
        return None
    if args.text:
        return args.text
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            return handle.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


def run_one(raw, quiet=False):
    obj, notes, error, quote_fixed = extract(raw)
    if error:
        if not quiet:
            print("解析失败：%s" % error)
            print("原始回复：%s" % raw.strip()[:120])
        return 1
    kind, issues = validate(obj)
    if quote_fixed:
        issues.append((WARN, "原回复用全角引号，不是合法 JSON，已自动修正；建议在提示词里明确要求半角引号"))
    if not quiet:
        print(render(kind, obj, notes, issues))
    return 1 if any(level == ERROR for level, _ in issues) else 0


def main():
    parser = argparse.ArgumentParser(
        description="解析并校验智能导购机器人输出的 JSON 动作指令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="?", help="模型原始回复文本")
    parser.add_argument("--file", help="从文件读取回复文本")
    parser.add_argument("--demo", action="store_true", help="跑一组内置示例")
    args = parser.parse_args()

    if args.demo:
        failed = 0
        for index, (raw, title) in enumerate(DEMO_CASES, 1):
            print("=" * 60)
            print("示例 %d：%s" % (index, title))
            print("-" * 60)
            if run_one(raw) != 0:
                failed += 1
            print()
        print("=" * 60)
        print("共 %d 个示例，%d 个未通过校验" % (len(DEMO_CASES), failed))
        return 0

    raw = read_input(args)
    if raw is None:
        parser.print_help()
        return 1
    return run_one(raw)


if __name__ == "__main__":
    sys.exit(main())
