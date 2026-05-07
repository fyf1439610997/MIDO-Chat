ID_PARSE_PROMPTS: dict[str, str] = {
    "id_parse_system": """
你是{assistant_name}。你的任务是把教师上传的教案文本抽取为严格 JSON 数组。
输出要求：
1) 仅输出 JSON，不要包含任何解释文字、Markdown 或代码块标记。
2) JSON 顶层必须是数组，数组元素对象字段固定为：
   - id: 整数，从 1 开始递增
   - time: 字符串，格式形如 "00:00-05:00"
   - phase: 字符串，教学环节名称（英文短语）
   - intent: 字符串，该环节教学意图（英文短语）
3) 优先按教案里出现的时间顺序组织。若缺少明确时间，请合理补全连续时段。
4) 环节数量控制在 {min_steps}~{max_steps} 个，尽可能覆盖整段课程流程。
""".strip(),
    "id_parse_user": """
请结构化以下教学设计文本：

{document_text}

以下是课堂音频转写（可能包含时间片段），请用于校准各环节 time 字段，使时间划分更贴近真实课堂进程：

{transcript_text}
""".strip(),
}
