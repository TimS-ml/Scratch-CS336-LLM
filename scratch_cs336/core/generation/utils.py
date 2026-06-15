"""
Generation utility functions for context building and math reasoning.
用于上下文构建和数学推理的生成实用函数。
"""

import re
import torch
from typing import List, Dict, Any, Optional


def make_context(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 0,
) -> torch.LongTensor:
    """
    Build context tokens for multi-turn conversations.
    为多轮对话构建上下文tokens。

    This function processes a list of conversation messages and converts them into
    tokenized format suitable for language model input. It handles system prompts,
    conversation history, and ensures the total length doesn't exceed model limits.

    此函数处理对话消息列表并将其转换为适合语言模型输入的token化格式。
    它处理系统提示、对话历史，并确保总长度不超过模型限制。

    Args:
        model: The language model instance with generation_config and config attributes.
               语言模型实例，包含generation_config和config属性。
        tokenizer: The tokenizer for encoding text.
                   用于编码文本的tokenizer。
        messages: List of message dictionaries with 'role' and 'content' keys.
                  Format: [{"role": "system/user/assistant", "content": "..."}]
                  消息字典列表，包含'role'和'content'键。
        system: Default system prompt if not provided in messages.
                如果消息中未提供，则使用的默认系统提示。
        max_new_tokens: Maximum number of new tokens to generate (overrides model config).
                        要生成的新token的最大数量（覆盖模型配置）。

    Returns:
        torch.LongTensor: Tokenized input tensor of shape [1, seq_len] on model's device.
                         形状为[1, seq_len]的token化输入张量，位于模型设备上。

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a math tutor."},
        ...     {"role": "user", "content": "What is 2+2?"},
        ...     {"role": "assistant", "content": "4"},
        ...     {"role": "user", "content": "What about 3+3?"}
        ... ]
        >>> input_ids = make_context(model, tokenizer, messages)

    Notes:
        - Messages must end with a user message / 消息必须以用户消息结束
        - History must have paired user-assistant messages / 历史必须有成对的用户-助手消息
        - Uses special tokens: <|system|>, <|user|>, <|assistant|> / 使用特殊token
    """
    # Determine max new tokens: use parameter or model default
    # 确定新生成的token数量：优先使用传入参数，否则使用模型配置中的默认值
    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens

    # Calculate maximum input length (model max length - new tokens to generate)
    # 计算模型允许的最大输入长度（模型最大长度减去新生成的token数）
    max_input_length = model.config.max_position_embeddings - max_new_tokens

    # Encode newline token
    # 编码换行符token
    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)

    def _parse_messages(messages: List[Dict[str, str]]) -> tuple:
        """
        Parse message list into system prompt, query, and conversation history.
        解析消息列表，分离系统消息、查询和对话历史。

        Args:
            messages: List of message dictionaries.
                     消息字典列表。

        Returns:
            tuple: (system_prompt, current_query, history)
                   - system_prompt: System instruction text / 系统指令文本
                   - current_query: Latest user query / 最新用户查询
                   - history: List of [user, assistant] pairs / [用户, 助手]对列表
        """
        system, query, history = "", "", []

        # Extract system message if present
        # 提取系统消息（如果存在）
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]

        # Extract current query (must be last message and from user)
        # 提取当前查询（必须是最后一条消息且来自用户）
        assert messages[-1]["role"] == "user", "Last message must be from user"
        query = messages[-1]["content"]
        messages = messages[:-1]

        # Extract conversation history (must be paired user-assistant messages)
        # 提取对话历史（必须是成对的用户-助手消息）
        assert len(messages) % 2 == 0, "History must have paired user-assistant messages"
        for i in range(0, len(messages), 2):
            assert messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant"
            history.append([messages[i]["content"], messages[i+1]["content"]])

        return system, query, history

    # Parse messages into components
    # 调用_parse_messages解析消息
    _system, query, history = _parse_messages(messages)

    # Build system tokens
    # 构建系统token
    system_text = _system if _system != "" else system
    system_tokens = []
    if system_text:
        system_tokens = tokenizer.encode(
            text=("<|system|>\n" + system_text.strip()),
            add_special_tokens=True,
            truncation=True
        ) + nl_tokens

    # Build query tokens
    # 构建查询token
    query_tokens = tokenizer.encode(
        text=("<|user|>\n" + query.strip()),
        add_special_tokens=False,
        truncation=True
    ) + nl_tokens

    # Build final assistant prefix tokens
    # 构建最终助手前缀token
    final_tokens = tokenizer.encode(
        "<|assistant|>",
        add_special_tokens=False,
        truncation=True
    ) + nl_tokens

    # Calculate maximum allowed history length
    # 计算允许的最大历史长度
    max_history_length = (
        max_input_length - len(system_tokens) - len(query_tokens) - len(final_tokens)
    )

    # Build history tokens (reverse order to keep most recent)
    # 逆序遍历对话历史，构建token序列（保留最近的对话）
    context_tokens = []
    for turn_query, turn_response in reversed(history):
        # Encode user query for this turn
        # 编码此轮的用户查询
        history_query_tokens = tokenizer.encode(
            "<|user|>\n" + turn_query.strip(),
            add_special_tokens=False,
            truncation=True
        ) + nl_tokens

        # Encode assistant response for this turn
        # 编码此轮的助手回复
        history_response_tokens = tokenizer.encode(
            "<|assistant|>\n" + turn_response.strip(),
            add_special_tokens=False,
            truncation=True
        ) + nl_tokens

        # Combine this turn's tokens
        # 组合此轮的token
        next_context_tokens = history_query_tokens + history_response_tokens

        # Check if adding this turn would exceed max history length
        # 确保加入这些token后总长度不超过允许的最大历史长度
        current_context_size = len(next_context_tokens) + len(context_tokens)
        if current_context_size < max_history_length:
            context_tokens = next_context_tokens + context_tokens
        else:
            break

    # Combine all token sequences
    # 组合所有token序列
    input_tokens = system_tokens + context_tokens + query_tokens + final_tokens

    return torch.LongTensor([input_tokens]).to(model.device)


def parse_pot_no_stream(inputs: str) -> str:
    """
    Parse and execute Program-of-Thought (PoT) code blocks for math reasoning.
    解析并执行程序思维(PoT)代码块以进行数学推理。

    This function processes input strings containing special code blocks in the format
    <<...>> which can be either simple math expressions or function definitions.
    It executes the code and replaces the blocks with computed results.

    此函数处理包含特殊格式<<...>>代码块的输入字符串，这些代码块可以是简单的
    数学表达式或函数定义。它执行代码并将块替换为计算结果。

    Args:
        inputs: Input string containing <<code>> blocks to be evaluated.
                包含要评估的<<代码>>块的输入字符串。

    Returns:
        str: Processed string with code blocks replaced by their results.
             处理后的字符串，代码块被其结果替换。

    Example:
        >>> text = "The answer is <<x = 5 + 3>>, so x equals 8."
        >>> parse_pot_no_stream(text)
        "The answer is , so 8 equals 8."

        >>> text = "Calculate <<ans = func()\\ndef func():\\n    return 42>>"
        >>> parse_pot_no_stream(text)
        "Calculate "  # with ans replaced by 42 in subsequent references

    Notes:
        PoT Processing Logic / PoT处理逻辑:
        1. For code with "func": Identifies function definitions, executes them,
           and replaces results. Handles sympy special cases.
           对于包含"func"的代码：识别函数定义，执行它们并替换结果。处理sympy特殊情况。

        2. For simple expressions: Directly evaluates right-hand side and replaces
           with computed value (converts floats to ints when appropriate).
           对于简单表达式：直接计算右侧并替换为计算值（适当时将浮点数转换为整数）。

        3. Handles variable propagation across multiple code blocks.
           处理多个代码块之间的变量传播。

    Warning:
        This function uses exec() and eval() which can be dangerous with untrusted input.
        Only use with controlled, trusted input strings.
        此函数使用exec()和eval()，对于不受信任的输入可能很危险。
        仅用于受控的、可信的输入字符串。
    """
    try:
        # Find all <<...>> patterns in the input
        # 尝试从输入字符串中找到形如 "<<...>>" 的模式
        s = re.findall(r'<<(.*?)>>', inputs, re.DOTALL)

        # If no patterns found, return original input
        # 如果没有找到匹配项，则直接返回原始输入
        if not s:
            return inputs

        index = 0
        # Process each matched pattern
        # 遍历所有匹配到的模式
        for k in s:
            try:
                # Check if pattern contains function definition
                # 检查模式内是否包含 "func"
                if "func" in k:
                    # Split by '=' to separate variable assignment from code
                    # 分割并处理函数定义
                    var = k.split("=", 1)
                    try:
                        # Execute function definition
                        # 去除空白字符并执行函数定义
                        var[1] = var[1].strip(" ")
                        exec(var[1], globals())
                        # Call function to get result
                        # 调用函数获取结果
                        ans = func()
                    except:
                        # Special handling for sympy library
                        # 特殊处理包含 'sympy' 的情况
                        if 'sympy' in var[1]:
                            var[1] = var[1].replace('res[x]', 'res[0][0]').replace('res[y]', 'res[0][1]')
                            exec(var[1], globals())
                            ans = func()
                        pass

                    # Parse variable names
                    # 解析变量名
                    var_list = [c.strip(" ") for c in var[0].split(",")]

                    # If single variable, wrap result in list
                    # 如果只有一个变量名，则将结果放入列表
                    if len(var_list) == 1:
                        ans = [ans]

                    # Convert results to float/int and replace in string
                    # 将结果转换为浮点数或整数形式，并替换到输入字符串中
                    for i in range(len(ans)):
                        try:
                            ans[i] = float(ans[i])
                            # Convert to int if very close to integer value
                            # 如果接近整数值则转换为整数
                            if abs(ans[i] - int(ans[i])) < 1e-10:
                                ans[i] = str(int(ans[i]))
                        except:
                            pass

                    # Remove code block and replace variable references
                    # 替换原字符串中的模式和变量名
                    inputs = inputs.replace("<<" + k + ">>", "")
                    for i in range(len(var_list)):
                        inputs = inputs.replace(var_list[i], str(ans[i]))

                    index += 1

                    # Update variables in subsequent patterns
                    # 更新后续模式中的变量值
                    for c in range(index, len(s)):
                        for i in range(len(var_list)):
                            s[c] = s[c].replace(var_list[i], str(ans[i]))
                else:
                    # Process non-function code blocks (simple expressions)
                    # 处理非函数的情况，直接计算并替换
                    var = k.replace(" ", "").split("=")
                    var[1] = var[1].replace("eval", "")

                    # Evaluate expression
                    # 评估表达式
                    ans = round(eval(var[1]), 10)
                    ans = float(ans)

                    # Convert to int if appropriate
                    # 如果接近整数则转换为整数
                    if abs(ans - int(ans)) < 1e-10:
                        ans = str(int(ans))

                    # Replace code block and variable references
                    # 替换原字符串中的模式和变量名
                    inputs = inputs.replace("<<" + k + ">>", "").replace(var[0], str(ans))

                    index += 1

                    # Update variables in subsequent patterns
                    # 更新后续模式中的变量值
                    for c in range(index, len(s)):
                        s[c] = s[c].replace(var[0], str(ans))
            except:
                # If processing fails for any block, return current state
                # 如果任何块处理失败，返回当前状态
                return inputs
    except Exception as e:
        # If overall processing fails, return original input
        # 如果整体处理失败，返回原始输入
        return inputs

    return inputs
