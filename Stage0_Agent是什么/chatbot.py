from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class SimpleChatbot:
    def __init__(self, api_key=None, model=None, system_prompt=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.messages = []  # 简单的对话历史
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    @staticmethod
    def _clean(text):
        """移除无效的 surrogate 字符，防止 UnicodeEncodeError"""
        if isinstance(text, str):
            return text.encode("utf-8", errors="replace").decode("utf-8")
        return text

    def chat(self, user_input, json_mode=False):
        """
        与模型对话。

        Args:
            user_input: 用户输入文本。
            json_mode: 若为 True，强制模型以 JSON 格式输出。
                       此时需在 prompt 中说明要输出的字段结构。
        """
        user_input = self._clean(user_input)
        self.messages.append({"role": "user", "content": user_input})

        # 构造请求参数
        kwargs = {
            "model": self.model,
            "messages": self.messages,
        }
        if json_mode:
            # ★ 启用 JSON 模式：强制模型返回合法 JSON 字符串
            #   注意：搭配 system prompt 告知模型输出结构效果更好
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        reply = response.choices[0].message.content
        reply = self._clean(reply)
        self.messages.append({"role": "assistant", "content": reply})
        return reply
    
if __name__ == '__main__':
    # 无需传 api_key，自动从 .env 文件读取 DEEPSEEK_API_KEY
    chat = SimpleChatbot(
        system_prompt="你是一个数据助手。请始终用 JSON 格式回复。输出格式你来定\n"
    )
    while True:
        meessages = input('请输入对话：')
        meessages = chat.chat(meessages)
        history = chat.messages
        print("历史对话：" + str(history) + "\n")
        print("回复对话：" + meessages + "\n")