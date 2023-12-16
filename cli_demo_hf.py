import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
parser.add_argument("--top_k", type=int, default=100, help='top k for top k sampling')
parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
parser.add_argument("--english", action='store_true', help='only output English')
parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="/mnt/d/LLM/models/visualglm-6b/", help='pretrained ckpt')
parser.add_argument("--prompt_zh", type=str, default="描述这张图片。", help='Chinese prompt for the first round')
parser.add_argument("--prompt_en", type=str, default="Describe the image.", help='English prompt for the first round')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained, trust_remote_code=True)
model = AutoModel.from_pretrained(args.from_pretrained, trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history, prefix):
    prompt = prefix
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nVisualGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    global stop_stream
    while True:
        history = []
        prefix = "欢迎使用 VisualGLM-6B 模型，输入图片路径和内容即可进行对话，clear 清空对话历史，stop 终止程序"
        print(prefix)
        image_path = input("\n请输入图片路径：")
        if image_path == "stop":
            break
        prefix = prefix + "\n" + image_path
        query = "描述这张图片。"
        while True:
            count = 0
            with torch.no_grad():
                for response, history in model.stream_chat(tokenizer, image_path, query, history=history):
                    if stop_stream:
                        stop_stream = False
                        break
                    else:
                        count += 1
                        if count % 8 == 0:
                            os.system(clear_command)
                            print(build_prompt(history, prefix), flush=True)
                            signal.signal(signal.SIGINT, signal_handler)
            os.system(clear_command)
            print(build_prompt(history, prefix), flush=True)
            query = input("\n用户：")
            if query.strip() == "clear":
                break
            if query.strip() == "stop":
                stop_stream = True
                exit(0)
            # if query.strip() == "clear":
            #     history = []
            #     os.system(clear_command)
            #     print(prefix)
            #     continue


if __name__ == "__main__":
    main()