import copy
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import fitz
import gradio as gr
import spaces
import torch
from transformers import (
    AutoProcessor,
    Glm4vForConditionalGeneration,
    TextIteratorStreamer,
)
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7890, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
args = parser.parse_args()

MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"
stop_generation = False
processor = None
model = None


def load_model():
    global processor, model
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )


class GLM4VModel:
    def __init__(self):
        pass

    def _strip_html(self, t):
        return re.sub(r"<[^>]+>", "", t).strip()

    def _wrap_text(self, t):
        return [{"type": "text", "text": t}]

    def _pdf_to_imgs(self, pdf_path):
        doc = fitz.open(pdf_path)
        imgs = []
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(dpi=180)
            img_p = os.path.join(
                tempfile.gettempdir(), f"{Path(pdf_path).stem}_{i}.png"
            )
            pix.save(img_p)
            imgs.append(img_p)
        doc.close()
        return imgs

    def _ppt_to_imgs(self, ppt_path):
        tmp = tempfile.mkdtemp()
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                tmp,
                ppt_path,
            ],
            check=True,
        )
        pdf_path = os.path.join(tmp, Path(ppt_path).stem + ".pdf")
        return self._pdf_to_imgs(pdf_path)

    def _files_to_content(self, media):
        out = []
        for f in media or []:
            ext = Path(f.name).suffix.lower()
            if ext in [
                ".mp4",
                ".avi",
                ".mkv",
                ".mov",
                ".wmv",
                ".flv",
                ".webm",
                ".mpeg",
                ".m4v",
            ]:
                out.append({"type": "video", "url": f.name})
            elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
                out.append({"type": "image", "url": f.name})
            elif ext in [".ppt", ".pptx"]:
                for p in self._ppt_to_imgs(f.name):
                    out.append({"type": "image", "url": p})
            elif ext == ".pdf":
                for p in self._pdf_to_imgs(f.name):
                    out.append({"type": "image", "url": p})
        return out

    def _stream_fragment(self, buf: str) -> str:
        think_html = ""
        if "<think>" in buf:
            if "</think>" in buf:
                seg = re.search(r"<think>(.*?)</think>", buf, re.DOTALL)
                if seg:
                    think_html = (
                        "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking</summary>"
                        "<div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                        + seg.group(1).strip().replace("\n", "<br>")
                        + "</div></details>"
                    )
            else:
                part = buf.split("<think>", 1)[1]
                think_html = (
                    "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking</summary>"
                    "<div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                    + part.replace("\n", "<br>")
                    + "</div></details>"
                )

        answer_html = ""
        if "<answer>" in buf:
            if "</answer>" in buf:
                seg = re.search(r"<answer>(.*?)</answer>", buf, re.DOTALL)
                if seg:
                    answer_html = seg.group(1).strip()
            else:
                answer_html = buf.split("<answer>", 1)[1]

        if not think_html and not answer_html:
            return self._strip_html(buf)
        return think_html + answer_html

    def _build_messages(self, raw_hist, sys_prompt):
        msgs = []

        if sys_prompt.strip():
            msgs.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sys_prompt.strip()}],
                }
            )

        for h in raw_hist:
            if h["role"] == "user":
                msgs.append({"role": "user", "content": h["content"]})
            else:
                raw = h["content"]
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
                raw = re.sub(r"<details.*?</details>", "", raw, flags=re.DOTALL)
                clean = self._strip_html(raw).strip()
                msgs.append({"role": "assistant", "content": self._wrap_text(clean)})
        return msgs

    @spaces.GPU(duration=240)
    def stream_generate(self, raw_hist, sys_prompt):
        global stop_generation, processor, model
        stop_generation = False
        msgs = self._build_messages(raw_hist, sys_prompt)
        inputs = processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        streamer = TextIteratorStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        gen_args = dict(
            inputs,
            max_new_tokens=8192,
            repetition_penalty=1.1,
            do_sample=True,
            top_k=2,
            temperature=None,
            top_p=1e-5,
            streamer=streamer,
        )

        generation_thread = threading.Thread(target=model.generate, kwargs=gen_args)
        generation_thread.start()

        buf = ""
        for tok in streamer:
            if stop_generation:
                break
            buf += tok
            yield self._stream_fragment(buf)

        generation_thread.join()


def format_display_content(content):
    if isinstance(content, list):
        text_parts = []
        file_count = 0
        for item in content:
            if item["type"] == "text":
                text_parts.append(item["text"])
            else:
                file_count += 1

        display_text = " ".join(text_parts)
        if file_count > 0:
            return f"[{file_count} file(s) uploaded]\n{display_text}"
        return display_text
    return content


def create_display_history(raw_hist):
    display_hist = []
    for h in raw_hist:
        if h["role"] == "user":
            display_content = format_display_content(h["content"])
            display_hist.append({"role": "user", "content": display_content})
        else:
            display_hist.append({"role": "assistant", "content": h["content"]})
    return display_hist


# 加载模型和处理器
load_model()
glm4v = GLM4VModel()


def check_files(files):
    vids = imgs = ppts = pdfs = 0
    for f in files or []:
        ext = Path(f.name).suffix.lower()
        if ext in [
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".mpeg",
            ".m4v",
        ]:
            vids += 1
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
            imgs += 1
        elif ext in [".ppt", ".pptx"]:
            ppts += 1
        elif ext == ".pdf":
            pdfs += 1
    if vids > 1 or ppts > 1 or pdfs > 1:
        return False, "Only one video or one PPT or one PDF allowed"
    if imgs > 10:
        return False, "Maximum 10 images allowed"
    if (ppts or pdfs) and (vids or imgs) or (vids and imgs):
        return False, "Cannot mix documents, videos, and images"
    return True, ""


def chat(files, msg, raw_hist, sys_prompt):
    global stop_generation
    stop_generation = False

    ok, err = check_files(files)
    if not ok:
        raw_hist.append({"role": "assistant", "content": err})
        display_hist = create_display_history(raw_hist)
        yield display_hist, copy.deepcopy(raw_hist), None, ""
        return

    payload = glm4v._files_to_content(files) if files else None
    if msg.strip():
        if payload is None:
            payload = glm4v._wrap_text(msg.strip())
        else:
            payload.append({"type": "text", "text": msg.strip()})

    user_rec = {"role": "user", "content": payload if payload else msg.strip()}
    if raw_hist is None:
        raw_hist = []
    raw_hist.append(user_rec)

    place = {"role": "assistant", "content": ""}
    raw_hist.append(place)

    display_hist = create_display_history(raw_hist)
    yield display_hist, copy.deepcopy(raw_hist), None, ""

    for chunk in glm4v.stream_generate(raw_hist[:-1], sys_prompt):
        if stop_generation:
            break
        place["content"] = chunk
        display_hist = create_display_history(raw_hist)
        yield display_hist, copy.deepcopy(raw_hist), None, ""

    display_hist = create_display_history(raw_hist)
    yield display_hist, copy.deepcopy(raw_hist), None, ""


def reset():
    global stop_generation
    stop_generation = True
    time.sleep(0.1)
    return [], [], None, ""


css = """.chatbot-container .message-wrap .message{font-size:14px!important}
details summary{cursor:pointer;font-weight:bold}
details[open] summary{margin-bottom:10px}"""

demo = gr.Blocks(title="GLM-4.1V Chat", theme=gr.themes.Soft(), css=css)
with demo:
    gr.Markdown("""
               <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                   GLM-4.1V-9B-Thinking Gradio Space🤗
                </div>
               <div style="text-align: center;">
               <a href="https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking">🤗 Model Hub</a> | 
               <a href="https://github.com/THUDM/GLM-4.1V-Thinking">🌐 Github</a> 
                </div>
                """)

    raw_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=7):
            chatbox = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=800,
                elem_classes="chatbot-container",
            )
            textbox = gr.Textbox(label="💭 Message")
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=3):
            up = gr.File(
                label="📁 Upload",
                file_count="multiple",
                file_types=["file"],
                type="filepath",
            )
            gr.Markdown("Supports images / videos / PPT / PDF")
            gr.Markdown(
                "The maximum supported input is 10 images or 1 video/PPT/PDF. During the conversation, video and images cannot be present at the same time."
            )
            sys = gr.Textbox(label="⚙️ System Prompt", lines=6)

    gr.on(
        triggers=[send.click, textbox.submit],
        fn=chat,
        inputs=[up, textbox, raw_history, sys],
        outputs=[chatbox, raw_history, up, textbox],
    )
    clear.click(reset, outputs=[chatbox, raw_history, up, textbox])

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )
