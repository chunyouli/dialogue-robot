import cv2
import pyaudio
import wave
import threading
import numpy as np
import time
from queue import Queue
import webrtcvad
import os
import threading
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
from funasr import AutoModel
import pygame
import edge_tts
import asyncio
from time import sleep
import langid
from langdetect import detect
import re
from pypinyin import pinyin, Style
from modelscope.pipelines import pipeline
from transformers import BitsAndBytesConfig
from rag_manager import RAGManager

# --- 配置huggingFace国内镜像 ---
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #通过镜像加载未下载的预训练模型

# 参数设置
AUDIO_RATE = 16000        # 音频采样率
AUDIO_CHANNELS = 1        # 单声道
CHUNK = 1024              # 音频块大小
VAD_MODE = 3              # VAD 模式 (0-3, 数字越大越敏感)
OUTPUT_DIR = "./output"   # TTS输出目录
NO_SPEECH_THRESHOLD = 1   # 无效语音阈值，单位：秒
folder_path = "./Test_QWen2_VL/"  #我的声音
audio_file_count = 0
audio_file_count_tmp = 0

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

# 创建俩队列用于音频和视频同步缓存
audio_queue = Queue()
video_queue = Queue()

# 全局变量
last_active_time = time.time()  #记录最后一次活动时间
recording_active = True         #标记是都在录音
segments_to_save = []           #保存待保存的音频段和已保存的时间间隔
saved_intervals = []
last_vad_end_time = 0  # 上次保存的 VAD 有效段结束时间    记录上次 VAD 停止的时间


# --- 唤醒词、声纹变量配置 ---
set_KWS = "ni hao xiao rong"
# set_KWS = "shuo hua xiao qian"
# set_KWS = "zhan qi lai"

flag_KWS = 0    #标志位，指示唤醒词是否已被触发。

flag_KWS_used = 1
flag_sv_used = 1  #标志唤醒词和声纹识别是否启用。
#1-1 

flag_sv_enroll = 0    #标志声纹识别是否在进行注册（训练）
thred_sv = 0.35      #设置声纹识别的阈值

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)


def extract_chinese_and_convert_to_pinyin(input_string):
    """
    提取字符串中的汉字，并将其转换为拼音。
    
    :param input_string: 原始字符串可以包含汉字、字母、数字等内容
    :return: 转换后的拼音字符串
    """
    # 使用正则表达式提取所有汉字  re.findall返回所有匹配的汉字字符是一个列表
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', input_string)  #`r'[\u4e00-\u9fa5]'`为Unicode 中汉字的范围
    # 将汉字列表合并为字符串
    chinese_text = ''.join(chinese_characters)
    
    # 转换为拼音
    pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
    # 将拼音列表拼接为字符串
    pinyin_text = ' '.join([item[0] for item in pinyin_result])
    
    return pinyin_text


# 音频录制线程
def audio_recorder():
    global audio_queue, recording_active, last_active_time, segments_to_save, last_vad_end_time
    
    p = pyaudio.PyAudio()  #`p`，用于处理音频流
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    audio_buffer = []  #创建一个空的 `audio_buffer` 列表，用于暂存录制的音频数据
    print("音频录制已开始")
    
    while recording_active:
        data = stream.read(CHUNK)
        audio_buffer.append(data)
        
        # 每 0.5 秒检测一次 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            # 拼接音频数据并检测 VAD
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)   #下面定义好的函数
            
            if vad_result:
                print("检测到语音活动")
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            else:
                print("静音中...")
            
            audio_buffer = []  # 清空缓冲区
        
        # 检查无效语音时间
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:  #超出1s
            # 检查是否需要保存
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:  #最后一个音频段的结束时间大于上次 VAD 结束时间
                save_audio_video()  #保存音视频，下面定义好的函数
                last_active_time = time.time()
            else:
                pass
                # print("无新增语音段，跳过保存")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# 视频录制线程
def video_recorder():
    global video_queue, recording_active
    
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    print("视频录制已开始")
    
    while recording_active:
        ret, frame = cap.read()
        if ret:
            video_queue.put((frame, time.time()))
            
            # 实时显示摄像头画面
            cv2.imshow("Real Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 键退出
                break
        else:
            print("无法获取摄像头画面")
    
    cap.release()
    cv2.destroyAllWindows()

# 检测 VAD 活动
def check_vad_activity(audio_data):#大于0.5s的音频
    # 将音频数据分块检测
    num, rate = 0, 0.5
    step = int(AUDIO_RATE * 0.02)  # 20ms 块大小
    flag_rate = round(rate * len(audio_data) // step)  #至少一半有语音活动

    for i in range(0, len(audio_data), step):  #通过步长 `step` 来遍历整个音频数据，每次取 `step` 长度的一块数据作为 `chunk`
        chunk = audio_data[i:i + step]  #一维切片
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    if num > flag_rate:
        return True
    return False

# 保存音频和视频
def save_audio_video():
    pygame.mixer.init()   #初始化 `pygame.mixer` 模块，用于音频播放控制，以便在需要时可以停止当前播放的音频

    global segments_to_save, video_queue, last_vad_end_time, saved_intervals

    # 全局变量，用于保存音频文件名计数
    global audio_file_count
    global flag_sv_enroll
    global set_SV_enroll

    if flag_sv_enroll:
        audio_output_path = f"{set_SV_enroll}/enroll_0.wav"  #注册声纹音频
    else:
        audio_file_count += 1
        audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"  #AI播放音频
    # audio_output_path = f"{OUTPUT_DIR}/audio_0.wav"

    if not segments_to_save:
        return
    
    # 停止当前播放的音频
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        print("检测到新的有效音，已停止当前音频播放")
        
    # 获取有效段的时间范围
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]
    
    # 检查是否与之前的片段重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save.clear()
        return
    
    # 保存音频
    audio_frames = [seg[0] for seg in segments_to_save] #将所有有效音频段（`segments_to_save`）中的音频帧（`seg[0]`）拼接成一个完整的音频数据
    if flag_sv_enroll:
        audio_length = 0.5 * len(segments_to_save)  #因为有音频和时间两个数据？
        if audio_length < 3:
            print("声纹注册语音需大于3秒，请重新注册")
            return 1

    wf = wave.open(audio_output_path, 'wb')  #使用 `wave` 模块创建一个 `.wav` 格式的音频文件
    wf.setnchannels(AUDIO_CHANNELS)  #单通道
    wf.setsampwidth(2)  # 采样宽度16-bit PCM
    wf.setframerate(AUDIO_RATE) #采样率
    wf.writeframes(b''.join(audio_frames))  #将拼接后的音频数据写入文件并关闭文件
    wf.close()
    print(f"音频保存至 {audio_output_path}")

    # Inference()

    if flag_sv_enroll:
        text = "声纹注册完成！现在只有你可以命令我啦！"
        print(text)
        flag_sv_enroll = 0
        system_introduction(text)   #下面有定义好的函数，向系统提供介绍信息
    else:
    # 使用线程执行推理
        inference_thread = threading.Thread(target=Inference, args=(audio_output_path,))   #启动推理线程及保存的音频文件路径
        inference_thread.start()
        
        # 记录保存的区间
        saved_intervals.append((start_time, end_time))
        
    # 清空缓冲区
    segments_to_save.clear()

# --- 播放音频 -
def play_audio(file_path):
    """
    播放音频文件，支持语音中断。
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        # 等待音频播放完成或检测到新的语音活动
        while pygame.mixer.music.get_busy():
            if not recording_active:  # 如果检测到新的语音活动
                pygame.mixer.music.stop()
                break
            time.sleep(0.1)
            
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

async def amain(TEXT, VOICE, OUTPUT_FILE, max_retries=3) -> None:
    """Main function"""
    for retry in range(max_retries):
        try:
            communicate = edge_tts.Communicate(TEXT, VOICE)
            await communicate.save(OUTPUT_FILE)
            return
        except Exception as e:
            print(f"TTS 生成失败 (尝试 {retry + 1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                await asyncio.sleep(1)  # 等待1秒后重试
            else:
                print("TTS 生成失败，将使用默认回复")
                # 使用默认回复
                default_text = "抱歉，语音生成出现问题，请稍后再试。"
                print(default_text)
                return default_text

import os

def is_folder_empty(folder_path):
    """
    检测指定文件夹内是否有文件。
    
    :param folder_path: 文件夹路径
    :return: 如果文件夹为空返回 True，否则返回 False
    """
    # 获取文件夹中的所有条目（文件或子文件夹）
    entries = os.listdir(folder_path)
    # 检查是否存在文件
    for entry in entries:
        # 获取完整路径
        full_path = os.path.join(folder_path, entry)
        # 如果是文件，返回 False
        if os.path.isfile(full_path):
            return False
    # 如果没有文件，返回 True
    return True


# -------- SenceVoice 语音识别 --模型加载-----
model_dir = r"D:\york\GPT\Qwen\pretrained_models\SenseVoiceSmall"
model_senceVoice = AutoModel( model=model_dir, trust_remote_code=True, )

# -------- CAM++声纹识别 -- 模型加载 --------
set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'
sv_pipeline = pipeline(
    task='speaker-verification',
    model='damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

# --------- DeepSeek 大语言模型 ---------------
# 配置量化参数
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_name = r"D:\york\GPT\Qwen\QwenQwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# ---------- 模型加载结束 -----------------------

class ChatMemory:
    def __init__(self, max_length=2048):
        self.history = []  # 存储对话内容
        self.audio_files = []  # 存储语音文件路径
        self.timestamps = []  # 存储时间戳
        self.max_length = max_length  # 最大输入长度
        self.history_file = "chat_history.json"  # 历史记录文件
        self.load_history()  # 加载历史记录

    def add_to_history(self, user_input, model_response, audio_file=None):
        """
        添加用户输入和模型响应到历史记录。
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 确保audio_file是绝对路径
        if audio_file and not os.path.isabs(audio_file):
            audio_file = os.path.abspath(audio_file)
        
        self.history.append({
            "user": user_input,
            "system": model_response,
            "timestamp": timestamp
        })
        self.audio_files.append(audio_file)
        self.timestamps.append(timestamp)
        self.save_history()  # 保存历史记录

    def save_history(self):
        """
        保存对话历史到文件
        """
        try:
            import json
            history_data = {
                "history": self.history,
                "audio_files": self.audio_files,
                "timestamps": self.timestamps
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            print(f"对话历史已保存到 {self.history_file}")
        except Exception as e:
            print(f"保存对话历史时出错: {e}")

    def load_history(self):
        """
        从文件加载对话历史
        """
        try:
            import json
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    self.history = history_data.get("history", [])
                    self.audio_files = history_data.get("audio_files", [])
                    self.timestamps = history_data.get("timestamps", [])
                print(f"已从 {self.history_file} 加载对话历史")
        except Exception as e:
            print(f"加载对话历史时出错: {e}")

    def get_context(self):
        """
        获取拼接后的对话上下文。
        """
        context = ""
        for entry in self.history:
            context += f"User: {entry['user']}\n"
            context += f"system: {entry['system']}\n"
        # 截断上下文，使其不超过 max_length
        if len(context) > self.max_length:
            context = context[-self.max_length:]
        return context
    
    def get_history(self):
        """
        获取完整的历史对话记录。
        """
        return self.history
    
    def get_audio_file(self, index):
        """
        获取指定索引的语音文件路径。
        """
        if 0 <= index < len(self.audio_files):
            audio_file = self.audio_files[index]
            if audio_file and os.path.exists(audio_file):
                return audio_file
            else:
                print(f"音频文件不存在: {audio_file}")
        return None

    def search_history(self, query):
        """
        搜索历史对话记录。
        """
        results = []
        for i, entry in enumerate(self.history):
            if query.lower() in entry['user'].lower() or query.lower() in entry['system'].lower():
                results.append((i, entry))
        return results

    def get_recent_history(self, days=1):
        """
        获取最近几天的历史记录。
        """
        now = time.time()
        recent_history = []
        for i, entry in enumerate(self.history):
            entry_time = time.mktime(time.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S"))
            if (now - entry_time) <= (days * 24 * 60 * 60):
                recent_history.append((i, entry))
        return recent_history

# -------- memory 初始化 --------
memory = ChatMemory(max_length=512)

def system_introduction(text):  #将传入的文本通过语音合成生成语音文件，并播放该语音文件
    global audio_file_count
    global folder_path
    text = text
    print("LLM output:", text)
    used_speaker = "zh-CN-XiaoyiNeural"
    
    try:
        # 尝试生成语音
        asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"sft_tmp_{audio_file_count}.mp3")))
        # 检查文件是否生成成功
        if os.path.exists(os.path.join(folder_path,f"sft_tmp_{audio_file_count}.mp3")):
            play_audio(f'{folder_path}/sft_tmp_{audio_file_count}.mp3')
        else:
            print("语音文件生成失败，将使用文本输出")
            print(text)
    except Exception as e:
        print(f"语音生成出错: {str(e)}")
        print("将使用文本输出")
        print(text)

def get_voice_input(max_retries=3):
    """
    获取用户语音输入并识别
    :param max_retries: 最大重试次数
    :return: 识别到的文本
    """
    global audio_file_count
    
    for retry in range(max_retries):
        try:
            print("请说出您的选择...")
            # 等待新的语音输入
            while True:
                if not recording_active:
                    continue
                # 检查是否有新的音频段需要保存
                if segments_to_save:
                    # 保存音频文件
                    audio_file = f"{OUTPUT_DIR}/user_input_{audio_file_count}.wav"
                    save_audio_video()
                    # 进行语音识别
                    res = model_senceVoice.generate(
                        input=audio_file,
                        cache={},
                        language="auto",
                        use_itn=False,
                    )
                    if res and res[0]['text']:
                        recognized_text = res[0]['text'].split(">")[-1]
                        print(f"识别结果: {recognized_text}")
                        return recognized_text
                time.sleep(0.1)
        except Exception as e:
            print(f"语音识别出错: {str(e)}")
            if retry < max_retries - 1:
                text = "抱歉，我没有听清楚，请再说一遍"
                print(text)
                asyncio.run(amain(text, "zh-CN-XiaoyiNeural", 
                         os.path.join(folder_path,f"retry_{audio_file_count}.mp3")))
                play_audio(f'{folder_path}/retry_{audio_file_count}.mp3')
            else:
                text = "抱歉，多次尝试后仍然无法识别，请稍后再试"
                print(text)
                asyncio.run(amain(text, "zh-CN-XiaoyiNeural", 
                         os.path.join(folder_path,f"error_{audio_file_count}.mp3")))
                play_audio(f'{folder_path}/error_{audio_file_count}.mp3')
                return None
    
    return None

def handle_history_query(prompt):
    """
    处理历史对话查询和语音播放。
    """
    global memory
    
    # 获取相关历史对话
    results = memory.search_history(prompt)
    
    if not results:
        text = "未找到相关历史对话记录"
        print(text)
        asyncio.run(amain(text, "zh-CN-XiaoyiNeural", os.path.join(folder_path,f"no_result_{audio_file_count}.mp3")))
        play_audio(f'{folder_path}/no_result_{audio_file_count}.mp3')
        return
    
    # 显示搜索结果
    print(f"\n找到{len(results)}条相关对话：")
    text = f"找到{len(results)}条相关对话，让我为您播放每条对话的内容："
    print(text)
    asyncio.run(amain(text, "zh-CN-XiaoyiNeural", os.path.join(folder_path,f"found_results_{audio_file_count}.mp3")))
    play_audio(f'{folder_path}/found_results_{audio_file_count}.mp3')
    
    # 播放每条对话
    for i, (index, entry) in enumerate(results):
        # 播放对话内容
        text = f"第{i+1}条对话：用户说：{entry['user']}，系统回复：{entry['system']}"
        print(text)
        asyncio.run(amain(text, "zh-CN-XiaoyiNeural", os.path.join(folder_path,f"entry_{i}_{audio_file_count}.mp3")))
        play_audio(f'{folder_path}/entry_{i}_{audio_file_count}.mp3')
        
        # 获取并播放原始音频
        audio_file = memory.get_audio_file(index)
        if audio_file:
            text = "要播放这条对话的原始语音吗？请说'是'或'否'"
            print(text)
            asyncio.run(amain(text, "zh-CN-XiaoyiNeural", os.path.join(folder_path,f"ask_play_{i}_{audio_file_count}.mp3")))
            play_audio(f'{folder_path}/ask_play_{i}_{audio_file_count}.mp3')
            
            # 获取用户选择
            play_choice = get_voice_input()
            if play_choice and any(keyword in play_choice for keyword in ["是", "好的", "要", "播放", "可以"]):
                print(f"正在播放音频: {audio_file}")
                play_audio(audio_file)
    
    # 结束提示
    text = "所有对话已播放完毕，查询结束"
    print(text)
    asyncio.run(amain(text, "zh-CN-XiaoyiNeural", os.path.join(folder_path,f"end_query_{audio_file_count}.mp3")))
    play_audio(f'{folder_path}/end_query_{audio_file_count}.mp3')

# 初始化RAG管理器
rag_manager = RAGManager()

def Inference(TEMP_AUDIO_FILE=f"{OUTPUT_DIR}/audio_0.wav"):
    print("开始推理过程...")
    global audio_file_count
    global set_SV_enroll
    global flag_sv_enroll
    global thred_sv
    global flag_sv_used
    global set_KWS
    global flag_KWS
    global flag_KWS_used
    
    os.makedirs(set_SV_enroll, exist_ok=True)
    if flag_sv_used and is_folder_empty(set_SV_enroll):
        print("检测到无声纹注册文件")
        text = f"无声纹注册文件！请先注册声纹，需大于三秒哦~"
        print(text)
        system_introduction(text)
        flag_sv_enroll = 1
    
    else:
        print("开始语音识别...")
        # -------- SenceVoice 推理 ---------
        input_file = (TEMP_AUDIO_FILE)
        res = model_senceVoice.generate(
            input=input_file,
            cache={},
            language="auto",
            use_itn=False,
        )
        print("语音识别完成，结果：", res)
        prompt = res[0]['text'].split(">")[-1]
        prompt_pinyin = extract_chinese_and_convert_to_pinyin(prompt)
        print("转换后的拼音：", prompt_pinyin)

        # --- 判断是否启动KWS
        if not flag_KWS_used:
            flag_KWS = 1
        if not flag_KWS:
            if set_KWS in prompt_pinyin:
                print("检测到唤醒词")
                flag_KWS = 1
                # 过滤掉唤醒词
                prompt = prompt.replace("你好小荣", "").strip()
                if not prompt:  # 如果过滤后为空，则等待新的输入
                    text = "请说出您的需求"
                    print(text)
                    system_introduction(text)
                    return
        
        # --- KWS成功，或不设置KWS
        if flag_KWS:
            print("开始声纹验证...")
            sv_score = sv_pipeline([os.path.join(set_SV_enroll, "enroll_0.wav"), TEMP_AUDIO_FILE], thr=thred_sv)
            print("声纹验证结果：", sv_score)
            sv_result = sv_score['text']
            if sv_result == "yes":
                print("声纹验证通过，开始大模型推理...")
                
                # 检查是否是历史对话查询
                history_keywords = ["历史对话", "历史记录", "昨天", "前天", "过去", "上次", "之前"]
                if any(keyword in prompt for keyword in history_keywords):
                    handle_history_query(prompt)
                    return
                
                # 使用RAG生成增强的提示词
                enhanced_prompt = rag_manager.generate_prompt_with_context(prompt)
                
                # 检索相关历史对话
                relevant_history = rag_manager.retrieve_relevant_history(prompt)
                
                # 构建历史对话上下文
                history_context = ""
                if relevant_history:
                    history_context = "\n以下是相关的历史对话：\n"
                    for conv in relevant_history:
                        history_context += f"用户: {conv['user_input']}\n"
                        history_context += f"系统: {conv['system_response']}\n\n"
                
                # Qwen2.5 对话格式
                messages = [
                    {"role": "system", "content": "你叫小荣，是一个无所不知的家庭助手，性格古灵精怪又不失温柔体贴，说话幽默风趣，被纠正错误会道歉并改正，回答问题越详细越好。"}
                ]
                
                # 检查是否包含历史查询关键词
                history_keywords = ["之前", "过去", "历史", "上次", "以前", "昨天", "前天"]
                is_history_query = any(keyword in enhanced_prompt for keyword in history_keywords)
                
                if is_history_query and relevant_history:
                    # 如果是历史查询，添加相关历史对话
                    for conv in relevant_history:
                        messages.append({"role": "user", "content": conv['user_input']})
                        messages.append({"role": "assistant", "content": conv['system_response']})
                
                # 添加当前用户输入
                messages.append({"role": "user", "content": enhanced_prompt})
                
                print("正在生成回复...")
                # 使用 Qwen2.5 的对话模板
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2048,  # 增加生成长度
                    do_sample=True,
                    temperature=0.7,  # 提高温度，使输出更有创造性
                    top_p=0.9,  # 提高top_p，增加多样性
                    repetition_penalty=1.1,  # 降低重复惩罚
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
                print("生成完成，回复内容：", response)
                
                # 提取回复内容
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0].strip()
                
                # 去除特殊符号
                special_chars = ['#', '*', '<|im_start|>', '<|im_end|>']
                for char in special_chars:
                    response = response.replace(char, '')
                response = response.strip()
                
                print("最终回复内容：", response)

                # 将对话添加到RAG系统
                rag_manager.add_conversation(prompt, response, TEMP_AUDIO_FILE)

                # 输入文本
                text = response
                # 语种识别
                language, confidence = langid.classify(text)
                print("检测到语言：", language)
                
                language_speaker = {
                    "ja": "ja-JP-NanamiNeural",
                    "fr": "fr-FR-DeniseNeural",
                    "es": "ca-ES-JoanaNeural",
                    "de": "de-DE-KatjaNeural",
                    "zh": "zh-CN-XiaoyiNeural",
                    "en": "en-US-AnaNeural",
                }

                if language not in language_speaker.keys():
                    used_speaker = "zh-CN-XiaoyiNeural"
                else:
                    used_speaker = language_speaker[language]
                    print("使用音色：", language_speaker[language])

                print("开始语音合成...")
                asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"sft_{audio_file_count}.mp3")))
                print("开始播放语音...")
                play_audio(f'{folder_path}/sft_{audio_file_count}.mp3')
            else:
                print("声纹验证失败")
                text = "很抱歉，声纹验证失败，我无法为您服务"
                print(text)
        else:
            print("未检测到唤醒词")
            text = "很抱歉，唤醒词错误，请说出正确的唤醒词哦"
            system_introduction(text)

# 主函数
if __name__ == "__main__":

    try:
        # 启动音视频录制线程
        audio_thread = threading.Thread(target=audio_recorder)
        # video_thread = threading.Thread(target=video_recorder)
        audio_thread.start()
        # video_thread.start()

        flag_info = f'{flag_sv_used}-{flag_KWS_used}'
        dict_flag_info = {
            "1-1": "您已开启声纹识别和关键词唤醒，",
            "0-1":"您已开启关键词唤醒",
            "1-0":"您已开启声纹识别",
            "0-0":"",
        }
        if flag_sv_used or flag_KWS_used:
            text = dict_flag_info[flag_info]
            system_introduction(text)

        print("按 Ctrl+C 停止录制")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("录制停止中...")
        recording_active = False
        audio_thread.join()
        # video_thread.join()
        print("录制已停止")
