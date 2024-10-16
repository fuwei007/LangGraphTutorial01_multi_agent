import base64
import json
from datetime import datetime
import os
import time
import uuid
from urllib.parse import urlencode

from dotenv import load_dotenv, find_dotenv  # 从 dotenv 库导入环境变量加载和查找函数

from util.file_util import FileUtils

load_dotenv(find_dotenv())  # 加载环境变量
from langchain_openai import ChatOpenAI  # 从 langchain_openai 库导入 ChatOpenAI 类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入聊天提示模板和消息占位符

# supervisor_agent

## 定义大型语言模型（LLM）
llm = ChatOpenAI(model="gpt-4o")  # 实例化 ChatOpenAI 对象

## 定义具有结构化输出的提示

### 定义提示
agent_online_search_name = 'agent_online_search'  # 定义在线搜索代理的名称
agent_online_extract_name = 'agent_online_extract'  # 定义在线提取数据代理的名称
agent_email_send_name = 'agent_email_send'  # 定义发送邮件代理的名称

# text -> multiple data
agent_generate_image_name = 'generate_image'  # 文字生成图片
agent_generate_audio_name = 'generate_audio'  # 文字生成音频
agent_generate_digital_human_video_name = 'generate_digital_human_video'  # 文字生成数字人

# image -> multiple data
agent_extract_text_from_image_name = 'extract_text_from_image'  # 从图片中提取出文字
agent_extract_content_information_from_image_name = 'extract_content_information_from_image'  # 描述图片内容
agent_remove_image_background_name = 'remove_image_background'  # 去除图片的背景图像

# audio -> multiple data
agent_transcribe_audio_name = 'transcribe_audio'  # 音频转成文字
agent_convert_audio_language_name = 'convert_audio_language'  # 音频转换成另一种语言音频

agent_lst = [agent_online_search_name, agent_online_extract_name, agent_email_send_name, agent_generate_image_name,
             agent_generate_audio_name, agent_generate_digital_human_video_name, agent_extract_text_from_image_name,
             agent_extract_content_information_from_image_name, agent_remove_image_background_name,
             agent_transcribe_audio_name, agent_convert_audio_language_name]  # 将代理名称存入列表
system_prompt = (  # 定义系统提示
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {agent_lst}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options = ['FINISH'] + agent_lst  # 定义可选项，包含 FINISH 和代理列表

# 创建聊天提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),  # 系统消息，包含提示内容
        MessagesPlaceholder(variable_name='messages'),  # 占位符，用于动态插入消息
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}"  # 提问，确定下一个执行的代理或结束
        ),
    ]
).partial(options=str(options), agent_lst=", ".join(agent_lst))  # 使用选项和代理列表填充模板

### 定义结构化输出
from pydantic import BaseModel  # 从 pydantic 导入 BaseModel，用于数据验证
from typing import Literal  # 从 typing 导入 Literal，用于定义固定值类型


class route_response(BaseModel):  # 定义路由响应模型
    next: Literal[
        'FINISH', 'agent_online_search', 'agent_online_extract', 'agent_email_send', 'generate_image', 'generate_audio',
        'generate_digital_human_video','extract_text_from_image', 'extract_content_information_from_image',
        'remove_image_background','transcribe_audio', 'convert_audio_language']  # 下一个执行的代理或结束


## 获取代理
def supervisor_node(state):  # 定义 supervisor_node 函数，接收状态作为参数
    chain = (
            prompt | llm.with_structured_output(route_response)  # 连接提示和 LLM，使用结构化输出
    )
    return chain.invoke(state)  # 调用链并返回结果


# 创建工作代理
from langgraph.prebuilt import create_react_agent  # 从 langgraph 导入创建反应代理的函数

from langchain_core.messages import HumanMessage  # 从 langchain_core 导入人类消息类
# from .tool import tool_tavily, tool_send_email  # 导入工具（被注释掉）
from util import tool
import requests


# 搜索数据
def research_node(state):  # 定义 research_node 函数，接收状态作为参数
    name = 'agent_online_search'  # 定义代理名称
    agent = create_react_agent(llm, tools=[tool.tool_tavily])  # 创建反应代理，传入工具
    result = agent.invoke(state)  # 调用代理并获取结果
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}  # 返回最后一条消息


# 提取数据
def extract_node(state):  # 定义 extract_node 函数，接收状态作为参数
    name = 'agent_online_extract'  # 定义代理名称

    class extract_parameter(BaseModel):  # 定义用于提取url参数的模型
        url: str  # 链接

    extract_prompt = ChatPromptTemplate.from_messages(
        [
            'system', "Extract the URL of the webpage from which the user wants to extract data.",  # 系统消息
            MessagesPlaceholder(variable_name='messages'),  # 占位符，用于动态插入消息
        ]
    )

    # 创建处理邮件的链，将提示与 LLM 结合，期望结构化输出
    chain = (extract_prompt | llm.with_structured_output(extract_parameter))
    result = chain.invoke(state)  # 执行链并获取结果

    print(result.url)
    response = tool.tavily_client.extract(urls=[result.url])
    # 处理结果并生成消息
    print(response)

    extracted_data = response['results'][0]  # 获取第一个结果
    raw_content = extracted_data.get('raw_content', '未找到数据')
    # 去除换行符并替换反斜杠为空格
    clean_content = raw_content.replace('\n', ' ').replace('\\', ' ')
    # 处理结果并生成消息
    print(clean_content)

    return {"messages": [HumanMessage(content="提取网页数据如下："+clean_content, name=name)]}  # 返回消息


# 发送邮件
def email_node(state):  # 定义发送邮件的节点函数，接受状态作为参数
    name = 'agent_email_send'  # 定义代理名称为 'agent_email_send'

    class email_parameter(BaseModel):  # 定义用于存储邮件参数的模型
        subject: str  # 邮件主题
        body: str  # 邮件内容
        url: str  # 链接

    # 创建邮件提示模板
    email_prompt = ChatPromptTemplate.from_messages(
        [
            'system', "Extract the email subject and body for me based on the input:",  # 系统消息，要求提取邮件主题和内容
            MessagesPlaceholder(variable_name='messages'),  # 占位符，用于动态插入消息
        ]
    )

    # 创建处理邮件的链，将提示与 LLM 结合，期望结构化输出
    chain = (email_prompt | llm.with_structured_output(email_parameter))
    result = chain.invoke(state)  # 执行链并获取结果

    # 发送邮件，使用提取的主题和内容，收件人是指定邮箱
    tool.send_email(subject=result.subject, body=result.body + result.url, to_email="663800595@qq.com")

    # 返回发送成功的消息
    return {"messages": [HumanMessage(content="Information has been emailed to you", name=name)]}


# 文字生成图片
def generate_image_node(state):
    name = 'generate_image'

    class image_parameter(BaseModel):  # 定义用于存储语音参数的模型
        prompt: str  # 转录内容

    # 创建提示模板
    image_prompt = ChatPromptTemplate.from_messages(
        [
            'system',"Based on the number of images already painted, extract the prompt words for this painting.",
            MessagesPlaceholder(variable_name='messages'),
        ]
    )

    # 创建处理链，将提示与 LLM 结合，期望结构化输出
    chain = image_prompt | llm.with_structured_output(image_parameter)
    result = chain.invoke(state)  # 执行链并获取结果


    url = "https://api.openai.com/v1/images/generations"

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }


    # 准备请求体
    payload = {
        "prompt": result.prompt,
        "n": 1,
        "size": "1024x1024"
    }

    try:
        print(payload)
        # 调用生成图片 API
        response = requests.post(url, json=payload, headers=headers)

        # 提取图像URL
        image_url = response.json()['data'][0]['url']

        # 下载图片
        image_response = requests.get(image_url)
        image_response.raise_for_status()  # 检查下载是否成功

        # 保存图片到本地
        image_bytes = image_response.content
        file_name = f"{FileUtils.random_uuid()}.png"
        folder = f"uploadFiles/{FileUtils.get_folder().replace(' ', '')}/"
        file_path = os.path.join(os.getenv('FILE_ROOT_PATH'), folder)

        # 创建目录（如果不存在）
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # 写入图片到本地文件
        FileUtils.upload_file(image_bytes, file_path, file_name)

        # 返回本地 URL
        relative_path = os.path.join(folder, file_name)
        file_url = f"{os.getenv('BASE_URL')}/{relative_path}"

        print(file_url)

        return {"messages": [HumanMessage(content="一张"+result.prompt+"图片生成如下："+file_url, name=name)]}

    except requests.exceptions.RequestException as e:
        return {"messages": [HumanMessage(content=f"请求失败: {str(e)}", name=name)]}


# 文字生成音频
def generate_audio_node(state):
    name = 'generate_audio'

    class voice_parameter(BaseModel):  # 定义用于存储语音参数的模型
        content: str  # 转录内容
        language: str  # 语言选择
        voice_name: str  # 角色选择

    # 创建提示模板
    voice_prompt = ChatPromptTemplate.from_messages(
        [
            'system',
            "Extract the text content that the user wants to convert to audio（Need to process the text that needs to be translated into the target language.）, along with the target language "
            "(e.g., en-US, zh-CN,ja-JP,fr-FR,de-DE, ru-RU,ko-KR and so on), and the appropriate OpenAI TTS API corresponding language code "
            "(such as alloy, echo, fable, onyx, nova, shimmer).",
            MessagesPlaceholder(variable_name='messages'),
        ]
    )

    # 创建处理链，将提示与 LLM 结合，期望结构化输出
    chain = voice_prompt | llm.with_structured_output(voice_parameter)
    result = chain.invoke(state)  # 执行链并获取结果

    """调用 OpenAI TTS API 将文字转换为音频"""
    openai_url = "https://api.openai.com/v1/audio/speech"

    # 构建请求体
    request_body = {
        "model": "tts-1",
        "input": result.content,
        "voice": result.voice_name
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",  # 从环境变量获取API密钥
        "Content-Type": "application/json"
    }

    # 发起POST请求到 OpenAI API
    try:
        response = requests.post(openai_url, headers=headers, json=request_body)
        response.raise_for_status()  # 检查请求是否成功

        # 获取音频内容
        audio_bytes = response.content  # 直接获取音频内容

        # 保存音频到本地
        file_name = f"{FileUtils.random_uuid()}.mp3"  # 假设音频格式为 mp3
        folder = f"uploadFiles/{FileUtils.get_folder().replace(' ', '')}/"
        file_path = os.path.join(os.getenv('FILE_ROOT_PATH'), folder)

        # 创建目录（如果不存在）
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # 写入音频到本地文件
        FileUtils.upload_file(audio_bytes, file_path, file_name)

        # 返回本地 URL
        relative_path = os.path.join(folder, file_name)
        file_url = f"{os.getenv('BASE_URL')}/{relative_path}"

        print(file_url)

        return {"messages": [HumanMessage(content="文字生成音频如下："+file_url, name=name)]}

    except requests.exceptions.RequestException as e:
        print(f"Error generating audio: {e}")
        return {"messages": [HumanMessage(content=f"请求失败: {str(e)}", name=name)]}


# 文字生成数字人视频
def generate_digital_human_video_node(state):
    name = 'generate_digital_human_video'


    api_key = "amluZnVsYWlrZWppQGdtYWlsLmNvbQ"  # 替换为你的 D-ID API 密钥
    api_secret = "089urJTyeCehlpoINVyOH"  # 替换为你的 D-ID API 密钥密码
    endpoint = "https://api.d-id.com/v1/talks"

    # 创建请求体
    request_body = {
        "script": {
            "type": "text",
            "input": "text",
            "voice": "voice",
            "language": "language"
        }
    }

    # 创建 Authorization 头
    credentials = f"{api_key}={api_secret}"

    # 发送 POST 请求
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        data=json.dumps(request_body)
    )

    # 检查响应状态码
    if response.status_code == 200:
        json_response = response.json()
        video_url = json_response.get("video_url")  # 根据文档确认返回的字段
        print(video_url)
        return {"messages": [HumanMessage(content=video_url, name=name)]}
    else:
        raise Exception(f"请求失败: {response.status_code}, {response.text}")

# 从图片中提取文字
def extract_text_from_image_node(state):
    name = 'extract_text_from_image'

    class image_parameter(BaseModel):  # 定义用于存储图像参数的模型
        url: str  # 图片url路径

    # 创建提示模板
    image_prompt = ChatPromptTemplate.from_messages(
        [
            'system', "Extract the URL path of the image from which the user wants to recognize text.",
            MessagesPlaceholder(variable_name='messages'),
        ]
    )

    # 创建处理链，将提示与 LLM 结合，期望结构化输出
    chain = image_prompt | llm.with_structured_output(image_parameter)
    result = chain.invoke(state)  # 执行链并获取结果

    endpoint_url = "https://imageextracttext.cognitiveservices.azure.com/vision/v3.2/read/analyze"

    headers = {
        "Ocp-Apim-Subscription-Key": os.getenv('AZURE_API_KEY'),
        "Content-Type": "application/json"
    }
    request_body = {
        "url": result.url
    }

    try:
        # 发送 POST 请求
        response = requests.post(endpoint_url, headers=headers, json=request_body)

        # 获取 Operation-Location 响应头
        operation_location = response.headers.get("Operation-Location")
        if operation_location is None:
            return {"messages": [HumanMessage(content="未获取到操作位置头", name=name)]}

        # 轮询获取结果
        is_completed = False
        json_response = ""

        while not is_completed:
            time.sleep(1)  # 等待1秒钟后再次查询
            result_response = requests.get(operation_location, headers=headers)
            json_response = result_response.json()
            status = json_response.get("status")

            if status == "succeeded":
                is_completed = True
            elif status == "failed":
                return {"messages": [HumanMessage(content="文本提取失败", name=name)]}

        # 提取识别结果
        extracted_text = ""
        if 'analyzeResult' in json_response:
            for line in json_response['analyzeResult']['readResults']:
                for word in line['lines']:
                    extracted_text += word['text'] + "\n"

        print(extracted_text.strip())
        return {"messages": [HumanMessage(content="图片中提取文字如下："+extracted_text.strip(), name=name)]}

    except requests.exceptions.RequestException as e:
        return {"messages": [HumanMessage(content=f"请求失败: {str(e)}", name=name)]}


# 描述图片内容
def extract_content_information_from_image_node(state):
    name = 'extract_content_information_from_image'

    class image_parameter(BaseModel):  # 定义用于存储图像参数的模型
        prompt: str  # 用户想要从图片中提取的内容
        url: str  # 图片url路径

    # 创建提示模板
    image_prompt = ChatPromptTemplate.from_messages(
        [
            'system', "Extract the content that the user wants to extract from the image, "
                      "along with the URL path of the image.",
            MessagesPlaceholder(variable_name='messages'),
        ]
    )

    # 创建处理链，将提示与 LLM 结合，期望结构化输出
    chain = image_prompt | llm.with_structured_output(image_parameter)
    result = chain.invoke(state)  # 执行链并获取结果

    azure_endpoint_url = "https://imageextracttext.cognitiveservices.azure.com/vision/v3.1/describe"
    params = "?visualFeatures=Description&details=Celebrities,Landmarks&language=en&maxCandidates=3"

    # 调用 Azure API 获取基础描述
    azure_request_body = {
        "url": result.url
    }

    azure_headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": os.getenv('AZURE_API_KEY')
    }

    try:
        response = requests.post(azure_endpoint_url + params, headers=azure_headers, json=azure_request_body)
        response.raise_for_status()  # 检查请求是否成功

        azure_response = response.json()
        safe_azure_response = str(azure_response).replace("\"", "\\\"")  # 转义引号
        # basic_description = azure_response.json()['description']['captions'][0]['text']

        print(f"Basic Description from Azure: {safe_azure_response}")
        print(result.prompt)

        # 调用 OpenAI GPT API 进一步丰富描述
        openai_endpoint = "https://api.openai.com/v1/chat/completions"
        openai_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }

        openai_request_body = {
            "model": "gpt-3.5-turbo",  # 或者根据需要选择模型
            "messages": [{"role": "user",
                          "content": "根据以下图片描述生成更详细的描述:" + safe_azure_response + ".。请着重于 " + result.prompt}],
            # "stream": True  # 启用流式返回
        }

        # 流式响应
        # openai_response = requests.post(openai_endpoint, headers=openai_headers, json=openai_request_body, stream=True)
        #
        # # 处理 OpenAI 的流式返回
        # for line in openai_response.iter_lines():
        #     if line:
        #         print(line.decode("utf-8"))  # 处理接收到的每一行

        # 非流式响应
        openai_response = requests.post(openai_endpoint, headers=openai_headers, json=openai_request_body)
        openai_response.raise_for_status()  # 检查请求是否成功

        result_description = openai_response.json()['choices'][0]['message']['content']
        print(f"Enhanced Description from OpenAI: {result_description}")
        return {"messages": [HumanMessage(content="图片内容如下:"+result_description, name=name)]}

    except requests.exceptions.RequestException as e:
        return {"messages": [HumanMessage(content=f"请求失败: {str(e)}", name=name)]}


# 去除图片背景
def remove_image_background_node(state):
    name = 'remove_image_background'

    class ImageParameter(BaseModel):  # 定义用于存储图像参数的模型
        url: str  # 图片url路径

    image_prompt = ChatPromptTemplate.from_messages(
        [
            'system', "Extract the URL path of the image from which the user wants to remove the background.",
            MessagesPlaceholder(variable_name='messages'),
        ]
    )

    # 创建处理链，将提示与 LLM 结合，期望结构化输出
    chain = image_prompt | llm.with_structured_output(ImageParameter)
    result = chain.invoke(state)  # 执行链并获取结果

    endpoint_url = "https://api.remove.bg/v1.0/removebg"

    headers = {
        "X-Api-Key": os.getenv('REMOVE_API_KEY'),
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # 准备请求体，发送图像URL
    request_body = {
        "image_url": result.url,
        "size": "auto"
    }

    try:
        # 发送POST请求
        response = requests.post(endpoint_url, headers=headers, data=urlencode(request_body))

        # 检查是否成功处理
        if response.status_code == 200:
            # 读取返回的图像数据
            image_bytes = response.content

            # 保存图像到本地
            file_name = f"{FileUtils.random_uuid()}.png"  # 假设图像格式为 png
            folder = f"uploadFiles/{FileUtils.get_folder().replace(' ', '')}/"
            file_path = os.path.join(os.getenv('FILE_ROOT_PATH'), folder)

            # 创建目录（如果不存在）
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            # 保存文件
            FileUtils.upload_file(image_bytes, file_path, file_name)

            # 返回本地 URL
            relative_path = os.path.join(folder, file_name)
            file_url = f"{os.getenv('BASE_URL')}/{relative_path}"

            print(file_url)
            return {"messages": [HumanMessage(content="完成去除图片背景如下："+file_url, name=name)]}
        else:
            print("Error:", response.status_code, response.text)
            return {"messages": [HumanMessage(content=f"请求失败: {response.text}", name=name)]}

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)}")
        return {"messages": [HumanMessage(content=f"请求失败: {str(e)}", name=name)]}


# 音频转文字
def transcribe_audio_node(state):
    name = 'transcribe_audio_node'

    class audio_parameter(BaseModel):  # 定义用于存储图像参数的模型
        url: str  # 图片url路径

    audio_prompt = ChatPromptTemplate.from_messages(
        [
            'system', "Extract the URL path of the audio that the user wants to transcribe.",
            MessagesPlaceholder(variable_name='messages'),
        ]
    )

    # 创建处理链，将提示与 LLM 结合，期望结构化输出
    chain = audio_prompt | llm.with_structured_output(audio_parameter)
    result = chain.invoke(state)  # 执行链并获取结果

    endpoint = "https://api.openai.com/v1/audio/transcriptions"  # OpenAI 的 Speech-to-Text API Endpoint

    # 准备POST请求
    files = {
        'file': (os.path.basename(result.url), requests.get(result.url).content, 'audio/mpeg'),
    }
    data = {
        'model': 'whisper-1',  # 这里指定模型
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    try:
        # 发送POST请求
        response = requests.post(endpoint, headers=headers, files=files, data=data)

        # 检查请求是否成功
        response.raise_for_status()
        print(response.json().get('text', ''))
        return {"messages": [HumanMessage(content="完成音频转文字如下："+response.json().get('text', ''), name=name)]}

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)}")
        return {"messages": [HumanMessage(content=f"请求失败: {str(e)}", name=name)]}


# 将音频转换成另一种语言的音频
def convert_audio_language_node(state):
    name = "convert_audio_language"

    # ----------------------------------------------------------------------------------------------------
    # 步骤 1: 将音频转录为文本
    class audio_parameter(BaseModel):  # 定义用于存储图像参数的模型
        language: str  # 目标语言代码
        voice_name: str  # 角色代码
        url: str  # 音频url路径

    audio_prompt = ChatPromptTemplate.from_messages(
        [
            'system', "Extract the URL address of the audio that the user wants to convert to text, "
                      "along with the target language (for example, en-US, zh-CN, ja-JP, fr-FR, de-DE, ru-RU, ko-KR, etc.),"
                      "along with the target voice (Choose the right one from the following: alloy, echo, fable, onyx, nova, shimmer).",
            MessagesPlaceholder(variable_name='messages'),
        ]
    )

    # 创建处理链，将提示与 LLM 结合，期望结构化输出
    chain = audio_prompt | llm.with_structured_output(audio_parameter)
    result = chain.invoke(state)  # 执行链并获取结果

    endpoint = "https://api.openai.com/v1/audio/transcriptions"  # OpenAI 的 Speech-to-Text API Endpoint

    # 准备POST请求
    files = {
        'file': (os.path.basename(result.url), requests.get(result.url).content, 'audio/mpeg'),
    }
    data = {
        'model': 'whisper-1',  # 这里指定模型
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    # 发送POST请求
    response = requests.post(endpoint, headers=headers, files=files, data=data)

    # 检查请求是否成功
    response.raise_for_status()
    print(response.json().get('text', ''))

    # ----------------------------------------------------------------------------------------------------
    # 步骤 2: 使用 GPT 将转录文本翻译为目标语言
    transcribed_text = response.json().get('text', '')

    openai_url = "https://api.openai.com/v1/chat/completions"

    # 构建请求体
    request_body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": f"Translate this text into {result.language}. Return only the translated result with no other replies."
            },
            {
                "role": "user",
                "content": transcribed_text
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    # 发起POST请求
    translation_response = requests.post(openai_url, headers=headers, json=request_body)
    translation_response.raise_for_status()
    translated_text = translation_response.json().get('choices', [{}])[0].get('message', {}).get('content', '')

    print(translated_text)
    # ----------------------------------------------------------------------------------------------------
    # 步骤 3: 使用 GPT 将文本转录成音频
    """调用 OpenAI TTS API 将文字转换为音频"""
    openai_url = "https://api.openai.com/v1/audio/speech"

    # 构建请求体
    request_body = {
        "model": "tts-1",
        "input": translated_text,
        "voice": result.voice_name
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",  # 从环境变量获取API密钥
        "Content-Type": "application/json"
    }

    # 发起POST请求到 OpenAI API
    try:
        response = requests.post(openai_url, headers=headers, json=request_body)
        response.raise_for_status()  # 检查请求是否成功

        # 获取音频内容
        audio_bytes = response.content  # 直接获取音频内容

        # 保存音频到本地
        file_name = f"{FileUtils.random_uuid()}.mp3"  # 假设音频格式为 mp3
        folder = f"uploadFiles/{FileUtils.get_folder().replace(' ', '')}/"
        file_path = os.path.join(os.getenv('FILE_ROOT_PATH'), folder)

        # 创建目录（如果不存在）
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # 写入音频到本地文件
        FileUtils.upload_file(audio_bytes, file_path, file_name)

        # 返回本地 URL
        relative_path = os.path.join(folder, file_name)
        file_url = f"{os.getenv('BASE_URL')}/{relative_path}"
        print(file_url)

        return {"messages": [HumanMessage(content="完成音频转换成另一种语言的音频如下："+file_url, name=name)]}

    except requests.exceptions.RequestException as e:
        print(f"Error generating audio: {e}")
        return {"messages": [HumanMessage(content=f"请求失败: {str(e)}", name=name)]}

