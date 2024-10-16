import os  # 导入操作系统模块，用于访问环境变量

import requests
from dotenv import load_dotenv, find_dotenv  # 从 dotenv 库导入加载和查找环境变量的函数
load_dotenv(find_dotenv())  # 加载环境变量
from langchain_core.tools import BaseTool  # 从 langchain_core 导入基础工具类


# 搜索工具
from langchain_community.tools.tavily_search import TavilySearchResults  # 导入 TavilySearchResults 类
tool_tavily = TavilySearchResults(max_results=5)  # 实例化搜索工具，限制最大结果数为 5
from tavily import TavilyClient
# 初始化提取工具，限制最大结果数为 5
tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))


# 邮件工具
import smtplib  # 导入 SMTP 库，用于发送邮件
from email.mime.text import MIMEText  # 从 email.mime.text 导入 MIMEText，用于构建邮件内容

def send_email(subject, body, to_email):  # 定义发送邮件的函数，接受主题、内容和收件人邮箱
    # SMTP 服务器配置
    smtp_server = 'smtp.qq.com'  # SMTP 服务器地址
    smtp_port = 465  # 通常用于 SSL 的端口
    sender_email = '663800595@qq.com'  # 发件人邮箱
    # password = 'khjxmpulqjbkbiei'  # 小心处理密码安全
    password = os.environ['EMAIL_CODE']  # 从环境变量中获取密码，确保密码安全


    # 创建邮件内容
    message = MIMEText(body, 'plain')  # 创建纯文本邮件
    message['From'] = sender_email  # 设置发件人
    message['To'] = to_email  # 设置收件人
    message['Subject'] = subject  # 设置邮件主题

    # 发送邮件
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)  # 使用 SSL 创建 SMTP 服务器连接
        server.login(sender_email, password)  # 登录到邮件服务器
        server.sendmail(sender_email, to_email, message.as_string())  # 发送邮件
        server.quit()  # 退出服务器
        print("Email sent successfully!")  # 打印发送成功的消息
    except Exception as e:  # 捕捉异常
        print(f"Failed to send email: {e}")  # 打印错误信息


def create_openai_connection(url, json_request_body):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"  # 从环境变量中获取API密钥
    }

    # 发送POST请求
    response = requests.post(url, json=json_request_body, headers=headers)
    response.raise_for_status()  # 检查响应状态
    return response.json()