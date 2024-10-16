import io
import operator  # 引入运算符模块，主要用于处理序列操作
import sys
from typing import Sequence, TypedDict, Annotated  # 从 typing 导入类型注解
from langchain_core.messages import BaseMessage  # 导入基础消息类

from langgraph.graph import END, START, StateGraph  # 从 langgraph 导入图的相关常量和类

import agent  # 导入自定义代理模块


# 定义代理状态的类型
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 消息列表，使用运算符进行添加
    next: str  # 下一个要执行的代理名称


# 定义有向无环图（DAG）
# 创建工作流图
supervisor_name = "supervisor"  # 定义主管的名称

workflow = StateGraph(AgentState)  # 实例化状态图，传入代理状态类型

# 定义代理名称和描述
agent_descriptions = {
    'agent_online_search': '执行在线搜索任务',
    'agent_online_extract': '从在线资源提取信息',
    'agent_email_send': '发送电子邮件',
    'generate_image': '根据文本生成图片',
    'generate_audio': '根据文本生成音频',
    'generate_digital_human_video': '生成数字人视频',
    'extract_text_from_image': '从图片中提取文本',
    'extract_content_information_from_image': '描述图片内容',
    'remove_image_background': '去除图片背景',
    'transcribe_audio': '将音频转成文字',
    'convert_audio_language': '将音频转换成另一种语言',
}


# ## 添加节点
workflow.add_node(supervisor_name, agent.supervisor_node)  # 添加主管节点
workflow.add_node(agent.agent_email_send_name, agent.email_node, metadata={"description": agent_descriptions["agent_email_send"]})  # 添加发送邮件代理节点
workflow.add_node(agent.agent_online_search_name, agent.research_node,metadata={"description": agent_descriptions["agent_online_search"]})  # 添加在线搜索代理节点
workflow.add_node(agent.agent_online_extract_name, agent.extract_node,metadata={"description": agent_descriptions["agent_online_extract"]})  # 添加在线提取数据代理节点
workflow.add_node(agent.agent_generate_image_name, agent.generate_image_node,metadata={"description": agent_descriptions["generate_image"]})  # 添加文本生成图片代理节点
workflow.add_node(agent.agent_generate_audio_name, agent.generate_audio_node,metadata={"description": agent_descriptions["generate_audio"]})  # 添加文本生成音频代理节点
workflow.add_node(agent.agent_generate_digital_human_video_name, agent.generate_digital_human_video_node,metadata={"description": agent_descriptions["generate_digital_human_video"]})  # 添加文字生成数字人代理节点
workflow.add_node(agent.agent_extract_text_from_image_name, agent.extract_text_from_image_node,metadata={"description": agent_descriptions["extract_text_from_image"]})  # 添加图片识别文字代理节点
workflow.add_node(agent.agent_extract_content_information_from_image_name,
                  agent.extract_content_information_from_image_node,metadata={"description": agent_descriptions["extract_content_information_from_image"]})  # 添加描述图片内容代理节点
workflow.add_node(agent.agent_remove_image_background_name, agent.remove_image_background_node,metadata={"description": agent_descriptions["remove_image_background"]})  # 添加去除图片背景代理节点
workflow.add_node(agent.agent_transcribe_audio_name, agent.transcribe_audio_node,metadata={"description": agent_descriptions["transcribe_audio"]})  # 添加音频转文字代理节点
workflow.add_node(agent.agent_convert_audio_language_name, agent.convert_audio_language_node,metadata={"description": agent_descriptions["convert_audio_language"]})  # 添加音频转另一种语言的音频代理节点

## 添加边
### 添加直接边
workflow.add_edge(START, supervisor_name)  # 从起始节点连接到主管节点
for agent_name in agent.agent_lst:  # 遍历代理列表
    workflow.add_edge(agent_name, supervisor_name)  # 每个代理连接到主管节点

### 添加条件边
condition_map = {k: k for k in agent.agent_lst}  # 创建条件映射，代理名称作为键值
condition_map['FINISH'] = END  # 将 FINISH 映射到结束节点
workflow.add_conditional_edges(supervisor_name, lambda x: x['next'], condition_map)  # 为主管节点添加条件边

graph = workflow.compile()  # 编译工作流图


from langchain_core.messages import HumanMessage  # 再次导入人类消息类

for s in graph.stream(
        {
            "messages": [  # 向图中发送初始消息
                # HumanMessage(content="Code hello world and print it to the terminal")
                # HumanMessage(content="search the weather in Pittsburgh online, and then send the information to my email ")
                HumanMessage(
                    content="成都最出名的动物？画出其图片")
            ]
        }
):
    if "__end__" not in s:  # 检查是否未到达结束标志
        print(s)  # 打印当前状态
        print('---')  # 打印分隔线
