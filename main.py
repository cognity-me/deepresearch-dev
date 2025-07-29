import asyncio
import os
import sys
import re
import json
from typing import Optional, Dict, List, Tuple, Set
from datetime import datetime
from dotenv import load_dotenv

from azure.ai.projects.aio import AIProjectClient
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import DeepResearchTool, MessageRole, ThreadMessage
from azure.identity.aio import DefaultAzureCredential
from openai import AsyncAzureOpenAI


class DeepResearchAssistant:
    """Advanced Deep Research Assistant with AI-powered conversational interface"""
    
    def __init__(self):
        self.research_context = {
            "topic": None,
            "scope": None,
            "depth": None,
            "format_preferences": None,
            "additional_requirements": [],
            "target_audience": None,
            "key_points": [],
            "constraints": []
        }
        self.conversation_history = []
        self.agents_client = None
        self.agent = None
        self.thread = None
        self.conversation_client = None  # GPT-4o-mini client for conversation
        self.project_client = None
        
    async def initialize(self):
        """Initialize the AI clients and agents"""
        load_dotenv()
        
        # Load environment variables
        try:
            # Deep Research variables
            project_endpoint = os.environ["PROJECT_ENDPOINT"]
            model_deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"]
            deep_research_model = os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"]
            bing_connection_name = os.environ["BING_RESOURCE_NAME"]
            
            # GPT-4o-mini variables for conversation
            gpt4o_mini_key = os.environ["GPT4O_MINI_API_KEY"]
            gpt4o_mini_endpoint = os.environ["GPT4O_MINI_ENDPOINT"]
            gpt4o_mini_deployment = os.environ["GPT4O_MINI_DEPLOYMENT_NAME"]
            
        except KeyError as e:
            print(f"❌ [오류] .env 파일에 다음 환경 변수가 설정되지 않았습니다: {e}")
            return False
            
        # Initialize conversation client (GPT-4o-mini)
        try:
            print("🔧 대화 시스템을 초기화하고 있습니다...")
            self.conversation_client = AsyncAzureOpenAI(
                api_key=gpt4o_mini_key,
                azure_endpoint=gpt4o_mini_endpoint,
                api_version="2025-01-01-preview"
            )
            self.gpt4o_mini_deployment = gpt4o_mini_deployment
            
        except Exception as e:
            print(f"❌ 대화 시스템 초기화 중 오류 발생: {e}")
            return False
            
        # Initialize Deep Research clients
        try:
            print("🔧 Deep Research 시스템을 초기화하고 있습니다...")
            self.project_client = AIProjectClient(
                endpoint=project_endpoint,
                credential=DefaultAzureCredential(),
            )
            
            # Get Bing connection
            bing_connection = await self.project_client.connections.get(name=bing_connection_name)
            
            # Define Deep Research tool
            deep_research_tool = DeepResearchTool(
                bing_grounding_connection_id=bing_connection.id,
                deep_research_model=deep_research_model,
            )
            
            # Create agent
            self.agents_client = self.project_client.agents
            self.agent = await self.agents_client.create_agent(
                model=model_deployment_name,
                name="advanced-research-assistant",
                instructions="""You are an advanced research assistant that helps users create comprehensive, 
                professional research reports. You excel at understanding user requirements through natural 
                conversation and producing high-quality, structured reports with tables, charts, and proper citations.""",
                tools=deep_research_tool.definitions,
            )
            
            # Create thread
            self.thread = await self.agents_client.threads.create()
                
            print("✅ 모든 시스템 초기화 완료!\n")
            return True
            
        except Exception as e:
            print(f"❌ Deep Research 시스템 초기화 중 오류 발생: {e}")
            return False
    
    def print_welcome_message(self):
        """Print welcome message and instructions"""
        print("=" * 80)
        print("🔬 Advanced Deep Research Assistant with AI Conversation 🔬".center(80))
        print("=" * 80)
        print("\n안녕하세요! 저는 당신의 AI 리서치 어시스턴트입니다.")
        print("자연스러운 대화를 통해 당신이 원하는 연구 주제를 파악하고,")
        print("전문적이고 깊이 있는 보고서를 작성해드립니다.\n")
        print("💡 사용 방법:")
        print("  - 연구하고 싶은 주제나 궁금한 점을 자유롭게 말씀해주세요")
        print("  - 대화를 통해 필요한 정보를 파악하겠습니다")
        print("  - 'exit' 또는 '종료'를 입력하면 프로그램을 종료합니다\n")
        print("-" * 80)
    
    async def get_user_input(self, prompt: str = "\n💬 당신: ") -> str:
        """Get user input with a custom prompt"""
        return await asyncio.get_event_loop().run_in_executor(None, input, prompt)
    
    async def get_ai_response(self, user_message: str) -> Tuple[str, Dict]:
        """Get AI response using GPT-4o-mini and function calling for structured context."""
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_research_context",
                    "description": "Updates the research context with information gathered from the user. Only call this when you have new or updated information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "The main topic of the research. e.g., '한국의 저출산 문제'"
                            },
                            "scope": {
                                "type": "string",
                                "description": "The scope or time period for the research. e.g., '2010년대 이후'"
                            },
                            "depth": {
                                "type": "string",
                                "description": "The desired depth of analysis. e.g., '심층 분석'"
                            },
                            "target_audience": {
                                "type": "string",
                                "description": "The target audience for the report. e.g., '정책 입안자'"
                            },
                            "is_ready": {
                                "type": "boolean",
                                "description": "Set to true only when enough information (at least topic and scope) is gathered to start the research."
                            }
                        },
                        "required": ["is_ready"]
                    }
                }
            }
        ]

        context_summary = []
        if self.research_context["topic"]:
            context_summary.append(f"주제: {self.research_context['topic']}")
        if self.research_context["scope"]:
            context_summary.append(f"범위: {self.research_context['scope']}")
        if self.research_context["depth"]:
            context_summary.append(f"깊이: {self.research_context['depth']}")
        if self.research_context["target_audience"]:
            context_summary.append(f"대상: {self.research_context['target_audience']}")

        current_context = "\n".join(context_summary) if context_summary else "아직 수집된 정보 없음"

        system_prompt = f"""You are a friendly and masterful research assistant. Your primary goal is to proactively guide the user to define their research needs through a natural, question-driven conversation in Korean.

        CURRENT COLLECTED INFORMATION:
        {current_context}

        YOUR TASK:
        1.  **Always ask a clear, guiding question.** Your response must always lead the conversation forward. Never be passive.
            - Good Example: "사회 문제에 대해 알아보고 싶으시군요. 혹시 특정 기간이나 범위에 관심이 있으신가요?"
            - Bad Example: "알겠습니다." or "정보를 업데이트했습니다."
        2.  Gather the following details one by one:
            - topic (주제) - REQUIRED
            - scope (범위/기간) - REQUIRED
            - depth (분석 깊이) - Optional
            - target_audience (대상) - Optional
        3.  **When you gather or update any information, you MUST call the `update_research_context` tool.** You must also provide a conversational response *in addition* to the tool call.
        4.  If the user's request is ambiguous (e.g., just "사회문제"), interpret it broadly for the context update (e.g., set topic to "한국 사회문제 전반"), and then immediately ask a clarifying question to narrow it down (e.g., ask about "저출산" or "고령화").
        5.  Once you have at least the topic and scope, you can set `is_ready` to `true` in your tool call.
        6.  Keep your conversational responses SHORT and TO THE POINT (1-2 sentences).
        7.  Do NOT ask about information you have already collected.
        """

        messages = [{"role": "system", "content": system_prompt}]
        for role, content in self.conversation_history[-6:]:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        try:
            response = await self.conversation_client.chat.completions.create(
                model=self.gpt4o_mini_deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=500
            )
            
            response_message = response.choices[0].message
            ai_response = response_message.content or ""
            context_data = {"is_ready": False}

            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                if tool_call.function.name == "update_research_context":
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        temp_context = {}
                        temp_context["topic"] = tool_args.get("topic")
                        temp_context["scope"] = tool_args.get("scope")
                        temp_context["depth"] = tool_args.get("depth")
                        temp_context["target_audience"] = tool_args.get("target_audience")
                        temp_context["is_ready"] = tool_args.get("is_ready", False)

                        context_data = temp_context
                        
                        if not ai_response:
                            ai_response = "더 구체적으로 설명해주시겠어요? 어떤 점이 궁금하신가요?"
                            
                    except json.JSONDecodeError:
                        print("⚠️ [경고] AI로부터 받은 도구 인자(tool arguments) 파싱에 실패했습니다.")
                        ai_response = "정보를 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요."
            
            return ai_response, context_data

        except Exception as e:
            print(f"❌ AI 응답 생성 중 오류 발생: {e}")
            return "죄송합니다. 다시 말씀해주시겠어요?", {"is_ready": False}
    
    def update_research_context_from_ai(self, context_data: Dict):
        """Update research context based on AI-extracted information"""
        # Update topic
        if context_data.get("topic"):
            # Always update if AI provides a topic (might be more specific)
            self.research_context["topic"] = context_data["topic"]
            
        # Update scope
        if context_data.get("scope"):
            # Always update if AI provides scope
            self.research_context["scope"] = context_data["scope"]
            
        # Update depth
        if context_data.get("depth") and context_data["depth"] not in ['None', 'string']:
            self.research_context["depth"] = context_data["depth"]
            
        # Update target audience
        if context_data.get("target_audience") and context_data["target_audience"] not in ['None', 'string']:
            self.research_context["target_audience"] = context_data["target_audience"]
            
        # Update format preferences
        if context_data.get("format_preferences"):
            self.research_context["format_preferences"] = context_data["format_preferences"]
            
        # Update key points
        if context_data.get("key_points"):
            # Add new key points without duplicates
            for point in context_data["key_points"]:
                if point and point not in self.research_context["key_points"]:
                    self.research_context["key_points"].append(point)
    
    def create_research_prompt(self) -> str:
        """Create a comprehensive research prompt based on collected information"""
        prompt_parts = []
        
        # Main topic
        if self.research_context["topic"]:
            prompt_parts.append(f"Research Topic: {self.research_context['topic']}")
        
        # Scope and timeframe
        if self.research_context["scope"]:
            prompt_parts.append(f"Scope and Timeframe: {self.research_context['scope']}")
            
        # Depth of analysis
        if self.research_context["depth"]:
            prompt_parts.append(f"Analysis Depth: {self.research_context['depth']}")
            
        # Target audience
        if self.research_context["target_audience"]:
            prompt_parts.append(f"Target Audience: {self.research_context['target_audience']}")
            
        # Key points to cover
        if self.research_context["key_points"]:
            prompt_parts.append(f"Key Points to Address: {', '.join(self.research_context['key_points'])}")
            
        # Professional report structure
        prompt_parts.append("""
Please create a comprehensive research report that includes:

1. **Executive Summary** (핵심 요약)
   - Key findings at a glance
   - Main recommendations

2. **Table of Contents** (목차)

3. **Introduction and Background** (서론 및 배경)
   - Context and importance
   - Research objectives

4. **Methodology** (연구 방법론)
   - Data sources and approach

5. **Main Findings** (주요 발견사항)
   - Organized in clear, logical sections
   - Use subheadings for different aspects

6. **Data Analysis** (데이터 분석)
   - Include relevant tables with clear headers
   - Describe trends and patterns
   - Use markdown tables for data presentation

7. **Visual Insights** (시각적 통찰)
   - Describe charts/graphs that would be helpful
   - Explain what each visualization would show

8. **Comparative Analysis** (비교 분석)
   - Compare different approaches/solutions/trends
   - Use comparison tables where appropriate

9. **Challenges and Limitations** (도전과제 및 한계)

10. **Conclusions and Recommendations** (결론 및 제언)
    - Clear, actionable insights
    - Future outlook

11. **References** (참고문헌)
    - All sources with proper citations
    - Use [1], [2], etc. for in-text citations

**Formatting Requirements:**
- Use clear markdown formatting with proper headers (##, ###)
- Include bullet points and numbered lists for clarity
- Create tables using markdown table syntax
- Bold important terms and concepts
- Include relevant statistics and data points
- Ensure all claims are backed by citations
""")
        
        # Special format requirements
        if self.research_context["format_preferences"]:
            prompt_parts.append(f"\nSpecial Format Requirements: {self.research_context['format_preferences']}")
            
        # Language preference (Korean-focused but with English terms where appropriate)
        prompt_parts.append("\nPlease write the report primarily in Korean, but use English technical terms where appropriate for clarity.")
            
        return "\n".join(prompt_parts)
    
    async def generate_research_plan(self) -> str:
        """Generates a detailed, context-aware research plan using an LLM call."""
        
        context = self.research_context
        prompt = f"""
        당신은 전문 리서치 기획자입니다. 아래 사용자가 수집한 연구 컨텍스트를 바탕으로, 전문적이고 상세한 연구 보고서 목차(계획)를 작성해주세요.

        **수집된 연구 컨텍스트:**
        - 주제: {context['topic']}
        - 범위: {context.get('scope', '전체')}
        - 분석 깊이: {context.get('depth', '표준')}
        - 대상 독자: {context.get('target_audience', '일반')}
        - 핵심 포인트: {', '.join(context['key_points']) if context['key_points'] else '지정되지 않음'}

        **지침:**
        1.  **목차 형식으로** 작성해주세요. (예: 1. 서론, 1-1. 연구 배경, 2. 본론...)
        2.  각 목차 항목에 **어떤 내용이 들어갈지 한두 문장으로 구체적인 설명**을 덧붙여주세요.
        3.  컨텍스트(주제, 범위 등)를 적극적으로 반영하여 **개인화된 계획**을 만들어주세요.
        4.  결과는 마크다운 형식의 번호 매기기 목록으로 출력해주세요.

        **예시:**
        1.  **서론**
            -   한국의 저출산 문제가 국가적 위기로 대두된 배경과 심각성을 설명합니다.
            -   본 연구의 목적과 범위를 명확히 제시합니다.
        2.  **지난 10년간 저출산 정책 분석**
            -   주요 정책들(예: 보육 지원, 주거 지원)을 시기별로 정리하고 내용을 분석합니다.
            -   각 정책의 성과와 한계를 통계 자료를 기반으로 평가합니다.
        
        이제 위의 내용을 바탕으로 연구 계획을 작성해주세요.
        """

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.conversation_client.chat.completions.create(
                model=self.gpt4o_mini_deployment,
                messages=messages,
                temperature=0.5,
            )
            plan = response.choices[0].message.content
            return plan
        except Exception as e:
            print(f"❌ 연구 계획 생성 중 오류 발생: {e}")
            return "연구 계획을 생성하는 데 실패했습니다. 기본 목차로 진행합니다."

    async def conduct_research(self, research_prompt: str) -> Optional[ThreadMessage]:
        """Conduct the actual deep research"""
        print("\n🔍 Deep Research를 시작합니다...")
        print("=" * 80)
        
        # Create message in thread
        await self.agents_client.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=research_prompt,
        )
        
        # Start the run
        run = await self.agents_client.runs.create(
            thread_id=self.thread.id,
            agent_id=self.agent.id
        )
        
        # Progress indicators
        progress_indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        progress_index = 0
        printed_step_ids: Set[str] = set()
        start_time = datetime.now()
        
        # Poll for completion
        while run.status in ("queued", "in_progress"):
            await asyncio.sleep(2)
            run = await self.agents_client.runs.get(thread_id=self.thread.id, run_id=run.id)
            
            # Show progress spinner
            elapsed = (datetime.now() - start_time).seconds
            minutes = elapsed // 60
            seconds = elapsed % 60
            
            if minutes > 0:
                time_str = f"{minutes}분 {seconds}초"
            else:
                time_str = f"{seconds}초"
                
            print(f"\r{progress_indicators[progress_index % len(progress_indicators)]} "
                  f"Deep Research 진행 중... ({time_str} 경과) | 상태: {run.status}", 
                  end="", flush=True)
            progress_index += 1
            
            # Check for and print new run steps (Chain of Thought)
            try:
                run_steps = self.agents_client.runs.steps.list(thread_id=self.thread.id, run_id=run.id)
                async for step in run_steps:
                    if step.id not in printed_step_ids:
                        # A new step is found, break from the spinner line
                        print("\n\n" + "="*80)
                        
                        step_type_korean = "알 수 없음"
                        if step.step_details:
                            step_type_korean = "툴 호출" if step.step_details.type == 'tool_calls' else "메시지 생성"

                        print(f"🧠 AI 작업 단계 포착 (상태: {step.status})")
                        print(f"   - 종류: {step_type_korean}")

                        # If the step involves tool calls, print their details
                        if step.step_details and step.step_details.type == 'tool_calls':
                            for tool_call in step.step_details.tool_calls:
                                if hasattr(tool_call, 'deep_research') and tool_call.deep_research:
                                    details = tool_call.deep_research
                                    print("   - 툴: Deep Research")
                                    # `details` is a dict-like object, print its key-value pairs
                                    for key, value in details.items():
                                        print(f"     - {key}: {str(value).strip()}")

                        print("="*80 + "\n")
                        sys.stdout.flush()
                        printed_step_ids.add(step.id)
            except Exception:
                # Do not break the main loop for logging errors
                pass
        
        print(f"\n✅ Deep Research 완료! (총 소요시간: {minutes}분 {seconds}초)")
        
        if run.status == "failed":
            print(f"❌ 리서치 실패: {run.last_error}")
            return None
            
        # Get final message
        final_message = await self.agents_client.messages.get_last_message_by_role(
            thread_id=self.thread.id,
            role=MessageRole.AGENT
        )
        
        return final_message
    
    def enhance_report_formatting(self, content: str) -> str:
        """Enhance the report with better formatting and structure"""
        # Add timestamp and metadata
        timestamp = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
        
        # Create professional header
        header = f"""<div align="center">

# 📊 Research Report

### {self.research_context['topic']}

---

**작성일**: {timestamp}  
**대상 독자**: {self.research_context.get('target_audience', '일반')}  
**분석 범위**: {self.research_context.get('scope', '포괄적')}  
**분석 깊이**: {self.research_context.get('depth', '표준')}

</div>

---

"""
        
        # Add CSS-like styling for better markdown rendering
        style_section = """
<style>
table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}
th {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    padding: 12px;
    text-align: left;
    border: 1px solid #ddd;
}
td {
    padding: 12px;
    border: 1px solid #ddd;
}
tr:nth-child(even) {
    background-color: #f2f2f2;
}
tr:hover {
    background-color: #e8f5e9;
}
.highlight {
    background-color: #fff3cd;
    padding: 10px;
    border-left: 4px solid #ffc107;
    margin: 10px 0;
}
.important {
    color: #d32f2f;
    font-weight: bold;
}
</style>

"""
        
        # Process content to enhance formatting
        enhanced_content = content
        
        # Add emoji indicators for sections
        section_emojis = {
            "Executive Summary": "📋",
            "핵심 요약": "📋",
            "Table of Contents": "📑",
            "목차": "📑",
            "Introduction": "🎯",
            "서론": "🎯",
            "Methodology": "🔬",
            "연구 방법론": "🔬",
            "Main Findings": "🔍",
            "주요 발견사항": "🔍",
            "Data Analysis": "📊",
            "데이터 분석": "📊",
            "Visual Insights": "📈",
            "시각적 통찰": "📈",
            "Comparative Analysis": "⚖️",
            "비교 분석": "⚖️",
            "Challenges": "⚠️",
            "도전과제": "⚠️",
            "Conclusions": "✅",
            "결론": "✅",
            "References": "📚",
            "참고문헌": "📚"
        }
        
        for section, emoji in section_emojis.items():
            enhanced_content = enhanced_content.replace(f"## {section}", f"## {emoji} {section}")
            enhanced_content = enhanced_content.replace(f"### {section}", f"### {emoji} {section}")
        
        return header + style_section + enhanced_content
    
    def save_report(self, message: ThreadMessage, filename: str = None) -> str:
        """Save the research report to a file"""
        if not message:
            return None
            
        # Generate filename
        if not filename:
            safe_topic = re.sub(r'[^\w\s-]', '', self.research_context["topic"])
            safe_topic = re.sub(r'[-\s]+', '-', safe_topic)[:50]  # Limit length
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{safe_topic}_{timestamp}.md"
        
        try:
            with open(filename, "w", encoding="utf-8") as fp:
                # Combine all text messages
                content = "\n\n".join([t.text.value.strip() for t in message.text_messages])
                
                # Enhance formatting
                enhanced_content = self.enhance_report_formatting(content)
                fp.write(enhanced_content)
                
                # Add citations section with better formatting
                if message.url_citation_annotations:
                    fp.write("\n\n---\n\n## 📚 참고문헌 (References)\n\n")
                    fp.write("| No. | Title | Source | URL |\n")
                    fp.write("|-----|-------|--------|-----|\n")
                    
                    seen_urls = set()
                    for i, ann in enumerate(message.url_citation_annotations, 1):
                        url = ann.url_citation.url
                        title = ann.url_citation.title or "Untitled"
                        if url not in seen_urls:
                            # Extract domain from URL for source
                            domain_match = re.search(r'https?://([^/]+)', url)
                            source = domain_match.group(1) if domain_match else "Web"
                            
                            fp.write(f"| [{i}] | {title[:50]}{'...' if len(title) > 50 else ''} | {source} | [Link]({url}) |\n")
                            seen_urls.add(url)
                            
                # Add footer
                fp.write(f"\n\n---\n\n")
                fp.write(f"<div align='center'>\n\n")
                fp.write(f"*이 보고서는 **Advanced Deep Research Assistant**에 의해 자동 생성되었습니다.*\n\n")
                fp.write(f"*Powered by Azure OpenAI GPT-4o & Deep Research*\n\n")
                fp.write(f"</div>")
                
            return filename
            
        except IOError as e:
            print(f"❌ 파일 저장 중 오류 발생: {e}")
            return None
    
    async def chat_loop(self):
        """Main chat loop for AI-powered interaction with user"""
        self.print_welcome_message()
        
        # Initial greeting - more concise
        print("\n🤖 AI 어시스턴트: 안녕하세요! 무엇을 연구하고 싶으신가요?")

        # 수정 요청 횟수 카운터
        modification_count = 0  # 최대 10회까지 허용
        
        while True:
            # Get user input
            user_input = await self.get_user_input()
            
            # Check for exit
            if user_input.lower() in ['exit', 'quit', '종료', '끝', '나가기']:
                print("\n👋 리서치 어시스턴트를 종료합니다. 좋은 하루 되세요!")
                break
            
            # Add to conversation history
            self.conversation_history.append(("user", user_input))
            
            # Get AI response and context extraction
            ai_response, context_data = await self.get_ai_response(user_input)
            
            # Update research context
            self.update_research_context_from_ai(context_data)
            
            # Add AI response to history
            self.conversation_history.append(("assistant", ai_response))
            
            # Print AI response
            print(f"\n🤖 AI 어시스턴트: {ai_response}")
            
            # Check if ready for research AND has topic
            if context_data.get("is_ready", False) and self.research_context["topic"]:
                
                # Loop for modifying the research plan
                while modification_count < 10:
                    print("\n" + "="*80)
                    print("📋 연구 계획 제안")
                    print("="*80)
                    
                    print(f"📌 주제: {self.research_context['topic']}")
                    print(f"📅 범위: {self.research_context.get('scope', '전체 기간')}")
                    print(f"📊 분석 깊이: {self.research_context.get('depth', '표준 분석')}")
                    print(f"👥 대상 독자: {self.research_context.get('target_audience', '일반 독자')}")
                    
                    if self.research_context['key_points']:
                        print(f"\n🔑 핵심 포인트:")
                        for i, point in enumerate(self.research_context['key_points'][:5], 1):
                            print(f"   {i}. {point}")
                    
                    print("\n📑 제안 목차:")
                    detailed_plan = await self.generate_research_plan()
                    print(detailed_plan)
                    print("="*80)
                    
                    user_choice = await self.get_user_input("\n✅ 이 계획으로 리서치를 시작할까요? (Enter: 시작, '수정': 수정하기): ")

                    if user_choice.lower() != '수정':
                        break  # Exit modification loop to start research
                    else:
                        modification_count += 1
                        if modification_count >= 10:
                            print("\n⚠️ 수정 요청이 10회를 초과했습니다. 마지막 계획으로 자동 진행합니다.")
                            break

                        mod_request = await self.get_user_input("\n🤖 AI 어시스턴트: 알겠습니다. 어떤 부분을 수정하고 싶으신가요? 자유롭게 말씀해주세요: ")
                        self.conversation_history.append(("user", mod_request))
                        ai_response, context_data = await self.get_ai_response(mod_request)
                        self.update_research_context_from_ai(context_data)
                        self.conversation_history.append(("assistant", ai_response))
                        print(f"\n🤖 AI 어시스턴트: {ai_response}")
                        # Continue modification loop
                
                # --- Start Research ---
                research_prompt = self.create_research_prompt()
                final_message = await self.conduct_research(research_prompt)
                
                if final_message:
                    filename = self.save_report(final_message)
                    if filename:
                        print(f"\n✅ 보고서가 성공적으로 생성되었습니다!")
                        print(f"📄 파일명: {filename}")
                        
                        print("\n" + "="*80)
                        print("📖 보고서 미리보기")
                        print("="*80)
                        
                        preview_text = "\n".join([t.text.value.strip() for t in final_message.text_messages])
                        preview_lines = preview_text[:800].split('\n')
                        for line in preview_lines[:15]:
                            if line.strip():
                                print(line)
                        
                        print("\n... (전체 내용은 파일을 확인하세요)")
                        print("="*80)
                
                # Ask if user wants to continue
                continue_input = await self.get_user_input("\n새로운 연구를 시작하시겠습니까? (yes/no): ")
                if continue_input.lower() in ['no', 'n', '아니오', '아니요']:
                    print("\n👋 감사합니다. 좋은 하루 되세요!")
                    break
                else:
                    self.research_context = {
                        "topic": None,
                        "scope": None,
                        "depth": None,
                        "format_preferences": None,
                        "additional_requirements": [],
                        "target_audience": None,
                        "key_points": [],
                        "constraints": []
                    }
                    self.conversation_history = []
                    modification_count = 0
                    print("\n🔄 새로운 연구를 시작합니다.")
                    print("\n🤖 AI 어시스턴트: 어떤 주제를 연구하고 싶으신가요?")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.agent and self.agents_client:
            try:
                await self.agents_client.delete_agent(self.agent.id)
                print("\n🧹 Agent 리소스가 정리되었습니다.")
            except Exception as e:
                print(f"⚠️ Agent 정리 중 오류가 발생했습니다: {e}")

        # Close clients
        if self.project_client:
            try:
                await self.project_client.close()
                print("🧹 Deep Research client 연결이 종료되었습니다.")
            except Exception as e:
                print(f"⚠️ Deep Research client 종료 중 오류가 발생했습니다: {e}")

        if self.conversation_client:
            try:
                await self.conversation_client.close()
                print("🧹 Conversation client 연결이 종료되었습니다.")
            except Exception as e:
                print(f"⚠️ Conversation client 종료 중 오류가 발생했습니다: {e}")


async def main():
    """Main entry point"""
    assistant = DeepResearchAssistant()
    
    # Initialize the system
    if not await assistant.initialize():
        return
    
    try:
        # Start the chat loop
        await assistant.chat_loop()
    finally:
        # Clean up
        await assistant.cleanup()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 