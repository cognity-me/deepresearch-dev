#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Deep‑Research Assistant
────────────────────────────────
  * 자연스러운 대화로 연구 파라미터 수집
  * 연구 계획서(프롬프트) 확인 & 진행/수정(최대 10회)
  * Deep‑Research 수행 중 사고 과정(COT) 실시간 노출
  * 완료 후 Markdown 보고서 파일 저장
"""

import asyncio
import os
import re
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    DeepResearchTool,
    MessageRole,
    ThreadMessage,
)
from openai import AsyncAzureOpenAI

# 사용자‑경험 확보용 상수
MAX_MODIFICATIONS = 10
CONTEXT_KEYS = ["topic", "scope", "depth", "target_audience"]

# ──────────────────────────────────────────────────────────────
#  Deep‑Research Assistant Class
# ──────────────────────────────────────────────────────────────
class DeepResearchAssistant:
    """AI‑기반 고급 리서치 어시스턴트"""

    # ─────────────────── 초기화 ───────────────────
    def __init__(self):
        self.research_ctx: Dict[str, Optional[str | List[str]]] = {
            "topic": None,
            "scope": None,
            "depth": None,
            "target_audience": None,
            "key_points": [],
        }
        self.conversation_history: List[Tuple[str, str]] = []
        self.modification_count = 0

        # Azure / OpenAI 핸들들
        self.project_client: Optional[AIProjectClient] = None
        self.agents_client: Optional[AgentsClient] = None
        self.agent = None
        self.thread = None
        self.chat_client: Optional[AsyncAzureOpenAI] = None
        self.gpt4o_deployment = None

    async def initialize(self) -> bool:
        """환경 변수 읽고 각종 클라이언트 생성"""
        load_dotenv()

        try:
            # Deep‑Research
            self.PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
            self.DR_MODEL_DEPLOYMENT = os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"]
            self.BING_CONNECTION_NAME = os.environ["BING_RESOURCE_NAME"]
            self.MODEL_DEPLOYMENT_NAME = os.environ["MODEL_DEPLOYMENT_NAME"]

            # GPT‑4o‑mini (대화)
            gpt_key = os.environ["GPT4O_MINI_API_KEY"]
            gpt_endpoint = os.environ["GPT4O_MINI_ENDPOINT"]
            self.gpt4o_deployment = os.environ["GPT4O_MINI_DEPLOYMENT_NAME"]

        except KeyError as err:
            print(f"❌ .env 에 {err} 변수가 없습니다.")
            return False

        # 1) 대화용 GPT‑4o‑mini
        try:
            self.chat_client = AsyncAzureOpenAI(
                api_key=gpt_key,
                azure_endpoint=gpt_endpoint,
                api_version="2025-01-01-preview",
            )
        except Exception as e:
            print(f"❌ GPT‑4o‑mini 초기화 실패: {e}")
            return False

        # 2) Deep‑Research용 클라이언트/에이전트
        try:
            self.project_client = AIProjectClient(
                endpoint=self.PROJECT_ENDPOINT,
                credential=DefaultAzureCredential(),
            )
            async with self.project_client as pc:
                bing_conn = await pc.connections.get(name=self.BING_CONNECTION_NAME)
                dr_tool = DeepResearchTool(
                    bing_grounding_connection_id=bing_conn.id,
                    deep_research_model=self.DR_MODEL_DEPLOYMENT,
                )
                # agent 생성
                self.agents_client = pc.agents
                self.agent = await self.agents_client.create_agent(
                    model=self.MODEL_DEPLOYMENT_NAME,
                    name="advanced-research-assistant",
                    instructions=(
                        "You are an advanced research assistant that produces professional, "
                        "well‑structured research reports with full citations."
                    ),
                    tools=dr_tool.definitions,
                )
                # thread 생성
                self.thread = await self.agents_client.threads.create()

        except Exception as e:
            print(f"❌ Deep‑Research 초기화 실패: {e}")
            return False

        return True

    # ─────────────────── I/O 헬퍼 ───────────────────
    async def ainput(self, prompt: str = "") -> str:
        return await asyncio.get_event_loop().run_in_executor(None, input, prompt)

    def pretty_print_welcome(self):
        print("=" * 80)
        print("🔬 Advanced Deep‑Research Assistant 🔬".center(80))
        print("=" * 80)
        print("자연어 대화를 통해 연구 주제를 정의하고, 실제 심층 리서치까지 수행합니다.\n")
        print("• 종료하려면 'exit' 또는 '종료' 입력\n")

    # ─────────────────── 대화 단계 ───────────────────
    async def get_ai_response(self, user_msg: str) -> Tuple[str, Dict]:
        """GPT‑4o‑mini 사용, [RESPONSE]/[CONTEXT] 구조 강제"""

        # 1) 시스템 프롬프트에 현재까지의 컨텍스트 삽입
        current_ctx = "\n".join(
            f"{k}: {self.research_ctx[k]}" for k in CONTEXT_KEYS if self.research_ctx[k]
        ) or "정보 없음"

        sys_prompt = f"""
You are a concise Korean research‑needs assistant.

CURRENT_COLLECTED_INFO:
{current_ctx}

RULES:
1. 응답은 1‑2문장.
2. 한 번에 질문은 하나.
3. 이미 수집된 항목은 묻지 말 것.
4. topic과 scope가 있으면 is_ready=true.
5. 사용자 '진행' 입력 시 is_ready=true.
6. 대답은 반드시 [RESPONSE] … [/RESPONSE] 와
   [CONTEXT]{{...}}[/CONTEXT] JSON 블록 포함.

COLLECT_KEYS: topic, scope, depth(optional), target_audience(optional)

Respond in Korean.
"""

        # 2) 메시지 스택
        msgs = [{"role": "system", "content": sys_prompt}]
        for role, content in self.conversation_history[-6:]:
            msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": user_msg})

        # 3) 호출
        resp = await self.chat_client.chat.completions.create(
            model=self.gpt4o_deployment,
            messages=msgs,
            temperature=0.7,
            max_tokens=300,
        )
        full = resp.choices[0].message.content

        # 4) 파싱
        def extract(tag: str) -> str:
            m = re.search(rf"\[{tag}\](.*?)\[/\s*{tag}\]", full, re.S)
            return m.group(1).strip() if m else ""

        response_text = extract("RESPONSE") or full.strip()
        ctx_json_str = extract("CONTEXT") or "{}"

        try:
            ctx_data = json.loads(ctx_json_str)
        except json.JSONDecodeError:
            ctx_data = {}

        return response_text, ctx_data

    def update_ctx(self, ctx: Dict):
        for k in CONTEXT_KEYS:
            if ctx.get(k):
                self.research_ctx[k] = ctx[k]
        # key_points 병합
        if isinstance(ctx.get("key_points"), list):
            for p in ctx["key_points"]:
                if p and p not in self.research_ctx["key_points"]:
                    self.research_ctx["key_points"].append(p)

    # ─────────────────── 연구 프롬프트 생성 ───────────────────
    def build_research_prompt(self) -> str:
        parts = [
            f"Research Topic: {self.research_ctx['topic']}",
            f"Scope/Timeframe: {self.research_ctx['scope']}",
            f"Analysis Depth: {self.research_ctx.get('depth','표준')}",
            f"Target Audience: {self.research_ctx.get('target_audience','일반')}",
            "",
            "### 요구 사항",
            "1. Executive Summary",
            "2. Table of Contents",
            "3. Introduction & Background",
            "4. Methodology",
            "5. Main Findings",
            "6. Data Analysis (tables)",
            "7. Visual Insights (charts 설명)",
            "8. Comparative Analysis",
            "9. Challenges & Limitations",
            "10. Conclusions & Recommendations",
            "11. References (with citations)",
            "",
            "• 보고서는 주로 한국어, 필요 시 영문 기술용어 사용",
            "• 마크다운 포맷 철저히 준수",
        ]
        return "\n".join(parts)

    # ─────────────────── Deep‑Research 수행 ───────────────────
    async def run_deep_research(self, prompt: str) -> Optional[ThreadMessage]:
        """리서치 실행 & Chain‑of‑Thought 스트리밍"""
        print("\n🔍 Deep‑Research 시작\n" + "-" * 60)
        # 1. 사용자 메시지 삽입
        await self.agents_client.messages.create(
            thread_id=self.thread.id, role="user", content=prompt
        )
        # 2. 실행
        run = await self.agents_client.runs.create(
            thread_id=self.thread.id, agent_id=self.agent.id
        )

        last_msg_id = None
        start = datetime.now()

        # 3. 폴링
        while run.status in ("queued", "in_progress"):
            await asyncio.sleep(2)
            run = await self.agents_client.runs.get(
                thread_id=self.thread.id, run_id=run.id
            )

            # 새 COT 메시지 출력
            try:
                latest = await self.agents_client.messages.get_last_message_by_role(
                    thread_id=self.thread.id, role=MessageRole.AGENT
                )
                if latest and latest.id != last_msg_id:
                    last_msg_id = latest.id
                    # 모든 텍스트 합쳐 출력
                    content = "\n".join(t.text.value for t in latest.text_messages)
                    print(f"\n🧠 {content}\n" + "-" * 40)
            except Exception:
                pass

            # 진행상태 간단 표시
            elapsed = (datetime.now() - start).seconds
            sys.stdout.write(
                f"\r⏳ 진행 중… {elapsed//60:02d}:{elapsed%60:02d}  상태: {run.status} "
            )
            sys.stdout.flush()

        print()  # 줄바꿈

        if run.status == "failed":
            print(f"❌ 리서치 실패: {run.last_error}")
            return None

        final_msg = await self.agents_client.messages.get_last_message_by_role(
            thread_id=self.thread.id, role=MessageRole.AGENT
        )
        print("✅ Deep‑Research 완료!")
        return final_msg

    # ─────────────────── 보고서 저장 ───────────────────
    def save_markdown(self, msg: ThreadMessage) -> Optional[str]:
        if not msg:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(r"[^\w\s-]", "", self.research_ctx["topic"] or "report")
        fname = f"research_{safe_topic[:40]}_{timestamp}.md"

        body = "\n\n".join(t.text.value for t in msg.text_messages)
        header = (
            f"# 📊 Research Report – {self.research_ctx['topic']}\n"
            f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}\n"
            f"**Scope:** {self.research_ctx['scope']}  "
            f"**Depth:** {self.research_ctx.get('depth','표준')}  "
            f"**Audience:** {self.research_ctx.get('target_audience','일반')}\n\n---\n"
        )

        with open(fname, "w", encoding="utf-8") as fp:
            fp.write(header + body)

        return fname

    # ─────────────────── 메인 대화 루프 ───────────────────
    async def chat_loop(self):
        self.pretty_print_welcome()
        while True:
            user_in = await self.ainput("💬 당신: ")

            if user_in.lower() in {"exit", "종료"}:
                print("👋 종료합니다.")
                break

            # 저장
            self.conversation_history.append(("user", user_in))

            # GPT‑응답
            ai_resp, ctx = await self.get_ai_response(user_in)
            self.update_ctx(ctx)

            # 히스토리
            self.conversation_history.append(("assistant", ai_resp))
            print(f"\n🤖 {ai_resp}")

            # ready?
            if ctx.get("is_ready") and self.research_ctx["topic"]:
                await self.confirm_and_research()

    # ─────────────────── 계획서 확인 & 진행/수정 ───────────────────
    async def confirm_and_research(self):
        """연구 계획서 보여주고 진행 여부 확인, 수정 루프(≤10회)"""

        while True:
            print("\n" + "=" * 70)
            print("📋 연구 계획서")
            print("=" * 70)
            for k in CONTEXT_KEYS:
                print(f"{k.capitalize():>15}: {self.research_ctx.get(k) or '(없음)'}")
            if self.research_ctx["key_points"]:
                print("    Key Points :", ", ".join(self.research_ctx["key_points"]))
            print("=" * 70)

            choice = (await self.ainput("진행(p) / 수정(m) / 취소(c) > ")).lower().strip()
            if choice in {"p", "", "진행"}:
                # 실제 리서치 시작
                prompt = self.build_research_prompt()
                final_msg = await self.run_deep_research(prompt)
                fname = self.save_markdown(final_msg)
                if fname:
                    print(f"\n📄 보고서 저장: {fname}\n")
                return

            elif choice in {"m", "수정"}:
                if self.modification_count >= MAX_MODIFICATIONS:
                    print("⚠️ 수정 한도(10회)에 도달했습니다. 바로 진행합니다.")
                    prompt = self.build_research_prompt()
                    final_msg = await self.run_deep_research(prompt)
                    fname = self.save_markdown(final_msg)
                    if fname:
                        print(f"\n📄 보고서 저장: {fname}\n")
                    return
                self.modification_count += 1
                to_fix = await self.ainput(
                    f"수정할 내용을 말씀하세요 ({self.modification_count}/{MAX_MODIFICATIONS})> "
                )
                # 대화 히스토리에 'assistant'로 수정 지시 삽입 → 재질문 유도
                self.conversation_history.append(("user", to_fix))
                ai_resp, ctx = await self.get_ai_response(to_fix)
                self.update_ctx(ctx)
                self.conversation_history.append(("assistant", ai_resp))
                print(f"\n🤖 {ai_resp}")
                # 다시 루프
            else:
                print("취소되었습니다.")
                return

    # ─────────────────── 정리 ───────────────────
    async def cleanup(self):
        try:
            if self.agent:
                await self.agents_client.delete_agent(self.agent.id)
        except Exception:
            pass


# ─────────────────── 메인 엔트리 ───────────────────
async def main():
    assistant = DeepResearchAssistant()
    if not await assistant.initialize():
        return
    try:
        await assistant.chat_loop()
    finally:
        await assistant.cleanup()


if __name__ == "__main__":
    asyncio.run(main())