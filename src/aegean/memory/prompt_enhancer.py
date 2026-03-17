"""
Prompt enhancement utilities for integrating memory system with agents.

Provides:
- Prompt templates with memory context
- RAG-enhanced prompt generation
- Context formatting utilities
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from aegean.memory.global_memory import GlobalMemorySystem, MemoryContext


@dataclass
class PromptTemplate:
    """
    Prompt template with placeholders for memory context.
    
    Supports:
    - Knowledge base context
    - Historical cases
    - Agent performance
    - Custom variables
    """
    name: str
    template: str
    description: str
    variables: List[str]
    
    def render(self, **kwargs) -> str:
        """
        Render template with provided variables.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Rendered prompt string
        """
        rendered = self.template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            rendered = rendered.replace(placeholder, str(value))
        return rendered


class PromptEnhancer:
    """
    Prompt enhancer that integrates memory system with agent prompts.
    
    Features:
    - RAG-enhanced prompts
    - Template management
    - Context formatting
    - Domain-specific templates
    """
    
    def __init__(
        self,
        memory_system: GlobalMemorySystem,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize prompt enhancer.
        
        Args:
            memory_system: GlobalMemorySystem instance
            config: Configuration options
        """
        self.memory_system = memory_system
        self.config = config or {}
        
        # Load default templates
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        
        # General reasoning template
        self.templates["reasoning"] = PromptTemplate(
            name="reasoning",
            template="""你是一个专业的推理专家。

{memory_context}

【当前任务】
{task}

请提供你的推理过程和最终答案。

要求：
1. 参考组织知识库中的相关信息
2. 借鉴历史案例的经验
3. 给出清晰的推理链
4. 提供最终答案和置信度

你的回答：""",
            description="General reasoning with memory context",
            variables=["memory_context", "task"]
        )
        
        # Financial credit assessment template
        self.templates["credit_assessment"] = PromptTemplate(
            name="credit_assessment",
            template="""你是一个专业的信用评估分析师。

{memory_context}

【客户信息】
{customer_data}

【评估任务】
请根据客户信息和组织知识库，评估客户的信用等级。

评估维度：
1. 还款能力（收入、资产）
2. 还款意愿（信用历史）
3. 稳定性（工作、居住）
4. 负债情况

参考标准：
- AAA: 优秀（违约率<0.5%）
- AA: 良好（违约率0.5-1%）
- A: 中上（违约率1-2%）
- BBB: 中等（违约率2-5%）
- BB: 中下（违约率5-10%）
- B: 较差（违约率>10%）

请给出：
1. 信用等级（AAA/AA/A/BBB/BB/B）
2. 详细评估理由
3. 风险提示
4. 置信度（0-1）

你的评估：""",
            description="Credit assessment for financial domain",
            variables=["memory_context", "customer_data"]
        )
        
        # Fraud detection template
        self.templates["fraud_detection"] = PromptTemplate(
            name="fraud_detection",
            template="""你是一个专业的反欺诈分析师。

{memory_context}

【交易信息】
{transaction_data}

【检测任务】
请分析该交易是否存在欺诈风险。

检测维度：
1. 交易模式异常
2. 金额异常
3. 地理位置异常
4. 设备/IP异常
5. 行为模式异常

风险等级：
- 高风险：建议拒绝
- 中风险：建议人工审核
- 低风险：可以通过
- 无风险：正常交易

请给出：
1. 风险等级（高/中/低/无）
2. 异常特征说明
3. 建议措施
4. 置信度（0-1）

你的分析：""",
            description="Fraud detection for financial domain",
            variables=["memory_context", "transaction_data"]
        )
        
        # Consensus refinement template
        self.templates["consensus_refinement"] = PromptTemplate(
            name="consensus_refinement",
            template="""你是一个专业的推理专家，正在参与多Agent共识过程。

{memory_context}

【原始任务】
{task}

【其他Agent的答案】
{previous_solutions}

【精化任务】
请综合考虑其他Agent的答案和推理过程，给出你的精化答案。

要求：
1. 分析其他Agent答案的合理性
2. 指出可能的错误或不足
3. 综合各方观点
4. 给出你的最终答案
5. 说明你的置信度

你的精化答案：""",
            description="Consensus refinement with peer solutions",
            variables=["memory_context", "task", "previous_solutions"]
        )

        # ── Risk validator templates ──────────────────────────────────────────

        self.templates["risk_identity"] = PromptTemplate(
            name="risk_identity",
            template="""你是一个专业的身份验证专家（KYA - Know Your Agent/Account）。

{memory_context}

【待评估主体】
{subject_info}

【当前行为】
{action_info}

{pre_screen_notes}

请从身份验证角度评估该主体的风险，输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体风险信号]
分析: [详细的身份验证分析，2-4句话]""",
            description="Identity/KYA risk validator prompt",
            variables=["memory_context", "subject_info", "action_info", "pre_screen_notes"]
        )

        self.templates["risk_anomaly"] = PromptTemplate(
            name="risk_anomaly",
            template="""你是一个专业的异常行为检测专家。

{memory_context}

【主体背景】
{subject_info}

【当前行为环境】
{action_info}

{pre_screen_notes}

请从行为异常检测角度评估风险，输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体异常信号]
分析: [详细的行为异常分析，2-4句话]""",
            description="Behavioral anomaly detection validator prompt",
            variables=["memory_context", "subject_info", "action_info", "pre_screen_notes"]
        )

        self.templates["risk_compliance"] = PromptTemplate(
            name="risk_compliance",
            template="""你是一个专业的反洗钱(AML)合规专家。

{memory_context}

【合规背景信息】
{compliance_context}

【主体信息】
{subject_info}

【交易详情】
{action_info}

{pre_screen_notes}

请从AML合规角度评估，检查洗钱类型（分层、整合、拆分等），输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体AML风险信号]
分析: [详细的合规分析，2-4句话]""",
            description="AML compliance validator prompt",
            variables=["memory_context", "compliance_context", "subject_info", "action_info", "pre_screen_notes"]
        )

        self.templates["risk_amount"] = PromptTemplate(
            name="risk_amount",
            template="""你是一个专业的交易金额与频率风险分析师。

{memory_context}

【主体画像】
{subject_info}

【本次交易】
{action_info}

【近期行为】
{velocity_info}

{pre_screen_notes}

请从金额与频率风险角度评估，判断金额是否与主体背景相符，输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体风险信号]
分析: [详细分析，2-4句话]""",
            description="Amount and velocity risk validator prompt",
            variables=["memory_context", "subject_info", "action_info", "velocity_info", "pre_screen_notes"]
        )

        self.templates["risk_context"] = PromptTemplate(
            name="risk_context",
            template="""你是一个专业的上下文与推理链分析专家。

{memory_context}

【请求主体】
{subject_info}

【请求详情】
{action_info}

【推理轨迹 (trace_context)】
{trace_context}

{pre_screen_notes}

请从上下文一致性和推理合理性角度评估：
1. 推理链是否内部一致、逻辑通顺？
2. 当前操作是否与描述的目的相符？
3. 是否存在推理操纵或欺骗迹象？

输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体上下文风险信号]
分析: [详细分析，2-4句话]""",
            description="Context and reasoning trace validator prompt",
            variables=["memory_context", "subject_info", "action_info", "trace_context", "pre_screen_notes"]
        )

    async def enhance_prompt(
        self,
        task: str,
        template_name: str = "reasoning",
        category: Optional[str] = None,
        user_id: Optional[str] = None,
        custom_vars: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance a prompt with memory context.
        
        Args:
            task: Task description
            template_name: Template to use
            category: Knowledge category filter
            user_id: Optional user ID
            custom_vars: Custom template variables
            
        Returns:
            Enhanced prompt string
        """
        # Get template
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Retrieve memory context
        context = await self.memory_system.retrieve_context(
            query=task,
            user_id=user_id,
            category=category,
            include_knowledge=True,
            include_cases=True,
            include_performance=self.config.get("include_performance", False)
        )
        
        # Format memory context
        memory_context_str = context.format_for_prompt(
            max_docs=self.config.get("max_docs", 3),
            max_cases=self.config.get("max_cases", 2)
        )
        
        # Prepare template variables
        template_vars = {
            "task": task,
            "memory_context": memory_context_str if memory_context_str else "（暂无相关历史信息）"
        }
        
        # Add custom variables
        if custom_vars:
            template_vars.update(custom_vars)
        
        # Render template
        enhanced_prompt = template.render(**template_vars)
        
        return enhanced_prompt
    
    async def enhance_refinement_prompt(
        self,
        task: str,
        previous_solutions: List[Dict[str, Any]],
        category: Optional[str] = None
    ) -> str:
        """
        Enhance a refinement prompt with peer solutions.
        
        Args:
            task: Original task
            previous_solutions: List of solutions from other agents
            category: Knowledge category filter
            
        Returns:
            Enhanced refinement prompt
        """
        # Format previous solutions
        solutions_str = self._format_solutions(previous_solutions)
        
        # Enhance with memory context
        enhanced_prompt = await self.enhance_prompt(
            task=task,
            template_name="consensus_refinement",
            category=category,
            custom_vars={
                "previous_solutions": solutions_str
            }
        )
        
        return enhanced_prompt
    
    def add_template(self, template: PromptTemplate):
        """
        Add a custom template.
        
        Args:
            template: PromptTemplate to add
        """
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def _format_solutions(self, solutions: List[Dict[str, Any]]) -> str:
        """Format previous solutions for prompt."""
        if not solutions:
            return "（暂无其他Agent的答案）"
        
        formatted = []
        for i, sol in enumerate(solutions, 1):
            agent_id = sol.get("agent_id", f"Agent {i}")
            answer = sol.get("answer", "")
            reasoning = sol.get("reasoning", "")
            confidence = sol.get("confidence", 0.0)
            
            formatted.append(
                f"{i}. {agent_id}:\n"
                f"   答案: {answer}\n"
                f"   推理: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}\n"
                f"   置信度: {confidence:.2f}"
            )
        
        return "\n\n".join(formatted)


class MemoryAwareAgent:
    """
    Wrapper for agents to make them memory-aware.
    
    Automatically enhances prompts with memory context.
    """
    
    def __init__(
        self,
        base_agent: Any,
        prompt_enhancer: PromptEnhancer,
        template_name: str = "reasoning",
        category: Optional[str] = None
    ):
        """
        Initialize memory-aware agent.
        
        Args:
            base_agent: Base agent to wrap
            prompt_enhancer: PromptEnhancer instance
            template_name: Default template to use
            category: Default knowledge category
        """
        self.base_agent = base_agent
        self.prompt_enhancer = prompt_enhancer
        self.template_name = template_name
        self.category = category
    
    async def generate_solution(
        self,
        task: str,
        context: Optional[Dict] = None
    ):
        """
        Generate solution with memory-enhanced prompt.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Solution from base agent
        """
        # Enhance prompt
        enhanced_task = await self.prompt_enhancer.enhance_prompt(
            task=task,
            template_name=self.template_name,
            category=self.category
        )
        
        # Call base agent with enhanced prompt
        return await self.base_agent.generate_solution(
            task=enhanced_task,
            context=context
        )
    
    async def refine_solution(
        self,
        refinement_set: List[Any],
        context: Optional[Dict] = None
    ):
        """
        Refine solution with memory-enhanced prompt.
        
        Args:
            refinement_set: Previous solutions
            context: Additional context
            
        Returns:
            Refined solution from base agent
        """
        # Get original task from context
        task = context.get("task", "") if context else ""
        
        # Format previous solutions
        previous_solutions = [
            {
                "agent_id": sol.agent_id,
                "answer": sol.answer,
                "reasoning": sol.reasoning,
                "confidence": sol.confidence if hasattr(sol, "confidence") else 0.0
            }
            for sol in refinement_set
        ]
        
        # Enhance refinement prompt
        enhanced_task = await self.prompt_enhancer.enhance_refinement_prompt(
            task=task,
            previous_solutions=previous_solutions,
            category=self.category
        )
        
        # Call base agent with enhanced prompt
        return await self.base_agent.generate_solution(
            task=enhanced_task,
            context=context
        )
    
    @property
    def agent_id(self) -> str:
        """Get agent ID from base agent."""
        return self.base_agent.agent_id if hasattr(self.base_agent, "agent_id") else "unknown"
    
    def __getattr__(self, name):
        """Delegate other attributes to base agent."""
        return getattr(self.base_agent, name)

