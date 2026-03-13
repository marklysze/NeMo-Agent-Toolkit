# AG2 Integration Status

## Framework Comparison

| Component | LangChain | LlamaIndex | CrewAI | Strands | **AG2** | AutoGen | Sem. Kernel | Agno | ADK | FastMCP |
|-----------|:---------:|:----------:|:------:|:-------:|:-------:|:-------:|:-----------:|:----:|:---:|:-------:|
| **LLM Wrappers** | ✓ | ✓ | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ | - |
| **Tool Wrapper** | ✓ | ✓ | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Callback Handler** | ✓ | ✓ | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ | - |
| **Agent Registration** | ✓ (6 types) | - | - | - | **✓** (1 type) | - | - | - | - | - |
| **Parser** | ✓ | ✓ | - | - | **-** | - | - | - | ✓ | - |
| **Embedder** | ✓ | ✓ | - | - | **-** | - | - | - | - | - |
| **Retriever** | ✓ | - | - | - | **-** | - | - | - | - | - |
| **Control Flow** | ✓ (Router, Sequential) | - | - | - | **-** | - | - | - | - | - |
| **Custom Tools** | ✓ (3) | - | - | - | **-** | - | - | ✓ (1) | - | - |

## Notes

- **AG2 is now on par with the standard integration pattern** (LLM + Tools + Callback + Agent), matching or exceeding CrewAI, Strands, AutoGen, Semantic Kernel, and Agno.
- AG2 is one of only two frameworks (alongside LangChain) with agent workflow registration.
- **LangChain** is the most feature-rich with 6 agent types, advanced control flow (routing, sequential execution), LangSmith integration, evaluation framework, and dataset loader.
- **Standard pattern** (CrewAI, Strands, AutoGen, Semantic Kernel): LLM + Tool Wrapper + Callback Handler only.
- **FastMCP** takes a different architectural approach as a front-end plugin for MCP protocol support rather than wrapping an agent framework.

## Remaining Gaps vs LangChain

- **Parser** — for eval/fine-tuning pipeline
- **Multi-agent control flow** — GroupChat, sequential chats
- **Embedder** — embedding model wrappers
- **Retriever** — RAG retriever integration
