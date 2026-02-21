# Process Isolation & Sandboxing 2025-2026

## Key Sources

1. **2026 AI代理沙箱化：MicroVMs与gVisor隔离技术**
   - URL: https://northflank.com/blog/how-to-sandbox-ai-agents
   - MicroVMs (Firecracker, Kata) and gVisor for strong isolation

2. **NVIDIA代理工作流沙箱安全指导**
   - URL: https://developer.nvidia.com/blog/practical-security-guidance-for-sandboxing-agentic-workflows-and-managing-execution-risk/
   - Virtualization isolation, file access limits

3. **AI代理沙箱化：2026生产部署未解难题**
   - URL: https://www.softwareseni.com/understanding-ai-agent-sandboxing-why-production-deployment-remains-unsolved-in-2026/
   - Traditional process isolation insufficient, need hypervisor-level

4. **硬化运行时隔离保护Agentic AI系统**
   - URL: https://edera.dev/stories/securing-agentic-ai-systems-with-hardened-runtime-isolation
   - Embedded runtime isolation, prevent prompt injection

5. **Kubernetes开源Agent Sandbox安全部署**
   - URL: https://www.infoq.com/news/2025/12/agent-sandbox-kubernetes/
   - gVisor and Kata for isolated LLM code execution

## Key Insights

**Isolation Levels**:
- Container (Docker): Shared kernel, weak isolation
- gVisor: User-space kernel, medium isolation
- MicroVM (Firecracker, Kata): Hypervisor-level, strong isolation

**Pi-mono Approach**:
- Process isolation via `child_process.spawn()`
- Independent memory space per sub-agent
- No shared state between processes
- Simpler than container/VM but effective for context isolation

**2025-2026 Trends**:
- Move towards hypervisor-level isolation
- Sandboxing AI-generated code execution
- Defense-in-depth security models
