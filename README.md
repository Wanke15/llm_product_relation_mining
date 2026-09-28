# 商品关系挖掘与审核

本地 Python + HTML 框架，默认读取 `data/prod_df.csv`。支持 OpenAI-compatible Chat Completions、TypeSafe Jev 原生 API 和离线演示后端。无需安装前端工具链，Python 3.11+、零第三方运行依赖；建议使用 uv。

## 开始

在项目目录执行（PowerShell）：

```powershell
# 仅首次创建；已有 .env 时不要覆盖
Copy-Item .env.example .env
# 编辑 .env 后运行
uv run python -m relations run
uv run python -m relations serve
```

打开 http://127.0.0.1:8765 。`serve` 保持运行，Ctrl+C 停止。也可直接用 `python` 替代 `uv run python`。

也可以用 `./start.ps1 -Limit 20` 一次运行并启动审核页；仅查看已有结果用 `./start.ps1 -ReviewOnly`。如果审核服务已经启动，无需重复启动，跑完新任务后在页面点“刷新”。uv 缓存放在项目 `.uv-cache/`，避免 Windows 用户缓存目录权限问题。

初始 `.env` 使用 `demo`，仅验证数据流，页面明确标注演示。切换真实后端：

```dotenv
RELATION_PROVIDER=llm
LLM_URL=https://YOUR-ENDPOINT/v1/chat/completions
LLM_API_KEY=YOUR_KEY
LLM_MODEL=YOUR_MODEL
```

或者：

```dotenv
RELATION_PROVIDER=jev
JEV_URL=https://api.typesafe.ai/v1/systemone
JEV_API_KEY=YOUR_KEY
JEV_MODEL=jev-latest
```

`.env` 支持空行、整行注释、KEY=value 和成对引号，不支持插值、多行值或行内注释。系统环境变量优先。密钥仅服务端使用，不进入页面或运行快照。`.env` 和 `runs/` 已加入 gitignore。

建议首次真实请求 `uv run python -m relations run --limit 5`，核对效果后扩大。默认 200 对、串行请求，避免无意全量产生费用。真实运行会把候选商品的文字字段发送到所配置服务。图片 URL 仅用于人工审核，不发送模型，浏览器加载图片时直接访问图片服务。

## 工作流

```powershell
# 离线演示
uv run python -m relations run --provider demo --limit 30
# 失败重试 / 中断恢复：已成功商品对不会再次调用
uv run python -m relations run --resume RUN_ID
# 指定候选：CSV 两列 a,b，均为 SPU ID
uv run python -m relations run --provider llm --pairs pairs.csv --limit 200
uv run python -m relations run --provider jev --pairs pairs.csv --limit 200
# 测试
uv run python -m unittest discover -s tests -v
```

审核页面可导出当前筛选的商品对 CSV，交给另一个后端得到同样的候选集合。页面选择对比运行后查看评分变化；数据快照不同会提示。自定义 CSV 也受 `--limit` 限制。默认候选为固定种子的同细类、同部门同风格跨类、随机对，排除后勤和无名称商品；均为候选而非真实标签，不是完整召回系统。数据量不足时可能少于请求数，以 manifest 的实际 pair_count 为准。后续可以替换 `core.candidates`，接入类目互补规则、向量召回或行为召回。

## 二阶段流水线（v2）

在原有 `run`/`serve` 一次性判定之外，新增「召回 → 初筛 → 精判」分阶段流水线。当前已落地 **Phase 1：商品向量化与三通道召回**；Jev Noul 初筛与 LLM 精判（Phase 2）尚未接入。

```powershell
# 1) 同步商品、生成缺失向量、重建索引（embedding 走缓存，不重复调用）
uv run python -m relations index

# 2) 三通道召回（similar / functional / style），不调用 Jev/LLM
uv run python -m relations retrieve --top-k 30              # 全量
uv run python -m relations retrieve --top-k 30 --limit 500  # 限候选商品对数量

# 3) 预览候选规模、缓存命中与覆盖（Phase 2 完整流水线在此展开）
uv run python -m relations pipeline --dry-run

# 4) 审核页（顶部新增「流水线统计 · 召回阶段」面板）
uv run python -m relations serve
```

数据与状态落在 SQLite（默认 `data/relations.db`），向量索引落 `data/index/`；历史 `runs/` 判定结果不受影响、`serve` 仍可查看。

- **embedding**：优先用 `.env` 的 OpenAI-compatible embeddings；未配置时回退本地确定性哈希 embedding（仅打通流程，非真实语义）。
- **候选有界**：每商品最多 `CANDIDATES_PER_PRODUCT=30` 个无序候选；相似 / 功能互补 / 风格搭配配额 10/10/10，停用通道的配额自动让渡给启用通道。
- **三个通道不同**：相似用商品自身向量；功能互补/风格搭配用独立构造的查询描述（互为“互补”“穿搭”目标），不把三通道做成同一向量重复查询。
- **覆盖**：功能互补/风格搭配依赖 `relations/config/complement_roles.json` 的角色映射；未映射的类目该通道不启用，`retrieve` 末尾报告各通道覆盖/未覆盖数。
- **缓存与失效**：商品 embedding 按「输入文本哈希 + 模型/版本」缓存；查询 embedding 按「描述哈希 + 模板版本 + 模型/版本」缓存；商品字段变化自动改变哈希使缓存失效。相同配置重复执行不再产生新的 embedding 调用。
- **状态区分**：召回候选是「待判定」，不是「有关系」；Phase 2 将把 初筛过滤 / 精判无关系 / 调用失败 分别记录。
- **性别过滤**：`relations/config/gender_rules.json` 按「部门|类目组」标注 女/男/中性（未命中默认中性）；召回后丢弃男女相反的候选（中性候选、同性别保留），三通道统一生效，实测过滤约 2.2 万条跨性别候选边。
- **百炼 Batch embedding**：设 `EMBEDDING_BATCH_MIN` 后，达到阈值的批量 embedding 改走 Batch 文件接口（`/v1/files` 上传 → `/v1/batches` → 轮询 → `/v1/files/{id}/content` 下载），费用约实时调用 50%；小批量仍走同步接口（`EMBEDDING_BATCH_SIZE` 控制每请求文本数）。

当前向量索引用 numpy 暴力余弦（12k 规模毫秒级）；hnswlib 在 Python3.11 Windows 无预编译 wheel、已实测不可装，`VectorIndex` 协议保留，后续有 wheel 可直接插回。

## 统一判定与可扩展性

- `relations/core.py`：SPU 输入规范、候选、证据校验、关系标签推导。
- `prompts/rubric.json`：两个后端共享的业务定义及评分口径；可直接编辑，运行保留其快照与哈希。
- `relations/providers.py`：`Judge` 协议，`judge(a,b) -> (统一结果, 原始响应)`；新增后端实现协议并注册 `REGISTRY`。执行、存储、页面均不依赖某个后端的响应格式。
- 三个独立维度：similarity / functional_complement / style_pairing；0 不成立、1 弱、2 明确但有条件、3 强、null 未知。
- Jev 一次请求三个 Choice 问题，选项包含 0/1/2/3/unknown。为了显式保留 unknown，本版不用连续 Score。保留每维概率分布与 confidence；不是把概率当强度，不生成不存在的解释。以后若改 Score，需要同时设计信息充分性判定和统一映射。
- LLM 请求 JSON 输出，解析后检查评分、字符串数组及证据是否等于原输入字段完整值。错误证据导致该对失败，不静默接受。自由文本理由的语义真实性仍需人工审核。
- 默认不指定 temperature，兼容不支持该参数的模型。端点须支持 Chat Completions；JSON 模式不支持时设 `LLM_RESPONSE_FORMAT=none`，仍要求 JSON 内容并校验。Responses API 或厂商私有协议需要另写适配器。
- 429/部分 5xx/超时自动有限退避；认证与格式错误不自动无限重试。失败单独记录，绝不记为“无关系”。运行结束存在错误时显示 completed_with_errors。

TypeSafe 原生接口核对来源：[API](https://docs.typesafe.ai/api)、[Choice](https://docs.typesafe.ai/primitives/choice)、[组合判定](https://docs.typesafe.ai/patterns/composite-scoring)。没有真实密钥时，只能验证请求合同和离线流程，不能声称模型质量或真实服务连通性已验证。

## 数据与审核

每个 `runs/<run_id>/` 保存：manifest（配置非敏感摘要、模型、时间、种子、哈希）、商品快照、候选、规则快照、逐对原子保存的 results、人工 reviews。恢复运行使用原快照，禁止更换模型和端点；修改规则后需新运行。同一 run 请勿启动多个写入进程。

HTML 支持：部门/类目/关系/审核状态/商品检索、可调关系阈值、评分分布、部门关系矩阵、分页双商品卡片、原始字段与响应、Jev 概率、单商品关系筛选、人工修正三维评分及备注、版本变化、结果与标注 JSON 导出。标注保存到服务端本地文件，刷新不丢失。

统计只代表候选样本。关系矩阵显示样本分母（含失败）；评分分布仅统计成功响应。人工一致率按有明确人工评分的维度计算，不能等同无偏总体准确率。“同意”可快速采用模型分数；需要改分时选“修正”。信息不足选“无法判断”。结果导出包含筛选结果及完整当前运行标注。

验证建议：对同一候选集比较两后端；随机抽取高分、低分与未知进行独立标注；按品类分层审查。AB 交换和重复运行稳定性可以直接用 Judge 协议编写专项实验，本版页面展示跨运行评分差异。尚未实现多模态判定、自动全量挖掘、统计显著性检验或线上推荐。

服务仅绑定本机 127.0.0.1，不提供文件目录浏览或配置读取；不应作为公网多人服务使用。
