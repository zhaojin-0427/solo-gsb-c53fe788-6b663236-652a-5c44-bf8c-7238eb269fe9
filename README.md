# 贝叶斯设备故障诊断 API

基于 **Python + FastAPI + SQLite + NumPy** 的离散贝叶斯网络精确推理服务。客户端登记带版本的条件概率表（CPT）模型后，可提交诊断请求：在**缺失观测**（部分证据）下用**变量消元**计算精确后验、最可能故障组合（MPE），以及指定 **do 干预**后的反事实差异。证据矛盾时返回**可定位的最小冲突项**且不保存任何半成品；同一模型版本 + 幂等键的重放返回**字节一致**的结果。

## 功能特性

- **模型登记**：5～20 个离散变量、有向无环边、逐变量 CPT；校验每行归一化（容差 1e-6）、图无环（Kahn 拓扑）、父状态组合全覆盖、概率状态与变量状态一致，所有错误带可定位路径（如 `cpts.TempHigh.rows[0]`）。
- **精确推理**：变量消元（min-fill 排序，字典序打破平局保证确定性）计算后验边缘、MPE（max-product + 回溯）、do 干预（图截断）后的后验与逐状态差异 `delta`。
- **证据矛盾**：`P(evidence)=0` 时返回 422 与最小冲突证据子集（逐项 `evidence.<变量>` 路径），请求不落库。
- **幂等重放**：诊断结果以规范化 JSON 字节存入 SQLite；同 `(model_id, model_version, idempotency_key)` 且请求体一致的重放原样返回已存字节（响应头 `X-Idempotent-Replay: true`）；键相同而请求体不同返回 409。
- **性能**：20 个变量的完整诊断（后验 + MPE + 干预）实测约 150ms，远低于 2 秒；每次诊断记录消元摘要（消元顺序、最大因子规模、耗时）并随结果返回、落库。

## 快速启动

### Docker Compose（一键）

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`，SQLite 数据持久化在命名卷 `bayes-data`（容器内 `/data/bayesdiag.db`）。交互式文档见 `http://localhost:8000/docs`。

### 本地运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 数据库路径可用环境变量覆盖，默认 ./data/bayesdiag.db
BAYES_DB_PATH=/tmp/bayesdiag.db uvicorn app.main:app
```

### 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/models` | 登记模型版本（同内容重复登记幂等返回 200，内容不同返回 409） |
| `GET` | `/models/{model_id}/versions/{version}` | 查询已登记模型 |
| `POST` | `/diagnoses` | 提交诊断（幂等键重放返回字节一致结果） |
| `GET` | `/diagnoses/{diagnosis_id}` | 按 ID 查询诊断结果（与提交时字节一致） |
| `GET` | `/health` | 健康检查 |

## 诊断请求示例

以下示例以一条泵组产线为例：两个故障根因（电机 `MotorFault`、轴承 `BearingFault`），三个症状（振动、温度、流量）。

### 1. 登记模型

```bash
curl -X POST http://localhost:8000/models \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "pump-line-a",
    "version": "1.0.0",
    "variables": [
      {"name": "MotorFault",   "states": ["ok", "fail"]},
      {"name": "BearingFault", "states": ["ok", "fail"]},
      {"name": "VibrationHigh","states": ["no", "yes"]},
      {"name": "TempHigh",     "states": ["no", "yes"]},
      {"name": "FlowLow",      "states": ["no", "yes"]}
    ],
    "edges": [
      ["MotorFault", "VibrationHigh"],
      ["BearingFault", "VibrationHigh"],
      ["MotorFault", "TempHigh"],
      ["BearingFault", "FlowLow"]
    ],
    "cpts": {
      "MotorFault":   {"parents": [], "rows": [{"when": {}, "probs": {"ok": 0.98, "fail": 0.02}}]},
      "BearingFault": {"parents": [], "rows": [{"when": {}, "probs": {"ok": 0.97, "fail": 0.03}}]},
      "VibrationHigh": {
        "parents": ["MotorFault", "BearingFault"],
        "rows": [
          {"when": {"MotorFault": "ok",   "BearingFault": "ok"},   "probs": {"no": 0.95, "yes": 0.05}},
          {"when": {"MotorFault": "ok",   "BearingFault": "fail"}, "probs": {"no": 0.30, "yes": 0.70}},
          {"when": {"MotorFault": "fail", "BearingFault": "ok"},   "probs": {"no": 0.20, "yes": 0.80}},
          {"when": {"MotorFault": "fail", "BearingFault": "fail"}, "probs": {"no": 0.02, "yes": 0.98}}
        ]
      },
      "TempHigh": {
        "parents": ["MotorFault"],
        "rows": [
          {"when": {"MotorFault": "ok"},   "probs": {"no": 0.90, "yes": 0.10}},
          {"when": {"MotorFault": "fail"}, "probs": {"no": 0.15, "yes": 0.85}}
        ]
      },
      "FlowLow": {
        "parents": ["BearingFault"],
        "rows": [
          {"when": {"BearingFault": "ok"},   "probs": {"no": 1.0,  "yes": 0.0}},
          {"when": {"BearingFault": "fail"}, "probs": {"no": 0.25, "yes": 0.75}}
        ]
      }
    }
  }'
```

响应（201）：`{"model_id":"pump-line-a","version":"1.0.0","spec_hash":"…","variables":5,"edges":4,"created_at":"…","replayed":false}`

### 2. 提交诊断

现场只观测到「振动高、温度高」（流量未观测——缺失观测），希望知道各故障的后验、最可能故障组合，以及「更换电机」（`do(MotorFault=ok)`）后其余变量分布的反事实差异：

```bash
curl -X POST http://localhost:8000/diagnoses \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "pump-line-a",
    "model_version": "1.0.0",
    "idempotency_key": "workorder-2026-10-02-0001",
    "evidence": {"VibrationHigh": "yes", "TempHigh": "yes"},
    "queries": {
      "posterior": ["MotorFault", "BearingFault"],
      "mpe": true,
      "interventions": [
        {"name": "replace_motor", "do": {"MotorFault": "ok"}, "observe": ["BearingFault", "FlowLow"]}
      ]
    }
  }'
```

响应要点（201，`queries.posterior` 留空则默认输出全部非证据变量）：

```json
{
  "diagnosis_id": "dg_69de285fc83f477f71d06e77",
  "p_evidence": 0.0205028,
  "posterior": {
    "MotorFault":   {"ok": 0.332198528981, "fail": 0.667801471019},
    "BearingFault": {"ok": 0.875246307821, "fail": 0.124753692179}
  },
  "mpe": {
    "assignment": {"MotorFault": "fail", "BearingFault": "ok", "FlowLow": "no"},
    "probability": 0.643424312777
  },
  "interventions": [
    {
      "name": "replace_motor",
      "do": {"MotorFault": "ok"},
      "p_evidence": 0.00695,
      "posterior": {"BearingFault": {"ok": 0.697841726619, "fail": 0.302158273381}, "FlowLow": {"no": 0.773381294964, "yes": 0.226618705036}},
      "delta": {"BearingFault": {"ok": -0.177404581203, "fail": 0.177404581203}, "FlowLow": {"no": -0.133053435902, "yes": 0.133053435902}}
    }
  ],
  "elimination": {"variables": 5, "evidence_count": 2, "tasks": [{"task": "p_evidence", "order": ["FlowLow", "BearingFault", "MotorFault"], "max_factor_entries": 4, "duration_ms": 0.404}, "…"], "total_duration_ms": 1.6}
}
```

- `mpe.assignment`：全部非证据变量的最可能联合取值（故障变量即其中的 `*Fault` 项），`probability` 为 `P(assignment | evidence)`。
- `interventions[].delta`：`P(x | evidence, do) − P(x | evidence)`，逐状态给出干预前后的反事实差异。
- `elimination.tasks`：每个计算任务（`p_evidence`、`posterior:<变量>`、`mpe`、`intervention:<名称>`）的消元顺序、因子乘法次数、最大因子变量数/条目数与耗时。

### 3. 幂等重放与结果查询

```bash
# 相同模型版本 + 幂等键 + 相同请求体：返回字节一致的结果（200，X-Idempotent-Replay: true）
curl -X POST http://localhost:8000/diagnoses -H 'Content-Type: application/json' -d @同一请求.json

# 按 ID 查询（与提交响应字节一致）
curl http://localhost:8000/diagnoses/dg_69de285fc83f477f71d06e77
```

同一幂等键搭配**不同**请求体返回 `409 idempotency_key_conflict`。

### 4. 证据矛盾（不保存半成品）

`FlowLow=yes` 在 `BearingFault=ok` 下概率为 0（确定性 CPT），提交矛盾证据：

```bash
curl -X POST http://localhost:8000/diagnoses \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "pump-line-a", "model_version": "1.0.0",
    "idempotency_key": "bad-1",
    "evidence": {"BearingFault": "ok", "FlowLow": "yes", "VibrationHigh": "yes"},
    "queries": {"posterior": ["MotorFault"]}
  }'
```

返回 422，给出**最小冲突证据子集**（`VibrationHigh` 与矛盾无关，被正确排除），且该请求不落库：

```json
{
  "detail": {
    "error": "evidence_contradiction",
    "message": "evidence has zero probability under the model; nothing was stored",
    "items": [
      {"path": "evidence.BearingFault", "variable": "BearingFault", "state": "ok",  "message": "part of a minimal zero-probability evidence set"},
      {"path": "evidence.FlowLow",      "variable": "FlowLow",      "state": "yes", "message": "part of a minimal zero-probability evidence set"}
    ]
  }
}
```

## 校验规则摘要

- 变量 5～20 个、名唯一；每变量 2～16 个互异状态。
- 边引用已登记变量、无自环、无重复，整体无环。
- 每个变量恰有一张 CPT；`parents` 与边推导的父集一致；行覆盖全部父状态组合且无重复；每行 `probs` 键与变量状态一致、取值在 [0,1]、归一化（|sum−1| ≤ 1e-6）。
- 诊断请求：证据/查询/干预引用的变量与状态必须存在；`do` 与证据变量互斥；对证据变量求后验会被拒绝（分布退化）。
- 所有校验错误以 `{"detail": {"error", "message", "items": [{"path", "message"}]}}` 返回，`path` 可定位到具体行/边/变量。

## 项目结构

```
app/
  main.py      # FastAPI 路由、幂等重放、错误处理
  network.py   # 模型规范校验（归一化/无环/状态完整）并构建网络
  bayes.py     # 因子运算、变量消元、MPE、do 干预、最小冲突定位
  db.py        # SQLite 持久化（models / diagnoses 两张表）
  schemas.py   # Pydantic 请求模型
  util.py      # 规范化 JSON、哈希、确定性浮点舍入
tests/test_api.py  # 27 个端到端测试（含暴力枚举对照与 20 变量性能测试）
Dockerfile, docker-compose.yml, requirements.txt
```

## 设计要点

- **字节一致重放**：响应体以 `sort_keys` 规范化 JSON 序列化后整体落库；重放与 `GET /diagnoses/{id}` 直接返回存储字节，浮点输出统一舍入到 12 位小数并消除 `-0.0`。
- **确定性推理**：消元顺序由 min-fill 启发式 + 变量名字典序平局决定，NumPy 运算顺序固定，相同输入必得相同输出。
- **诊断 ID 可推导**：`dg_<sha256(model_id|version|idempotency_key)[:24]>`，天然支持去重。
- **并发安全**：检查-插入竞态通过唯一约束 `(model_id, model_version, idempotency_key)` 兜底，冲突时回读已存结果返回。
